from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from six.moves import xrange
from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf
import json

tf.compat.v1.disable_eager_execution()

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256

flags.DEFINE_string('fold_dir', 'C:/Users/Ryan Tay/Desktop/UNIVERSITY STUFF/Yr 4 Sem 1/CZ4042 Neural Network & Deep Learning/Group Assignment/nndl_2/celeba/',
                           'Fold directory')

flags.DEFINE_integer('val_fold', 0,
                           'Validation fold')

flags.DEFINE_string('data_dir', 'C:/Users/Ryan Tay/Desktop/UNIVERSITY STUFF/Yr 4 Sem 1/CZ4042 Neural Network & Deep Learning/Group Assignment/img_align_celeba/',
                           'Data directory')

flags.DEFINE_string('output_dir', 'C:/Users/Ryan Tay/Desktop/UNIVERSITY STUFF/Yr 4 Sem 1/CZ4042 Neural Network & Deep Learning/Group Assignment/nndl_2/celeba/output/',
                           'Output directory')

flags.DEFINE_integer('train_shards', 10,
                            'Number of shards in training TFRecord files.')

flags.DEFINE_integer('valid_shards', 2,
                            'Number of shards in validation TFRecord files.')

flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')


FLAGS = flags.FLAGS

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label_1, label_2, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    height: integer, image height in pixels
    width: integer, image width in pixels
    Returns:
    Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label_age': _int64_feature(label_1),
        'image/class/label_gender': _int64_feature(label_2),
        'image/filename': _bytes_feature(str.encode(os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }))
    return example
    
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.compat.v1.Session()
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        cropped = tf.image.resize(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH])
        cropped = tf.cast(cropped, tf.uint8) 
        self._recoded = tf.image.encode_jpeg(cropped, format='rgb', quality=100)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def resample_jpeg(self, image_data):
        image = self._sess.run(self._recoded, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        return image
        

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def _process_image(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.compat.v1.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.resample_jpeg(image_data)
    return image, RESIZE_HEIGHT, RESIZE_WIDTH

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               labels_1, labels_2, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
    analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    
    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label_1 = int(labels_1[i])
            label_2 = int(labels_2[i])

            image_buffer, height, width = _process_image(filename, coder)
            
            example = _convert_to_example(filename, image_buffer, label_1, label_2,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, labels_1, labels_2, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(labels_1)
    assert len(filenames) == len(labels_2)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, labels_1, labels_2, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

def find_image_files(fold_dir, data_dir, mode="train"):
    labels_age = []
    labels_gender = []
    filenames = []
    age_index = 5
    gender_index = 20

    for file in os.listdir(fold_dir):
        if 'age' in file:
            filepath = os.path.join(fold_dir, file)
            age_labels = [l.strip().split(' ') for l in tf.compat.v1.gfile.FastGFile(
                filepath, 'r').readlines()]
            for line in age_labels[1:]:
                jpeg_file_path = "%s/%s" % (data_dir, line[0])
                if not os.path.exists(jpeg_file_path):
                    continue
                filenames.append(jpeg_file_path)
                labels_age.append(line[age_index])
        elif 'gender' in file:
            filepath = os.path.join(fold_dir, file)
            gender_labels = [l.strip().split(' ') for l in tf.compat.v1.gfile.FastGFile(
                filepath, 'r').readlines()]
            count = 0
            for line in gender_labels[2:]:
                line = list(filter(('').__ne__, line))
                jpeg_file_path = "%s/%s" % (data_dir, line[0])
                if not os.path.exists(jpeg_file_path):
                    continue
                filenames.append(jpeg_file_path)
                labels_gender.append(line[gender_index])   

    shuffled_index = list(range(len(labels_age)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    if mode == 'train':
        shuffled_index = shuffled_index[:round(0.7 * len(labels_age))]
    elif mode == 'val':
        shuffled_index = shuffled_index[round(0.7 * len(labels_age)):]
  
    filenames = list(set(filenames))
    filenames = [filenames[i] for i in shuffled_index]
    labels_age = [labels_age[i] for i in shuffled_index]
    labels_gender = [labels_gender[i] for i in shuffled_index]
    
    unique_labels = set(labels_age)
    #print(set(labels_age))
    #print(set(labels_gender))
    #print('Found %d JPEG files across %d labels inside %s.' %
    #       (len(filenames), len(unique_labels), data_dir))
    return filenames, labels_age, labels_gender

def _process_dataset(name, filename, directory, num_shards, mode):
    """Process a complete data set and save it as a TFRecord.
    Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
    """
    filenames, labels_age, labels_gender = find_image_files(filename, directory, mode=mode)

    _process_image_files(name, filenames, labels_age, labels_gender, num_shards)

    unique_labels = set(labels_age)
    return len(labels_age), unique_labels

def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.valid_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.valid_shards')
    print('Saving results to %s' % FLAGS.output_dir)

    if os.path.exists(FLAGS.output_dir) is False:
        print('creating %s' % FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    # Run it!
    valid, valid_outcomes = _process_dataset("val", FLAGS.fold_dir, FLAGS.data_dir, FLAGS.valid_shards, mode="val")

    train, train_outcomes = _process_dataset("train", FLAGS.fold_dir, FLAGS.data_dir, FLAGS.train_shards, mode="train")
    
    if len(valid_outcomes) != len(valid_outcomes | train_outcomes):
        print('Warning: unattested labels in training data [%s]' % (', '.join((valid_outcomes | train_outcomes) - valid_outcomes)))
        
    output_file = os.path.join(FLAGS.output_dir, 'md.json')


    md = { 'num_valid_shards': FLAGS.valid_shards, 
           'num_train_shards': FLAGS.train_shards,
           'valid_counts': valid,
           'train_counts': train,
           'timestamp': str(datetime.now()),
           'nlabels': len(train_outcomes) }
    with open(output_file, 'w') as f:
        json.dump(md, f)


if __name__ == '__main__':
    tf.compat.v1.app.run()

