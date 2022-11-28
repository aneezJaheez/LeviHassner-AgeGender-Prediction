import os

from functools import partial
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 227

def decode_jpeg(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [256, 256, 3])
    return image

def distort_image(image, height, width):
  distorted_image = tf.image.random_crop(image, [height, width, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
  return distorted_image

def data_normalization(image):
    image = tf.image.per_image_standardization(image)
    return image

def image_preprocessing(image_buffer, image_size, train):
    image = decode_jpeg(image_buffer)
    
    if train:
        image = distort_image(image, image_size, image_size)
    else:
        image = tf.image.per_image_standardization(image)
        
    image = data_normalization(image)
    return image

def get_data_filenames(base_dir, split):
    if split not in ["train", "validation", "val"]:
        print("Invalid data split")
        exit(-1)

    data_prefix = os.path.join(base_dir, "%s-*"%split)
    filenames = tf.io.gfile.glob(data_prefix)
    
    print("TFRecord files for training: \n", filenames)

    if not filenames:
        print("No Files found in specified data directory")
        exit(-1)

    return filenames

def parse_example_proto(train, example_serialized):
  # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label_age': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        
        'image/class/label_gender': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        # 'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
        #                                         default_value=-1),
        'image/height': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                            default_value=-1),
        'image/width': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                            default_value=-1),

    }

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label_age = tf.cast(features['image/class/label_age'], dtype=tf.int32)
    label_gender = tf.cast(features['image/class/label_gender'], dtype=tf.int32)
    # label = tf.cast(features['image/class/label'], dtype=tf.int32)
    image = features["image/encoded"]
    # filename = features["image/filename"]

    image_processed = image_preprocessing(image, image_size=IMAGE_SIZE, train=train)

    return image_processed, label_age, label_gender
    # return image_processed, label

def build_dataset(data_dir, batch_size=4, image_size=227):
    train_filenames = get_data_filenames(data_dir, "train")
    val_filenames = get_data_filenames(data_dir, "val")
    # val_filenames = get_data_filenames(data_dir, "validation")

    train_ds = tf.data.TFRecordDataset(train_filenames)
    val_ds = tf.data.TFRecordDataset(val_filenames)

    train_ds = train_ds.map(partial(parse_example_proto, True))
    val_ds = val_ds.map(partial(parse_example_proto, True))

    train_ds_batched = train_ds.batch(batch_size)
    val_ds_batched = val_ds.batch(batch_size)

    return train_ds_batched, val_ds_batched