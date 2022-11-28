import os
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics

from utils.logger import write_to_file

# @tf.function
def train_step(model, x, y, optimizer, loss_fn, acc_metric, target_index=0):
    with tf.GradientTape() as tape:
        train_preds = model(x, training=True)
        loss_value = loss_fn(y[target_index], train_preds)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    acc_metric.update_state(y[target_index], train_preds)

    return loss_value

# @tf.function
def test_step(model, x, y, loss_fn, acc_metric, target_index=0):
    val_preds = model(x, training=False)
    acc_metric.update_state(y[target_index], val_preds)

    val_loss = loss_fn(y[target_index], val_preds)

    return val_loss

def test_model_singletask(model, val_ds, loss_fn, acc_metric, target_index=0):
    
    for X_batch_val, y_batch_val_age, y_batch_val_gender in val_ds:
            val_loss = test_step(model, X_batch_val, [y_batch_val_age, y_batch_val_gender], 
                loss_fn, acc_metric, target_index)

    val_acc = acc_metric.result()
    acc_metric.reset_states()

    return val_loss, np.array([val_acc])

def train_model_singletask(model, train_ds, val_ds, epochs=10, target_index=0,
                            optimizer=optimizers.SGD(), loss_fn=losses.SparseCategoricalCrossentropy(), 
                            acc_metric=metrics.SparseCategoricalAccuracy(), log_file=None, tensorboard_dir=None, tensorboard_prefix="val1",
                                        early_stopping_patience=5):

    best_model = None

    #Set up tensorboard

    #Track mean training losses
    mean_train_loss = metrics.Mean("train_loss", dtype=tf.float32)
    mean_val_loss = metrics.Mean("val_loss", dtype=tf.float32)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(tensorboard_dir, "train"))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tensorboard_dir, "val"))
    #end setup tensorboard

    i = 0
    for x, y, z in train_ds.take(5):
        with train_summary_writer.as_default():
            tf.summary.image("Training data examples", x, step=i, max_outputs=25)
            i += 1

    i = 0
    for x, y, z in val_ds.take(5):
        with test_summary_writer.as_default():
            tf.summary.image("Validation data examples", x, step=i, max_outputs=25)
            i += 1
    
    wait = 0
    best_val_acc = 0
    
    for epoch in range(epochs):
        
        print("Start of epoch %d" % (epoch + 1, ))
        start_time = time.time()

        for step, (X_batch_train, y_batch_train_age, y_batch_train_gender) in enumerate(train_ds):
            train_loss = train_step(model, X_batch_train, [y_batch_train_age, y_batch_train_gender], 
                        optimizer, loss_fn, acc_metric, target_index=target_index)
            mean_train_loss(train_loss)

            if step % 500 == 0 and step > 0:
                to_write = "Training loss at step %d: %.4f \n" % (step, float(train_loss))
                write_to_file(log_file, to_write)
                
                print(to_write, end="")
        
        train_acc = acc_metric.result()
        acc_metric.reset_states()

        to_write = "Training accuracy over epoch %d: %.4f \n" % (epoch+1, train_acc)
        print(to_write)
        write_to_file(log_file, to_write)

        with train_summary_writer.as_default():
            tf.summary.scalar(tensorboard_prefix + '_loss', mean_train_loss.result(), step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy', train_acc, step=epoch)
        mean_train_loss.reset_states()

        val_loss, val_accs = test_model_singletask(model, val_ds, loss_fn, acc_metric, target_index)
        mean_val_loss(val_loss)
        to_write = "Validation accuracy over epoch %d: %.4f\n" % (epoch + 1, val_accs[0])
        to_write += "Time taken for epoch %d: %.2fs\n\n" % (epoch+1, time.time() - start_time)
        print(to_write, end="")
        write_to_file(log_file, to_write)

        with test_summary_writer.as_default():
            tf.summary.scalar(tensorboard_prefix + '_loss', mean_val_loss.result(), step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy', val_accs[0], step=epoch)
        mean_val_loss.reset_states()

        if early_stopping_patience > 0:
            wait += 1
            if val_accs[0] > best_val_acc:
                best_val_acc = val_accs[0]
                wait = 0

            if wait >= early_stopping_patience:
                to_write = "Val loss has not improved over %d epochs. Stopping training...\n" % (early_stopping_patience)
                write_to_file(log_file, to_write)
                print(to_write)
                break

    return model, np.array([best_val_acc])
