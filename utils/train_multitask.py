import time
import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics

from utils.logger import write_to_file

# @tf.function
def train_step(model, x, y, optimizer, loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2):
    with tf.GradientTape() as tape:
        train_preds = model(x, training=True)
        loss_value = loss_fn_1(y[0], train_preds[0])
        loss_value += loss_fn_2(y[1], train_preds[1])

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    acc_metric_1.update_state(y[0], train_preds[0])
    acc_metric_2.update_state(y[1], train_preds[1])

    return loss_value

# @tf.function
def test_step(model, x, y, loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2):
    val_preds = model(x, training=False)
    acc_metric_1.update_state(y[0], val_preds[0])
    acc_metric_2.update_state(y[1], val_preds[1])

    val_loss = loss_fn_1(y[0], val_preds[0])
    val_loss += loss_fn_2(y[1], val_preds[1])

    return val_loss

def test_model_multitask(model, val_ds, loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2):
    
    for X_batch_val, y_batch_val_age, y_batch_val_gender in val_ds:
            val_loss = test_step(model, X_batch_val, [y_batch_val_age, y_batch_val_gender], loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2)

    val_acc_1 = acc_metric_1.result()
    val_acc_2 = acc_metric_2.result()
    acc_metric_1.reset_states()
    acc_metric_2.reset_states()

    return val_loss, np.array([val_acc_1, val_acc_2])

def train_model_multitask(model, train_ds, val_ds, epochs=10,
                            optimizer=optimizers.SGD(), loss_fn_1=losses.SparseCategoricalCrossentropy(),
                                loss_fn_2=losses.BinaryCrossentropy(), acc_metric_1=metrics.SparseCategoricalAccuracy(),
                                    acc_metric_2=metrics.BinaryAccuracy(), log_file=None, tensorboard_dir=None, tensorboard_prefix="fold_1",
                                        early_stopping_patience=-1):

    best_model = None
    best_accs = None

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

    if early_stopping_patience > 0:
        wait = 0
        best_val_loss = 0
    
    for epoch in range(epochs):
        
        print("Start of epoch %d" % (epoch + 1, ))
        start_time = time.time()

        for step, (X_batch_train, y_batch_train_age, y_batch_train_gender) in enumerate(train_ds):
            train_loss = train_step(model, X_batch_train, [y_batch_train_age, y_batch_train_gender], 
                        optimizer, loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2)
            mean_train_loss(train_loss)


            if step % 500 == 0 and step > 0:
                to_write = "Combined training loss at step %d: %.4f \n" % (step, float(train_loss))
                write_to_file(log_file, to_write)
                
                print(to_write, end="")
        
        train_acc_1 = acc_metric_1.result()
        train_acc_2 = acc_metric_2.result()
        acc_metric_1.reset_states()
        acc_metric_2.reset_states()

        to_write = "Training accuracy on task 1 over epoch %d: %.4f \n" % (epoch+1, train_acc_1)
        to_write += "Training accuracy on task 2 over epoch %d: %.4f \n" % (epoch+1, train_acc_2)
        print(to_write)
        write_to_file(log_file, to_write)

        with train_summary_writer.as_default():
            tf.summary.scalar(tensorboard_prefix + '_loss', mean_train_loss.result(), step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy_age', train_acc_1, step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy_gender', train_acc_2, step=epoch)
        mean_train_loss.reset_states()

        val_loss, val_accs = test_model_multitask(model, val_ds, loss_fn_1, loss_fn_2, acc_metric_1, acc_metric_2)
        mean_val_loss(val_loss)
        to_write = "Validation accuracy on task 1 over epoch %d: %.4f\n" % (epoch + 1, val_accs[0])
        to_write += "Validation accuracy on task 2 over epoch %d: %.4f\n" % (epoch + 1, val_accs[1])
        to_write += "Time taken for epoch %d: %.2fs\n\n" % (epoch+1, time.time() - start_time)
        print(to_write, end="")
        write_to_file(log_file, to_write)

        with test_summary_writer.as_default():
            tf.summary.scalar(tensorboard_prefix + '_loss', mean_val_loss.result(), step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy_age', val_accs[0], step=epoch)
            tf.summary.scalar(tensorboard_prefix + '_accuracy_gender', val_accs[1], step=epoch)
        mean_val_loss.reset_states()

        if best_accs is None:
            best_accs = val_accs
        if val_accs[0] > best_accs[0] and val_accs[1] > best_accs[1]:
            best_accs = val_accs

        if early_stopping_patience > 0:
            wait += 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0

            if wait >= early_stopping_patience:
                print("Val loss has not improved over %d epochs. Stopping training..." % (early_stopping_patience))
                break

    return model, best_accs
