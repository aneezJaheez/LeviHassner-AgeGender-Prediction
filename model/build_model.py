import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, initializers

class MultiTaskHead(Model):
    def __init__(self, num_classes_1 = 8, num_classes_2 = 1, activation_1 = "softmax", 
                    activation_2 = "sigmoid", name_1 = "age", name_2 = "gender"):
        super(MultiTaskHead, self).__init__()

        self.head_1 = layers.Dense(
            units=num_classes_1,
            activation=activation_1,
            name=name_1 + "_head",
        )
        
        self.head_2 = layers.Dense(
            units=num_classes_2,
            activation=activation_2,
            name=name_2 + "_head",
        )

    def call(self, inputs):
        return [self.head_1(inputs), self.head_2(inputs)]

class LeviHassnerBackbone(Model):
    def __init__(self, weight_decay=5e-4, dropout_prob=0.5, include_head=False, 
                    num_classes=None, initializer="levi_hassner"):
        super(LeviHassnerBackbone, self).__init__()
        
        self.include_head = include_head

        bias_initializer_fc = "zeros"
        
        if initializer == "levi_hassner":
            kernel_regularizer = regularizers.L2(l2=weight_decay)
            kernel_initializer = tf.random_normal_initializer(stddev=1e-2)
            # bias_initializer_conv = tf.constant_initializer(0)
            bias_initializer_conv = None
        else:
            kernel_regularizer = None
            kernel_initializer = "glorot_uniform"
            bias_initializer_conv = None

        self.conv1 = layers.Conv2D(
            filters=96,
            kernel_size=[7, 7],
            strides=(4, 4),
            padding="valid",
            bias_initializer=bias_initializer_conv,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
            input_shape=(227, 227, 3),
            name="Conv1",
        )
        self.pool1 = layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding="valid",
            name="Pool1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=1, 
            epsilon=0.001, 
            momentum=0.9997,
            name="BN1"
        )

        self.conv2 = layers.Conv2D(
            filters=256, 
            kernel_size=[5,5],
            strides=[1,1],
            padding="same",
            bias_initializer=bias_initializer_conv,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
            name="Conv2",
        )
        self.pool2 = layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding="valid",
            name="Pool2",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=1, 
            epsilon=0.001, 
            momentum=0.9997,
            name="BN2"
        )

        self.conv3 = layers.Conv2D(
            filters=384, 
            kernel_size=[3,3],
            strides=[1,1],
            padding="same",
            bias_initializer=bias_initializer_conv,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
            name="Conv3",
        )
        self.pool3 = layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding="valid",
            name="BN3",
        )

        self.flat = layers.Flatten(name="Flatten1")

        self.fc1 = layers.Dense(
            units=512,
            bias_initializer=bias_initializer_fc,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
            name="FC1",
        )

        self.drop1 = layers.Dropout(
            rate=dropout_prob,
            name="Dropout1"
        )

        self.fc2 = layers.Dense(
            units=512,
            bias_initializer=bias_initializer_fc,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
            name="FC2",
        )

        self.drop2 = layers.Dropout(
            rate=dropout_prob,
            name="Dropout2",
        )

        if(include_head):
            self.fc3 = layers.Dense(
                units=num_classes,
                # bias_initializer=bias_initializer_fc,
                # kernel_initializer=kernel_initializer,
                # kernel_regularizer=kernel_regularizer,
                activation="softmax" if num_classes > 1 else "sigmoid",
                name="FC3",
            )

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flat(x)

        x = self.fc1(x)
        x = self.drop1(x, training=training)
        
        x = self.fc2(x)
        x = self.drop2(x, training=training)

        if(self.include_head):
            return self.fc3(x)
        else:
            return x


