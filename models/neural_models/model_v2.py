
import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt



def generator(noise, reuse=None):

    with tf.variable_scope("generator_model", reuse=reuse):
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        hidden_1 = tf.layers.dense(inputs=noise, units=1024 * 4 * 4, kernel_initializer=weight_init, use_bias=False, )
        hidden_1 = tf.layers.batch_normalization(hidden_1, axis=-1)
        hidden_1 = tf.nn.relu(hidden_1)

        hidden_2 = tf.reshape(hidden_1, shape=[-1, 4, 4, 1024])

        hidden_3 = tf.layers.conv2d_transpose(inputs=hidden_2, filters=512, kernel_size=5, strides=2,
                                              kernel_initializer=weight_init, use_bias=False, padding="same")
        hidden_3 = tf.layers.batch_normalization(hidden_3, axis=-1)
        hidden_3 = tf.nn.relu(hidden_3)

        hidden_4 = tf.layers.conv2d_transpose(inputs=hidden_3, filters=256, kernel_size=5, strides=2,
                                              kernel_initializer=weight_init,use_bias=False,padding="same", )
        hidden_4 = tf.layers.batch_normalization(hidden_4, axis=-1)
        hidden_4 = tf.nn.relu(hidden_4)

        hidden_5 = tf.layers.conv2d_transpose(inputs=hidden_4, filters=128, kernel_size=5, strides=2,
                                              kernel_initializer=weight_init, use_bias=False, padding="same",)
        hidden_5 = tf.layers.batch_normalization(hidden_5, axis=-1)
        hidden_5 = tf.nn.relu(hidden_5)

        op = tf.layers.conv2d_transpose(inputs=hidden_5, filters=3, kernel_size=5, strides=2,
                                        kernel_initializer=weight_init, use_bias=False, activation=tf.nn.tanh,
                                        padding="same",)
        return op


def discriminator(image, reuse = None):

    with tf.variable_scope("discriminator_model", reuse=reuse):

        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        hidden_1 = tf.layers.conv2d(inputs = image, filters=64, kernel_size=5, strides=2,
                                    kernel_initializer=weight_init, use_bias=False, padding = "same")
        hidden_1 = tf.nn.leaky_relu(hidden_1, alpha=0.2)

        hidden_2 = tf.layers.conv2d(inputs=hidden_1, filters=128, kernel_size=5, strides=2,
                                    kernel_initializer=weight_init, use_bias=False, padding="same")
        hidden_2 = tf.layers.batch_normalization(hidden_2, axis=-1)
        hidden_2 = tf.nn.leaky_relu(hidden_2, alpha=0.2)

        hidden_3 = tf.layers.conv2d(inputs=hidden_2, filters=256, kernel_size=5, strides=2,
                                    kernel_initializer=weight_init, use_bias=False, padding="same")
        hidden_3 = tf.layers.batch_normalization(hidden_3, axis=-1)
        hidden_3 = tf.nn.leaky_relu(hidden_3, alpha=0.2)

        hidden_4 = tf.layers.conv2d(inputs = hidden_3,
                                    filters = 512,
                                    kernel_size = 5,
                                    strides = 2,
                                    kernel_initializer = weight_init,
                                    use_bias = False,
                                    padding = "same"
                                  )
        hidden_4 = tf.layers.batch_normalization(hidden_4, axis = -1)
        hidden_4 = tf.nn.leaky_relu(hidden_4, alpha = 0.2)

        # flattening
        flatten = tf.layers.flatten(hidden_4)

        # output layer, units = 1
        output = tf.layers.dense(flatten, units = 1, activation = tf.nn.sigmoid)

        return output

