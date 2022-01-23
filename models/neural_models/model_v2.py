#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from tensorflow.keras import Input
from tensorflow.keras import activations
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from models.neural_models.neural_model import NeuralModel
import numpy

import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt


def generator(z, output_channel_dim, training):
    with tf.variable_scope("generator", reuse=not training):
        # 8x8x1024
        fully_connected = tf.layers.dense(z, 8 * 8 * 1024)
        fully_connected = tf.reshape(fully_connected, (-1, 8, 8, 1024))
        fully_connected = tf.nn.leaky_relu(fully_connected)

        # 8x8x1024 -> 16x16x512
        trans_conv1 = tf.layers.conv2d_transpose(inputs=fully_connected,
                                                 filters=512,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv1")
        batch_trans_conv1 = tf.layers.batch_normalization(inputs=trans_conv1,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv1")
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1,
                                           name="trans_conv1_out")

        # 16x16x512 -> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(inputs=trans_conv1_out,
                                                 filters=256,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv2")
        batch_trans_conv2 = tf.layers.batch_normalization(inputs=trans_conv2,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv2")
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2,
                                           name="trans_conv2_out")

        # 32x32x256 -> 64x64x128
        trans_conv3 = tf.layers.conv2d_transpose(inputs=trans_conv2_out,
                                                 filters=128,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv3")
        batch_trans_conv3 = tf.layers.batch_normalization(inputs=trans_conv3,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv3")
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3,
                                           name="trans_conv3_out")

        # 64x64x128 -> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(inputs=trans_conv3_out,
                                                 filters=64,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv4")
        batch_trans_conv4 = tf.layers.batch_normalization(inputs=trans_conv4,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv4")
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4,
                                           name="trans_conv4_out")

        # 128x128x64 -> 128x128x3
        logits = tf.layers.conv2d_transpose(inputs=trans_conv4_out,
                                            filters=3,
                                            kernel_size=[5, 5],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.truncated_normal_initializer(
                                                stddev=WEIGHT_INIT_STDDEV),
                                            name="logits")
        out = tf.tanh(logits, name="out")
        return out


def discriminator(x, reuse):
    with tf.variable_scope("discriminator", reuse=reuse):
        # 128*128*3 -> 64x64x64
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv1')
        batch_norm1 = tf.layers.batch_normalization(conv1,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm1')
        conv1_out = tf.nn.leaky_relu(batch_norm1,
                                     name="conv1_out")

        # 64x64x64-> 32x32x128
        conv2 = tf.layers.conv2d(inputs=conv1_out,
                                 filters=128,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv2')
        batch_norm2 = tf.layers.batch_normalization(conv2,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm2')
        conv2_out = tf.nn.leaky_relu(batch_norm2,
                                     name="conv2_out")

        # 32x32x128 -> 16x16x256
        conv3 = tf.layers.conv2d(inputs=conv2_out,
                                 filters=256,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv3')
        batch_norm3 = tf.layers.batch_normalization(conv3,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm3')
        conv3_out = tf.nn.leaky_relu(batch_norm3,
                                     name="conv3_out")

        # 16x16x256 -> 16x16x512
        conv4 = tf.layers.conv2d(inputs=conv3_out,
                                 filters=512,
                                 kernel_size=[5, 5],
                                 strides=[1, 1],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv4')
        batch_norm4 = tf.layers.batch_normalization(conv4,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm4')
        conv4_out = tf.nn.leaky_relu(batch_norm4,
                                     name="conv4_out")

        # 16x16x512 -> 8x8x1024
        conv5 = tf.layers.conv2d(inputs=conv4_out,
                                 filters=1024,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv5')
        batch_norm5 = tf.layers.batch_normalization(conv5,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm5')
        conv5_out = tf.nn.leaky_relu(batch_norm5,
                                     name="conv5_out")

        flatten = tf.reshape(conv5_out, (-1, 8 * 8 * 1024))
        logits = tf.layers.dense(inputs=flatten,
                                 units=1,
                                 activation=None)
        out = tf.sigmoid(logits)
        return out, logits






class ModelsV1(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.generator_block = None
        self.discriminator_block = None
        self.create_neural_network()

    def create_neural_network():

        input_generator_block = Input(shape=(self.length_latency_space))
        input_discriminator_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))
        self.generator_block = self.create_generator_block(input_generator_block)
        self.discriminator_block = self.create_discriminator_block(input_discriminator_block)
        self.discriminator_block.trainable = True
        self.model = self.create_generative_adversarial_model(self.generator_block, self.discriminator_block)

        @staticmethod

    def create_generative_adversarial_model(generative_model, discriminator_model):

        discriminator_model.trainable = True
        adversarial_neural_network = Sequential()
        adversarial_neural_network.add(generative_model)
        adversarial_neural_network.add(discriminator_model)
        adversarial_neural_network.compile(loss='binary_crossentropy', optimizer='adam')
        adversarial_neural_network.summary()
        return adversarial_neural_network

    def create_generator(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        first_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = Activation(activations.relu)(first_convolution)

        second_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = Activation(activations.relu)(second_convolution)

        third_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = Activation(activations.relu)(third_convolution)

        fourth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution = Activation(activations.relu)(fourth_convolution)

        fifth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fourth_convolution)
        fifth_convolution = Activation(activations.relu)(fifth_convolution)

        sixth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fifth_convolution)
        sixth_convolution = Activation(activations.relu)(sixth_convolution)

        first_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(sixth_convolution)
        first_deconvolution = Activation(activations.relu)(first_deconvolution)

        interpolation = Add()([first_deconvolution, fifth_convolution])

        second_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        second_deconvolution = Activation(activations.relu)(second_deconvolution)

        interpolation = Add()([second_deconvolution, fourth_convolution])

        third_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        third_deconvolution = Activation(activations.relu)(third_deconvolution)

        interpolation = Add()([third_deconvolution, third_convolution])

        fourth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fourth_deconvolution = Activation(activations.relu)(fourth_deconvolution)

        interpolation = Add()([fourth_deconvolution, second_convolution])

        fifth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fifth_deconvolution = Activation(activations.relu)(fifth_deconvolution)

        interpolation = Add()([fifth_deconvolution, first_convolution])

        sixth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        sixth_deconvolution = Activation(activations.relu)(sixth_deconvolution)

        interpolation = Add()([sixth_deconvolution, input_layer_block])

        convolution_model = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(interpolation)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(convolution_model)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2D(1, (1, 1))(convolution_model)

        convolution_model = Model(input_layer_block, convolution_model)
        convolution_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model = convolution_model

    def create_discriminator_block(self, input_layer_block):

        first_convolution = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = LeakyReLU()(first_convolution)

        second_convolution = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = LeakyReLU()(second_convolution)

        third_convolution = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = LeakyReLU()(third_convolution)

        fourth_convolution = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution = LeakyReLU()(fourth_convolution)

        fifth_convolution = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(fourth_convolution)
        fifth_convolution = LeakyReLU()(fifth_convolution)

        neural_discriminator_dense_block = Flatten()(fifth_convolution)

        neural_discriminator_dense_block = Dense(1)(neural_discriminator_dense_block)
        neural_discriminator_dense_block = Activation(activations.sigmoid)(neural_discriminator_dense_block)

        convolution_model = Model(input_layer_block, neural_discriminator_dense_block)
        convolution_model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=self.metrics)
        convolution_model.summary()
        return convolution_model

    @staticmethod
    def check_feature_empty(list_feature_samples):

        number_true_samples = 0

        for i in list_feature_samples:

            for j in i:

                if int(j) == 1:
                    number_true_samples += 1

        if number_true_samples > 0:
            return 1

        return 0

    def remove_empty_features(self, x_training, y_training):

        x_training_list = []
        y_training_list = []

        for i in range(len(x_training)):

            if self.check_feature_empty(x_training[i]):
                x_training_list.append(x_training[i])
                y_training_list.append(y_training[i])

        return numpy.array(x_training_list), numpy.array(y_training_list)

    def training(self, x_training, y_training, evaluation_set):

        x_training, y_training = self.remove_empty_features(x_training, y_training)

        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)
            batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.model.fit(x=batch_training_in, y=batch_training_out, epochs=1, verbose=1)

            if i % 10 == 0:
                feature_predicted = self.model.predict(batch_training_in[0:10])
                self.save_image_feature(feature_predicted[0], batch_training_out[0], batch_training_in[0], i)

        return 0

        @staticmethod

    def check_feature_empty(list_feature_samples):

        number_true_samples = 0

        for i in list_feature_samples:

            for j in i:

                if int(j) == 1:
                    number_true_samples += 1

        if number_true_samples > 0:
            return 1

        return 0

    def remove_empty_features(self, x_training, y_training=None):

        print('Checking empty feature')
        x_training_list = []
        y_training_list = []

        for i in range(len(x_training)):

            if self.check_feature_empty(x_training[i]):

                x_training_list.append(x_training[i])

                if y_training is not None:
                    y_training_list.append(y_training[i])

        return numpy.array(x_training_list), numpy.array(y_training_list)

    def get_synthetic_sample(self, image_file):

        return self.generator_block.predict(image_file)

    def calibration(self, x_training, y_training):

        # x_training, y_training = self.remove_empty_features(x_training, y_training)

        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)

            batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            batch_training_out = self.get_feature_batch(y_training, random_array_feature)

            label_real_image = numpy.ones(len(random_array_feature))
            label_synthetic_image = numpy.zeros(len(random_array_feature))

            synthetic_image = self.get_synthetic_sample(numpy.array(batch_training_in))

            real_images = batch_training_out

            input_samples_training = numpy.concatenate((synthetic_image, real_images), axis=0)

            output_samples_training = numpy.concatenate((label_synthetic_image, label_real_image), axis=0)

            output_samples_training = numpy.reshape(output_samples_training, (len(output_samples_training), 1))

            loss_disc = self.discriminator_block.train_on_batch(input_samples_training, output_samples_training)

            loss_gen = self.model.train_on_batch(numpy.array(batch_training_in), numpy.array(label_real_image))

            print('Epoch {}  Loss Discriminator {}   Loss Generator {}'.format(i, loss_disc, loss_gen))

            if i % 100 == 0:
                feature_predicted = self.get_synthetic_sample(numpy.array(batch_training_in[0:2]))
                self.save_image_feature(feature_predicted[0], feature_predicted[0], feature_predicted[0], i)

        return 0

        def create_latency_noise(self):

            return numpy.array(numpy.random.uniform(size=self.length_latency_space))


def training(self, x_training, y_training=None, evaluation_set=None):
    x_training, y_training = self.remove_empty_features(x_training, y_training)

    for i in range(self.epochs):

        random_array_feature = self.get_random_batch(x_training)
        batch_training_in = [self.create_latency_noise() for i in range(len(random_array_feature))]
        batch_training_out = self.get_feature_batch(y_training, random_array_feature)

        label_real_image = ones(len(random_array_feature), 1)
        label_synthetic_image = zeros(len(random_array_feature), 0)

        synthetic_image = self.get_synthetic_sample(batch_training_in)
        real_images = batch_training_out

        input_samples_training = synthetic_image + real_images
        output_samples_training = label_synthetic_image + label_real_image
        loss_disc = self.discriminator_block.train_on_batch(input_samples_training, output_samples_training)
        loss_gen = self.model.train_on_batch(batch_training_in, label_real_image)
        print('Epoch {}  Loss Discriminator {}   Loss Generator {}'.format(i, loss_disc, loss_gen))

        if i % 50 == 0:
            feature_predicted = self.get_synthetic_sample(batch_training_in[0:2])
            self.save_image_feature(feature_predicted[0], feature_predicted[0], feature_predicted[0], i)

    return 0


