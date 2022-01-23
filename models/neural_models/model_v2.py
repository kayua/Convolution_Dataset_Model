#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import logging

from keras import activations
from keras.layers import Add
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow import random
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
from tensorflow import shape
from tensorflow import GradientTape

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.optimizers import Adam

from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

from models.neural_models.neural_model import NeuralModel
import numpy


class ModelsV1(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.generator_block = None
        self.discriminator_block = None
        self.adversarial_block = None
        self.create_neural_network()

    def create_neural_network(self):

        self.generator_block = self.create_generator_block()
        self.discriminator_block = self.create_discriminator_block()
        self.adversarial_block = self.create_generative_adversarial_model(self.generator_block,
                                                                          self.discriminator_block)

    def create_generative_adversarial_model(self, generative_model, discriminator_model):

        adversarial_block = AdversarialClass(discriminator_model, generative_model, self.length_latency_space)
        adversarial_block.compile(d_optimizer=Adam(learning_rate=0.0001), loss_fn=BinaryCrossentropy(),
                                  g_optimizer=Adam(learning_rate=0.0001), )
        return adversarial_block

    def create_generator_block(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        first_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = Activation(activations.relu)(first_convolution)
        first_convolution = BatchNormalization()(first_convolution)

        second_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = Activation(activations.relu)(second_convolution)
        second_convolution = BatchNormalization()(second_convolution)

        third_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = Activation(activations.relu)(third_convolution)
        third_convolution = BatchNormalization()(third_convolution)

        fourth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution = Activation(activations.relu)(fourth_convolution)
        fourth_convolution = BatchNormalization()(fourth_convolution)

        fifth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fourth_convolution)
        fifth_convolution = Activation(activations.relu)(fifth_convolution)
        fifth_convolution = BatchNormalization()(fifth_convolution)

        sixth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fifth_convolution)
        sixth_convolution = Activation(activations.relu)(sixth_convolution)
        sixth_convolution = BatchNormalization()(sixth_convolution)


        first_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(sixth_convolution)
        first_deconvolution = Activation(activations.relu)(first_deconvolution)
        first_deconvolution = BatchNormalization()(first_deconvolution)

        interpolation = Add()([first_deconvolution, fifth_convolution])

        second_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        second_deconvolution = Activation(activations.relu)(second_deconvolution)
        second_deconvolution = BatchNormalization()(second_deconvolution)

        interpolation = Add()([second_deconvolution, fourth_convolution])

        third_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        third_deconvolution = Activation(activations.relu)(third_deconvolution)
        third_deconvolution = BatchNormalization()(third_deconvolution)

        interpolation = Add()([third_deconvolution, third_convolution])

        fourth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fourth_deconvolution = Activation(activations.relu)(fourth_deconvolution)
        fourth_deconvolution = BatchNormalization()(fourth_deconvolution)

        interpolation = Add()([fourth_deconvolution, second_convolution])

        fifth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fifth_deconvolution = Activation(activations.relu)(fifth_deconvolution)
        fifth_deconvolution = BatchNormalization()(fifth_deconvolution)

        interpolation = Add()([fifth_deconvolution, first_convolution])

        sixth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        sixth_deconvolution = Activation(activations.relu)(sixth_deconvolution)
        sixth_deconvolution = BatchNormalization()(sixth_deconvolution)

        interpolation = Add()([sixth_deconvolution, input_layer_block])

        generator_model_block = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(interpolation)
        generator_model_block = Activation(activations.relu)(generator_model_block)

        generator_model_block = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(generator_model_block)
        generator_model_block = Activation(activations.relu)(generator_model_block)

        generator_model_block = Conv2D(1, (1, 1))(generator_model_block)

        generator_model_block = Model(input_layer_block, generator_model_block)

        return Model(input_layer_block, generator_model_block, name="generator")

    @staticmethod
    def create_discriminator_block():

        input_layer_block = Input(shape=(256, 256, 1))
        discriminator_block_model = Conv2D(128, kernel_size=4, strides=2, padding="same")(input_layer_block)
        discriminator_block_model = LeakyReLU(alpha=0.2)(discriminator_block_model)
        discriminator_block_model = BatchNormalization()(discriminator_block_model)
        discriminator_block_model = Dropout(0.2)(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=4, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU(alpha=0.2)(discriminator_block_model)
        discriminator_block_model = BatchNormalization()(discriminator_block_model)
        discriminator_block_model = Dropout(0.2)(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=4, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU(alpha=0.2)(discriminator_block_model)
        discriminator_block_model = BatchNormalization()(discriminator_block_model)
        discriminator_block_model = Dropout(0.2)(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=4, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU(alpha=0.2)(discriminator_block_model)
        discriminator_block_model = BatchNormalization()(discriminator_block_model)
        discriminator_block_model = Dropout(0.2)(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=4, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU(alpha=0.2)(discriminator_block_model)
        discriminator_block_model = BatchNormalization()(discriminator_block_model)
        discriminator_block_model = Dropout(0.2)(discriminator_block_model)

        discriminator_block_model = Flatten()(discriminator_block_model)
        discriminator_block_model = Dense(1)(discriminator_block_model)
        discriminator_block_model = Activation(activations='sigmoid')(discriminator_block_model)

        return Model(input_layer_block, discriminator_block_model, name='discriminator')

    @staticmethod
    def check_feature_empty(list_feature_samples):

        number_true_samples = 0

        for collum_feature in list_feature_samples:

            for trace_peer_position in collum_feature:

                if int(trace_peer_position) == 1:
                    number_true_samples += 1

        if number_true_samples > 0:
            return 1

        return 0

    def remove_empty_features(self, x_training, y_training=None):

        logging.info('Starting Checking empty features')
        x_training_list, y_training_list = [], []

        for i in tqdm(range(len(x_training))):

            if self.check_feature_empty(x_training[i]):

                x_training_list.append(x_training[i])

                if y_training is not None:
                    y_training_list.append(y_training[i])

        logging.info('End checking empty features')
        return numpy.array(x_training_list), numpy.array(y_training_list)

    def get_synthetic_sample(self, image_file):

        return self.generator_block.predict(image_file)

    def calibration(self, x_training, y_training):

        logging.info('Start calibration model')
        save_image_callback = self.ImageGeneratorCallback(self.length_latency_space)
        self.adversarial_block.fit(x_training, epochs=self.epochs, callbacks=[save_image_callback])
        logging.info('End calibration model')
        return 0

    def create_latency_noise(self):

        return numpy.array(numpy.random.uniform(size=self.length_latency_space))

    def training(self, x_training, y_training=None, evaluation_set=None):

        logging.info('Starting training model')
        save_image_callback = self.ImageGeneratorCallback(self.length_latency_space)
        self.adversarial_block.fit(x_training, epochs=self.epochs, callbacks=[save_image_callback])
        logging.info('End training model')
        return 0


class AdversarialClass(keras.Model):

    def __init__(self, discriminator, generator, number_latency_points):

        super(AdversarialClass, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = number_latency_points

    def compile(self, d_optimizer, g_optimizer, loss_fn):

        super(AdversarialClass, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):

        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):

        number_images_per_batch = tf.shape(real_images)[0]
        latency_matrix = random.normal(shape=(number_images_per_batch, self.latent_dim))
        synthetic_image_generated = self.generator(latency_matrix)
        batch_images_input = concat([synthetic_image_generated, real_images], axis=0)
        list_labels = concat([ones((number_images_per_batch, 1)), zeros((number_images_per_batch, 1))], axis=0)
        list_labels += 0.05 * random.uniform(shape(list_labels))

        with GradientTape() as gradient_tape:
            list_images_predicted = self.discriminator(batch_images_input)
            discriminator_loss = self.loss_fn(list_labels, list_images_predicted)

        update_gradient_function = gradient_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(update_gradient_function, self.discriminator.trainable_weights))

        latency_matrix = random.normal(shape=(number_images_per_batch, self.latent_dim))
        misleading_labels = zeros((number_images_per_batch, 1))

        with GradientTape() as gradient_tape:
            list_images_predicted = self.discriminator(self.generator(latency_matrix))
            generator_loss = self.loss_fn(misleading_labels, list_images_predicted)

        update_gradient_function = gradient_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(update_gradient_function, self.generator.trainable_weights))
        self.d_loss_metric.update_state(discriminator_loss)
        self.g_loss_metric.update_state(generator_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result(), }


