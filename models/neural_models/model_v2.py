#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import logging
import tensorflow.keras
from tensorflow.keras import activations
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import keras
from tensorflow import random
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
from tensorflow import shape
from tensorflow import GradientTape

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

from models.neural_models.neural_model import NeuralModel
import numpy


class ModelsV2(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.generator_block = None
        self.discriminator_block = None
        self.adversarial_block = None
        self.learning_rate = args.learning_rate
        self.create_neural_network()

    def create_neural_network(self):

        self.generator_block = self.create_generator_block()
        self.generator_block.summary()
        self.discriminator_block = self.create_discriminator_block()
        self.discriminator_block.summary()
        self.adversarial_block = self.create_generative_adversarial_model(self.generator_block,
                                                                          self.discriminator_block)

    def create_generative_adversarial_model(self, generative_model, discriminator_model):

        adversarial_block = AdversarialClass(discriminator_model, generative_model)
        adversarial_block.compile(d_optimizer=Adam(learning_rate=self.learning_rate), loss_fn=BinaryCrossentropy(),
                                  g_optimizer=Adam(learning_rate=self.learning_rate), )
        return adversarial_block

    def create_generator_block(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        first_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = Activation(activations.relu)(first_convolution)
        first_convolution = BatchNormalization()(first_convolution)

        second_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = Activation(activations.relu)(second_convolution)
        second_convolution = BatchNormalization()(second_convolution)

        third_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = Activation(activations.relu)(third_convolution)
        third_convolution = BatchNormalization()(third_convolution)

        fourth_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution = Activation(activations.relu)(fourth_convolution)
        fourth_convolution = BatchNormalization()(fourth_convolution)

        fifth_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(fourth_convolution)
        fifth_convolution = Activation(activations.relu)(fifth_convolution)
        fifth_convolution = BatchNormalization()(fifth_convolution)

        sixth_convolution = Conv2D(120, (3, 3), strides=(2, 2), padding='same')(fifth_convolution)
        sixth_convolution = Activation(activations.relu)(sixth_convolution)
        sixth_convolution = BatchNormalization()(sixth_convolution)

        first_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(sixth_convolution)
        first_deconvolution = Activation(activations.relu)(first_deconvolution)
        first_convolution = BatchNormalization()(first_convolution)

        interpolation = Add()([first_deconvolution, fifth_convolution])

        second_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(interpolation)
        second_deconvolution = Activation(activations.relu)(second_deconvolution)

        interpolation = Add()([second_deconvolution, fourth_convolution])

        third_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(interpolation)
        third_deconvolution = Activation(activations.relu)(third_deconvolution)

        interpolation = Add()([third_deconvolution, third_convolution])

        fourth_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fourth_deconvolution = Activation(activations.relu)(fourth_deconvolution)

        interpolation = Add()([fourth_deconvolution, second_convolution])

        fifth_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fifth_deconvolution = Activation(activations.relu)(fifth_deconvolution)

        interpolation = Add()([fifth_deconvolution, first_convolution])

        sixth_deconvolution = Conv2DTranspose(120, (3, 3), strides=(2, 2), padding='same')(interpolation)
        sixth_deconvolution = Activation(activations.relu)(sixth_deconvolution)

        interpolation = Add()([sixth_deconvolution, input_layer_block])

        convolution_model = Conv2DTranspose(120, (3, 3), strides=(1, 1), padding='same')(interpolation)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2DTranspose(120, (3, 3), strides=(1, 1), padding='same')(convolution_model)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2D(1, (1, 1))(convolution_model)

        return Model(input_layer_block, convolution_model, name="generator")

    def create_discriminator_block(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        discriminator_block_model = Conv2D(128, kernel_size=3, strides=2, padding="same")(input_layer_block)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(128, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(64, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(64, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(64, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Conv2D(64, kernel_size=3, strides=2, padding="same")(discriminator_block_model)
        discriminator_block_model = LeakyReLU()(discriminator_block_model)

        discriminator_block_model = Flatten()(discriminator_block_model)
        discriminator_block_model = Dense(1)(discriminator_block_model)
        discriminator_block_model = Activation(activations.sigmoid)(discriminator_block_model)

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
        save_image_callback = self.ImageGeneratorCallback(numpy.array(x_training), numpy.array(y_training))
        self.adversarial_block.set_dataset_model(x_training, y_training, self.steps_per_epochs)
        self.adversarial_block.fit(numpy.array(x_training), numpy.array(y_training), epochs=2000,
                                   callbacks=[save_image_callback], batch_size=1, steps_per_epoch=1)
        logging.info('End calibration model')
        return 0

    def create_latency_noise(self):

        return numpy.array(numpy.random.uniform(size=self.length_latency_space))

    def training(self, x_training, y_training=None, evaluation_set=None):

        logging.info('Starting training model')

        self.adversarial_block.set_dataset_model(x_training, y_training, self.steps_per_epochs)
        save_image_callback = self.ImageGeneratorCallback(self.length_latency_space)
        self.adversarial_block.fit(x_training, epochs=self.epochs, callbacks=[save_image_callback], batch_size=1)

        logging.info('End training model')
        return 0

    def create_training_packet(real_image, failed_image):

        list_input_dataset = []

        for i in range(len(real_image)):
            list_input_dataset.append(real_image[i])
            list_input_dataset.append(failed_image[i])

        return numpy.array(list_input_dataset)


class AdversarialClass(keras.Model):

    def __init__(self, discriminator, generator):
        super(AdversarialClass, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.dataset_input = None
        self.dataset_output = None
        self.steps_per_epoch = None

    def set_dataset_model(self, dataset_input, dataset_output, steps_per_epoch):
        self.dataset_input = dataset_input.copy()
        self.dataset_output = dataset_output.copy()
        self.steps_per_epoch = steps_per_epoch

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

    def get_random_batch(self, samples_training):
        return [numpy.random.randint(0, len(samples_training) - 1) for _ in range(self.steps_per_epoch)]

    def get_feature_batch(self, samples_training, random_array_feature):
        return numpy.array([samples_training[random_array_feature[i]] for i in range(self.steps_per_epoch)])

    def train_step(self, real_images):
        position_random_batch = self.get_random_batch(self.dataset_input)
        x_training_samples = self.get_feature_batch(self.dataset_input, position_random_batch)
        y_training_samples = self.get_feature_batch(self.dataset_output, position_random_batch)
        number_images_per_batch = len(position_random_batch)

        synthetic_image_generated = self.generator(x_training_samples)

        batch_images_input = concat([synthetic_image_generated, y_training_samples], axis=0)
        list_labels = concat([ones((number_images_per_batch, 1)), zeros((number_images_per_batch, 1))], axis=0)

        with GradientTape() as gradient_tape:
            list_images_predicted = self.discriminator(batch_images_input)
            discriminator_loss = self.loss_fn(list_labels, list_images_predicted)

        update_gradient_function = gradient_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(update_gradient_function, self.discriminator.trainable_weights))

        misleading_labels = zeros((number_images_per_batch, 1))

        with GradientTape() as gradient_tape:
            list_images_predicted = self.discriminator(self.generator(x_training_samples))
            generator_loss = self.loss_fn(misleading_labels, list_images_predicted)

        update_gradient_function = gradient_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(update_gradient_function, self.generator.trainable_weights))
        self.d_loss_metric.update_state(discriminator_loss)
        self.g_loss_metric.update_state(generator_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result(), }

