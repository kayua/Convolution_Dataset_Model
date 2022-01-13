#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from glob import glob
from random import randint

import numpy
import tensorflow
import cv2
from tensorflow.keras import Input, activations, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization

from tqdm import tqdm

from models.neural_models.neural_model import NeuralModel


class ModelsV1(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.create_neural_network()

    def create_neural_network(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))
        first_convolution_block = Conv2D(180, (3, 3))(input_layer_block)
        first_convolution_block = Activation(activations.relu)(first_convolution_block)
        first_convolution_block = ZeroPadding2D((1, 1))(first_convolution_block)
        first_convolution_block = MaxPooling2D((2, 2))(first_convolution_block)
        first_convolution_block = BatchNormalization()(first_convolution_block)

        second_convolution_block = Conv2D(180, (3, 3))(first_convolution_block)
        second_convolution_block = Activation(activations.relu)(second_convolution_block)
        second_convolution_block = ZeroPadding2D((1, 1))(second_convolution_block)
        second_convolution_block = MaxPooling2D((2, 2))(second_convolution_block)
        second_convolution_block = BatchNormalization()(second_convolution_block)

        third_convolution_block = Conv2D(180, (3, 3))(second_convolution_block)
        third_convolution_block = Activation(activations.relu)(third_convolution_block)
        third_convolution_block = ZeroPadding2D((1, 1))(third_convolution_block)
        third_convolution_block = MaxPooling2D((2, 2))(third_convolution_block)
        third_convolution_block = BatchNormalization()(third_convolution_block)

        fourth_convolution_block = Conv2D(180, (3, 3))(third_convolution_block)
        fourth_convolution_block = Activation(activations.relu)(fourth_convolution_block)
        fourth_convolution_block = ZeroPadding2D((1, 1))(fourth_convolution_block)
        fourth_convolution_block = MaxPooling2D((2, 2))(fourth_convolution_block)
        fourth_convolution_block = BatchNormalization()(fourth_convolution_block)

        fifth_convolution_block = Conv2D(180, (3, 3))(fourth_convolution_block)
        fifth_convolution_block = Activation(activations.relu)(fifth_convolution_block)
        fifth_convolution_block = ZeroPadding2D((1, 1))(fifth_convolution_block)
        fifth_convolution_block = MaxPooling2D((2, 2))(fifth_convolution_block)
        fifth_convolution_block = BatchNormalization()(fifth_convolution_block)

        sixth_convolution_block = Conv2D(180, (3, 3))(fifth_convolution_block)
        sixth_convolution_block = Activation(activations.relu)(sixth_convolution_block)
        sixth_convolution_block = ZeroPadding2D((1, 1))(sixth_convolution_block)
        sixth_convolution_block = MaxPooling2D((2, 2))(sixth_convolution_block)
        sixth_convolution_block = BatchNormalization()(sixth_convolution_block)

        firts_upsampling_block = Conv2D(180, (3, 3))(sixth_convolution_block)
        firts_upsampling_block = Activation(activations.relu)(firts_upsampling_block)
        firts_upsampling_block = ZeroPadding2D((1, 1))(firts_upsampling_block)
        firts_upsampling_block = UpSampling2D((2, 2))(firts_upsampling_block)
        firts_upsampling_block = BatchNormalization()(firts_upsampling_block)

        interpolation = Add()([fifth_convolution_block, firts_upsampling_block])

        second_upsampling_block = Conv2D(180, (3, 3))(interpolation)
        second_upsampling_block = Activation(activations.relu)(second_upsampling_block)
        second_upsampling_block = ZeroPadding2D((1, 1))(second_upsampling_block)
        second_upsampling_block = UpSampling2D((2, 2))(second_upsampling_block)
        second_upsampling_block = BatchNormalization()(second_upsampling_block)

        interpolation = Add()([fourth_convolution_block, second_upsampling_block])

        third_upsampling_block = Conv2D(180, (3, 3))(interpolation)
        third_upsampling_block = Activation(activations.relu)(third_upsampling_block)
        third_upsampling_block = ZeroPadding2D((1, 1))(third_upsampling_block)
        third_upsampling_block = UpSampling2D((2, 2))(third_upsampling_block)
        third_upsampling_block = BatchNormalization()(third_upsampling_block)

        interpolation = Add()([third_convolution_block, third_upsampling_block])

        fourth_upsampling_block = Conv2D(180, (3, 3))(interpolation)
        fourth_upsampling_block = Activation(activations.relu)(fourth_upsampling_block)
        fourth_upsampling_block = ZeroPadding2D((1, 1))(fourth_upsampling_block)
        fourth_upsampling_block = UpSampling2D((2, 2))(fourth_upsampling_block)
        fourth_upsampling_block = BatchNormalization()(fourth_upsampling_block)

        interpolation = Add()([second_convolution_block, fourth_upsampling_block])

        fifth_upsampling_block = Conv2D(180, (3, 3))(interpolation)
        fifth_upsampling_block = Activation(activations.relu)(fifth_upsampling_block)
        fifth_upsampling_block = ZeroPadding2D((1, 1))(fifth_upsampling_block)
        fifth_upsampling_block = UpSampling2D((2, 2))(fifth_upsampling_block)
        fifth_upsampling_block = BatchNormalization()(fifth_upsampling_block)

        interpolation = Add()([first_convolution_block, fifth_upsampling_block])

        sixth_convolution_block = Conv2D(180, (3, 3))(interpolation)
        sixth_convolution_block = Activation(activations.relu)(sixth_convolution_block)
        sixth_convolution_block = ZeroPadding2D((1, 1))(sixth_convolution_block)
        sixth_convolution_block = UpSampling2D((2, 2))(sixth_convolution_block)
        sixth_convolution_block = BatchNormalization()(sixth_convolution_block)

        interpolation = Add()([input_layer_block, sixth_convolution_block])

        convolution_model_block = Conv2D(1, (1, 1))(interpolation)
        convolution_model_block = Conv2D(1, (1, 1))(convolution_model_block)
        convolution_model_block = Model(input_layer_block, convolution_model_block)
        convolution_model_block.compile(loss='mse', optimizer='adam', metrics='binary_crossentropy')
        convolution_model_block.summary()
        self.model = convolution_model_block


    def training(self, x_training, y_training, evaluation_set):

        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)
            samples_batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            samples_batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.generator_model.fit(x=samples_batch_training_in, y=samples_batch_training_out, verbose=2)

            print('Epoch: %d  Discriminator: %.2f Generator: %.2f' % (i + 1, generator_loss, generator_loss))

            if (i + 1) % 50 == 0:
                fake_images, _ = self.get_fake_sample(samples_batch_training_in)
                self.save_image_feature(fake_images, samples_batch_training_in, i)

        return 0


    def get_random_batch(self, samples_training):

        return [randint(0, len(samples_training) - 1) for _ in range(self.steps_per_epochs)]

    def get_feature_batch(self, samples_training, random_array_feature):

        return numpy.array([samples_training[random_array_feature[i]] for i in range(self.steps_per_epochs)])
