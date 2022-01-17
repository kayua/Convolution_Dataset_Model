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


class ModelsV1(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.create_neural_network()

    def create_neural_network(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        first_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = Activation(activations.relu)(first_convolution)

        second_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = Activation(activations.relu)(second_convolution)

        third_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = Activation(activations.relu)(third_convolution)

        fourth_convolution_block = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution_block = Activation(activations.relu)(fourth_convolution_block)

        fifth_convolution_block = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fourth_convolution_block)
        fifth_convolution_block = Activation(activations.relu)(fifth_convolution_block)

        first_deconvolution_block = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(
            fifth_convolution_block)
        first_deconvolution_block = Activation(activations.relu)(first_deconvolution_block)

        interpolation = Add()([first_deconvolution_block, fourth_convolution_block])

        second_deconvolution_block = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        second_deconvolution_block = Activation(activations.relu)(second_deconvolution_block)

        interpolation = Add()([second_deconvolution_block, third_convolution])

        third_deconvolution_block = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        third_deconvolution_block = Activation(activations.relu)(third_deconvolution_block)

        interpolation = Add()([third_deconvolution_block, second_convolution])

        fourth_deconvolution_block = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fourth_deconvolution_block = Activation(activations.relu)(fourth_deconvolution_block)

        interpolation = Add()([fourth_deconvolution_block, first_convolution])

        fifth_deconvolution_block = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fifth_deconvolution_block = Activation(activations.relu)(fifth_deconvolution_block)

        interpolation = Add()([fifth_deconvolution_block, input_layer_block])

        convolution_model_block = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(interpolation)
        convolution_model_block = Activation(activations.relu)(convolution_model_block)

        convolution_model_block = Conv2D(1, (1, 1))(convolution_model_block)
        convolution_model_block = Conv2D(1, (1, 1))(convolution_model_block)

        convolution_model_block = Model(input_layer_block, convolution_model_block)
        convolution_model_block.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        convolution_model_block.summary()
        self.model = convolution_model_block

    @staticmethod
    def check_feature_empty(feature):

        number_true_samples = 0

        for i in range(len(feature)):

            for j in range(len(feature[0])):

                if int(feature[i][j]) == 1:
                    number_true_samples += 1

        if number_true_samples > 0:
            return 1
        else:
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
        print(x_training.shape)
        x_training, y_training = self.remove_empty_features(x_training, y_training)
        print(x_training.shape)
        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)
            samples_batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            samples_batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.model.fit(x=samples_batch_training_in, y=samples_batch_training_out, epochs=1, verbose=1)

            if i % 10 == 0:
                b = self.model.predict(samples_batch_training_in[0:100])
                self.save_image_feature(b[0:100], samples_batch_training_out[0:100], samples_batch_training_in[0:100],
                                        i)

        return 0
