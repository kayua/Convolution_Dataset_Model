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
