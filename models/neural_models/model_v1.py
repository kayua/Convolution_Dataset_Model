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
from tensorflow.keras import Input, activations, Model, Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization

from matplotlib import pyplot
from tqdm import tqdm

from models.neural_models.neural_model import NeuralModel


class ModelsV1a(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.discriminator_model = None
        self.extractor_model = None
        self.results_predicted = None
        self.create_neural_network()

    def create_neural_network(self):

        input_layer_discriminator = Input(shape=(self.feature_window_width, self.feature_window_length, 1))
        input_layer_generator = Input(shape=(self.feature_window_width, self.feature_window_length, 1))
        self.discriminator_model = self.create_block_discriminator(input_layer_discriminator)
        self.extractor_model = self.create_block_extractor_feature(input_layer_generator)
        self.generator_model = self.create_block_generator(input_layer_generator, self.extractor_model)
        self.neural_network = self.create_generative_adversarial_model(self.generator_model, self.discriminator_model)
        self.model = self.neural_network

    @staticmethod
    def create_block_extractor_feature(input_layer):

        return input_layer

    @staticmethod
    def create_block_generator(input_layer, extractor_block):

        first_convolution_block = Conv2D(180, (3, 3))(input_layer)
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

        interpolation = Add()([input_layer, sixth_convolution_block])

        model_block_generative = Conv2D(1, (1, 1))(interpolation)
        model_block_generative = Conv2D(1, (1, 1))(model_block_generative)
        model_block_generative = Model(input_layer, model_block_generative)
        model_block_generative.compile(loss='mse', optimizer='adam', metrics='binary_crossentropy')
        model_block_generative.summary()

        return model_block_generative

    @staticmethod
    def create_block_discriminator(input_layer):

        model_block_discriminator = Conv2D(32, (3, 3))(input_layer)
        model_block_discriminator = Activation(activations.relu)(model_block_discriminator)
        model_block_discriminator = MaxPooling2D((2, 2))(model_block_discriminator)

        model_block_discriminator = Conv2D(32, (3, 3))(model_block_discriminator)
        model_block_discriminator = Activation(activations.relu)(model_block_discriminator)
        model_block_discriminator = MaxPooling2D((2, 2))(model_block_discriminator)

        model_block_discriminator = Conv2D(32, (3, 3))(model_block_discriminator)
        model_block_discriminator = Activation(activations.relu)(model_block_discriminator)
        model_block_discriminator = MaxPooling2D((2, 2))(model_block_discriminator)

        model_block_discriminator = Conv2D(32, (3, 3))(model_block_discriminator)
        model_block_discriminator = Activation(activations.relu)(model_block_discriminator)
        model_block_discriminator = MaxPooling2D((2, 2))(model_block_discriminator)

        model_block_discriminator = Flatten()(model_block_discriminator)

        model_block_discriminator = Dense(16)(model_block_discriminator)
        model_block_discriminator = Activation(activations.relu)(model_block_discriminator)

        model_block_discriminator = Dense(1)(model_block_discriminator)
        model_block_discriminator = Activation(activations.softmax)(model_block_discriminator)

        model_block_discriminator = Model(input_layer, model_block_discriminator)
        model_block_discriminator.compile(loss='binary_crossentropy', optimizer='adam')

        return model_block_discriminator

    @staticmethod
    def create_generative_adversarial_model(generative_model, discriminator_model):

        discriminator_model.trainable = False
        model = Sequential()
        model.add(generative_model)
        model.add(discriminator_model)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.summary()
        return model

    def get_real_sample(self, x_training):

        return numpy.array(x_training), numpy.ones(self.steps_per_epochs)

    def get_fake_sample(self, x_training):

        fake_feature_generated = self.generator_model.predict(numpy.array(x_training))
        return fake_feature_generated, numpy.zeros(self.steps_per_epochs)

    def training(self, x_training, y_training, evaluation_set):

        for i in range(self.epochs):

            random_array_feature = [randint(0, len(x_training) - 1) for i in range(self.steps_per_epochs)]

            samples_batch_training_in = numpy.array(
                [x_training[random_array_feature[i]] for i in range(self.steps_per_epochs)])
            samples_batch_training_out = numpy.array(
                [y_training[random_array_feature[i]] for i in range(self.steps_per_epochs)])
            generator_loss = self.generator_model.fit(x=samples_batch_training_in, y=samples_batch_training_out,
                                                      verbose=2)

            # print('Epoch: %d  Discriminator: %.2f Generator: %.2f' % (i + 1, generator_loss, generator_loss))

            if (i + 1) % 50 == 0:
                fake_images, _ = self.get_fake_sample(samples_batch_training_in)
                self.save_image_feature(fake_images, samples_batch_training_in, i)

        return 0

    def parse_image(self, filename):

        image = tensorflow.io.read_file(filename)
        image = tensorflow.image.decode_png(image, channels=1)
        image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
        image = tensorflow.image.resize(image, [self.feature_window_width, self.feature_window_length])
        return image

    def load_images_test(self, path_images):

        list_samples_training = glob(path_images + "/*")
        list_samples_training.sort()

        list_samples_training = list_samples_training[:512]
        list_features_image_gray_scale = []

        for i in tqdm(list_samples_training, desc="Loading training set"):
            gray_scale_feature = self.parse_image(i)

            list_features_image_gray_scale.append(gray_scale_feature)
        return list_features_image_gray_scale

    @staticmethod
    def save_image_feature(examples, examples_a, epoch):

        cv2.imwrite('images/noise_{}.png'.format(epoch), numpy.array(examples_a[0] * 255))
        cv2.imwrite('images/{}.png'.format(epoch), examples[0] * 255)




