#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from glob import glob
from random import randint

import cv2
import numpy
import tensorflow
from tensorflow import keras
from tensorflow import random
from tqdm import tqdm
DEFAULT_CALIBRATION_PATH_IMAGE = 'images'


class NeuralModel:

    def __init__(self, args):

        self.model = None
        self.steps_per_epochs = args.steps_per_epoch
        self.epochs = args.epochs
        self.loss = args.loss
        self.optimizer = args.optimizer
        self.metrics = args.metrics
        self.feature_window_width = args.window_width
        self.feature_window_length = args.window_length

    def create_neural_network(self):
        pass

    @staticmethod
    def adapter_input(training_set_in, training_set_out):

        in_training_set = numpy.array(training_set_in, dtype=numpy.int32)
        out_training_set = numpy.array(training_set_out, dtype=numpy.int32)

        return in_training_set, out_training_set

    def training(self, x_training, y_training, evaluation_set):

        pass

    def calibration(self, x_training, y_training):

        for i in range(1200):

            random_array_feature = self.get_random_batch(x_training)
            samples_batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            samples_batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.model.fit(x=samples_batch_training_in, y=samples_batch_training_out, verbose=2)

            if (i % 10)==1:

                artificial_image = self.model.predict(samples_batch_training_in[0:10])
                self.save_image_feature(artificial_image[0], samples_batch_training_in[0], samples_batch_training_out[0], i)

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

        for i in tqdm(list_samples_training, desc="Loading dataset calibration"):
            gray_scale_feature = self.parse_image(i)

            list_features_image_gray_scale.append(gray_scale_feature)

        return list_features_image_gray_scale

    def get_random_batch(self, samples_training):

        return [randint(0, len(samples_training) - 1) for _ in range(self.steps_per_epochs)]

    def get_feature_batch(self, samples_training, random_array_feature):

        return numpy.array([samples_training[random_array_feature[i]] for i in range(self.steps_per_epochs)])

    @staticmethod
    def save_image_feature(examples, examples_a, example_b, epoch):

        cv2.imwrite('{}/output_{}.png'.format(DEFAULT_CALIBRATION_PATH_IMAGE, epoch), numpy.array(examples_a * 255))
        cv2.imwrite('{}/predicted_{}.png'.format(DEFAULT_CALIBRATION_PATH_IMAGE, epoch), examples * 255)
        cv2.imwrite('{}/input{}.png'.format(DEFAULT_CALIBRATION_PATH_IMAGE, epoch), example_b * 255)

    class ImageGeneratorCallback(keras.callbacks.Callback):

        def __init__(self, feature_window_width, feature_window_length):
            super().__init__()
            self.feature_window_width = feature_window_width
            self.feature_window_length = feature_window_length

        def on_epoch_end(self, epoch, logs=None):
            latency_matrix = random.normal(shape=(2, self.feature_window_width, self.feature_window_length))
            generated_image = self.model.generator(latency_matrix)
            generated_image = generated_image * 255
            generated_image.numpy()
            synthetic_image = keras.preprocessing.image.array_to_img(generated_image[0])
            synthetic_image.save('generated_img_{}.png'.format(epoch))