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
from tqdm import tqdm


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
    def adapter_input(training_set, evaluation_set=None):

        label_training_set = numpy.array(training_set, dtype=numpy.int32)

        if evaluation_set is not None:

            label_evaluation_set = numpy.array(evaluation_set, dtype=numpy.int32)
            return training_set, label_training_set, evaluation_set, label_evaluation_set

        else:

            return training_set, label_training_set, None, None

    def training(self, x_training, y_training, evaluation_set):

        pass

    def calibration(self, x_training, y_training):

        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)
            samples_batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            samples_batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.model.fit(x=samples_batch_training_in, y=samples_batch_training_out, verbose=2)

            if self.epochs % 10:

                artificial_image = self.model.predict(samples_batch_training_in)
                self.save_image_feature(artificial_image, samples_batch_training_in, i)

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
    def save_image_feature(examples, examples_a, epoch):

        cv2.imwrite('images/noise_{}.png'.format(epoch), numpy.array(examples_a[0] * 255))
        cv2.imwrite('images/{}.png'.format(epoch), examples[0] * 255)