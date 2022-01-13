#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from glob import glob

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
        self.feature_window_width = args.width_window
        self.feature_window_length = args.length_window

    def create_neural_network(self):
        pass

    @staticmethod
    def adapter_input(training_set, evaluation_set=None):

        label_training_set = numpy.array(training_set[1], dtype=numpy.int32)

        if evaluation_set is not None:

            label_evaluation_set = numpy.array(evaluation_set[1], dtype=numpy.int32)
            return training_set[0], label_training_set, evaluation_set[0], label_evaluation_set

        else:

            return training_set[0], label_training_set, None, None

    def training(self, x_training, y_training, evaluation_set):

        pass

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