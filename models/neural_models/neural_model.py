#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import numpy


class NeuralModel:

    def __init__(self, args):
        self.model = None
        self.generator_model = None
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