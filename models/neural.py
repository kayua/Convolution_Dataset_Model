#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from models.neural_models.neural_model import NeuralModel
from tensorflow.python.keras.models import model_from_json
import numpy
import logging


class Neural:

    def __init__(self, args):

        self.epochs = args.epochs
        self.metrics = args.metrics
        self.loss = args.loss
        self.optimizer = args.optimizer
        self.steps_per_epoch = args.steps_per_epoch
        self.saved_models = args.saved_models
        self.file_load_model = args.file_load_model
        self.neural_network = None
        self.args = args
        self.adversarial_model = args.adversarial_model
        self.verbosity = args.verbosity

        self.feature_window_width = args.width_window
        self.feature_window_length = args.length_window

    def create_neural_network(self, model_instance):

        logging.info('Creating neural network model')
        self.neural_network = model_instance

        if self.verbosity == logging.DEBUG:
            self.neural_network.model.summary()

    def load_model(self):

        try:

            self.neural_network = NeuralModel(self.args)
            logging.info('Loading neural network model')

            neural_model_json = open('{}.json'.format(self.file_load_model), 'r')
            logging.debug('Loaded file {}.json'.format(self.file_load_model))
            if self.adversarial_model:

                self.neural_network.generator_model = model_from_json(neural_model_json.read())
                self.neural_network.generator_model.load_weights('{}.h5'.format(self.file_load_model))
                self.neural_network.generator_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            else:

                self.neural_network.model = model_from_json(neural_model_json.read())
                self.neural_network.model.load_weights('{}.h5'.format(self.file_load_model))
                self.neural_network.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            logging.debug('Loaded file {}.h5'.format(self.file_load_model))

            logging.debug('Neural network compiled: {} {} {} '.format(self.loss, self.optimizer, self.metrics))

            neural_model_json.close()

        except FileNotFoundError:

            logging.error('File not found: Files {}.json {}.h5'.format(self.file_load_model, self.file_load_model))
            exit(-1)

    def save_network(self):

        if self.saved_models is None:
            return

        try:

            logging.info('Saving neural network model')

            if self.adversarial_model:

                model_architecture_json = self.neural_network.generator_model.to_json()
            else:

                model_architecture_json = self.neural_network.model.to_json()

            with open('{}.json'.format(self.saved_models), "w") as json_file:

                logging.debug('Write file {}.json'.format(self.saved_models))
                json_file.write(model_architecture_json)

                logging.debug('Write file {}.h5'.format(self.saved_models))

                if self.adversarial_model:

                    self.neural_network.generator_model.save_weights('{}.h5'.format(self.saved_models))
                else:

                    self.neural_network.model.save_weights('{}.h5'.format(self.saved_models))

        except FileNotFoundError:

            logging.error('Path not found: Path {}'.format(self.saved_models.split('/')[:-1]))
            exit(-1)

    def training(self, training_set, evaluation_set=None):

        logging.info('Training neural network model')
        x_sample, y_sample, x_eval, y_eval = self.neural_network.adapter_input(training_set, evaluation_set)
        x_training_set = numpy.array(x_sample, dtype=numpy.float32)
        y_training_set = numpy.array(y_sample, dtype=numpy.float32)
        x_training_set = x_training_set.reshape((int(x_sample.size/int(self.feature_window_width*self.feature_window_length)), self.feature_window_width, self.feature_window_length, 1))
        y_training_set = y_training_set.reshape((int(y_sample.size/int(self.feature_window_width*self.feature_window_length)), self.feature_window_width, self.feature_window_length, 1))

        if (x_eval and y_eval) is not None:

            x_evaluation_set = numpy.array(x_eval, dtype=numpy.float32)
            evaluation_set_data = [x_evaluation_set, y_eval]
            logging.info('Neural network sample shape x= {} y= {}'.format(x_training_set.shape, y_training_set.shape))
            self.neural_network.training(x_training_set, y_training_set, evaluation_set_data)
        else:

            logging.info('Neural network sample shape x= {} y= {}'.format(x_training_set.shape, y_training_set.shape))
            self.neural_network.training(x_training_set, y_sample, None)

        return 0

    def predict(self, sample_x):

        return self.neural_network.generator_model.predict(x=sample_x)

    def calibration_neural_network(self):

        features_in = self.neural_network.load_images_test('dataset/calibration_dataset/failed_image')
        features_out = self.neural_network.load_images_test('dataset/calibration_dataset/original_image')

        self.neural_network.training(features_in, features_out, None)