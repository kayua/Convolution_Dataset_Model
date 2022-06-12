#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from models.neural_models.neural_model import NeuralModel
from tensorflow.python.keras.models import model_from_json
import logging

DEFAULT_FEATURE_INPUT_CALIBRATION = 'dataset/calibration_dataset/failed_image'
DEFAULT_FEATURE_OUTPUT_CALIBRATION = 'dataset/calibration_dataset/original_image'


class Neural:

    def __init__(self, args):

        self.epochs = args.epochs
        self.metrics = args.metrics
        self.loss = args.loss
        self.optimizer = args.optimizer
        self.steps_per_epoch = args.steps_per_epoch
        self.file_save_models = args.save_model
        self.file_load_model = args.load_model
        self.neural_network = None
        self.args = args
        self.verbosity = args.verbosity
        self.feature_window_width = args.window_width
        self.feature_window_length = args.window_length

        self.number_block_per_samples = args.number_blocks

    def create_neural_network(self, model_instance):

        logging.info('Creating neural network model')
        self.neural_network = model_instance

        if self.verbosity == logging.DEBUG:
            self.neural_network.model.summary()

    def load_model(self):

        try:

            self.neural_network = NeuralModel(self.args)
            logging.info('Loading neural network model')
            logging.info('Architecture file: {}.json'.format(self.file_load_model))
            neural_model_json = open('{}.json'.format(self.file_load_model), 'r')
            self.neural_network.model = model_from_json(neural_model_json.read())
            logging.info('Loading architecture... done! {}.json'.format(self.file_load_model))

            logging.info('Weights file: {}.h5'.format(self.file_load_model))
            self.neural_network.model.load_weights('{}.h5'.format(self.file_load_model))
            self.neural_network.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            logging.info('Loading weights... done!  {}.h5'.format(self.file_load_model))

            logging.debug('Neural network compiled: {} {} {} '.format(self.loss, self.optimizer, self.metrics))

            neural_model_json.close()

        except FileNotFoundError:

            logging.error('File not found: Files {}.json {}.h5'.format(self.file_load_model, self.file_load_model))
            exit(-1)

    def save_network(self):

        try:

            if self.file_save_models is None:
                logging.warning('Model not saved. Model Empty')
                return

            logging.info('Saving neural network model')

            model_architecture_json = self.neural_network.model.to_json()
            logging.debug('Saving feedforward model model')

            with open('{}.json'.format(self.file_save_models), "w") as json_file:

                json_file.write(model_architecture_json)
                logging.debug('Write file {}.json'.format(self.file_save_models))
                self.neural_network.model.save_weights('{}.h5'.format(self.file_save_models))
                logging.debug('Write file {}.h5'.format(self.file_save_models))

            logging.debug('Neural network saved {}'.format(self.file_save_models))

        except FileNotFoundError:

            logging.error('Path not found: Path {}'.format(self.file_save_models.split('/')[:-1]))
            exit(-1)

    def training(self, training_set_in, training_set_out):
        #RM-2022-06-12
        # logging.info('Start training neural network model')
        # x_samples, y_samples = self.neural_network.adapter_input(training_set_in, training_set_out)
        # number_samples = int(x_samples.size / int(self.feature_window_width * self.feature_window_length))
        # logging.debug('Reshape list samples')
        # x_training_set = x_samples.reshape((number_samples, self.feature_window_width, self.feature_window_length, 1))
        # y_training_set = y_samples.reshape((number_samples, self.feature_window_width, self.feature_window_length, 1))
        # logging.info('Neural network sample shape x= {} y= {}'.format(x_training_set.shape, y_training_set.shape))
        # self.neural_network.training(x_training_set, y_training_set)

        logging.info('Start training neural network model')
        x_samples, y_samples = self.neural_network.adapter_input(training_set_in, training_set_out)
        number_samples_x = int(x_samples.size / (self.feature_window_width * self.number_block_per_samples))
        number_samples_y = int(x_samples.size / (self.feature_window_length * self.number_block_per_samples))
        logging.debug('Reshape list samples')
        x_training_set = x_samples.reshape((number_samples_x, self.feature_window_width, self.feature_window_length, 1))
        y_training_set = y_samples.reshape((number_samples_y, self.feature_window_width, self.feature_window_length, 1))
        logging.info('Neural network sample shape x= {} y= {}'.format(x_training_set.shape, y_training_set.shape))
        self.neural_network.training(x_training_set, y_training_set)

        return 0

    def predict(self, sample_x):

        logging.info('Waiting: Predicting samples')
        return self.neural_network.model.predict(x=sample_x)

    def calibration_neural_network(self):

        features_in = self.neural_network.load_images_test(DEFAULT_FEATURE_INPUT_CALIBRATION)
        features_out = self.neural_network.load_images_test(DEFAULT_FEATURE_OUTPUT_CALIBRATION)
        self.neural_network.calibration(features_in, features_out)