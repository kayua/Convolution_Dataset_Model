#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import os
from glob import glob
import numpy
import logging
from librosa import load
from tensorflow.keras.models import model_from_json

from tools.dataset.dataset import Dataset
from tools.features.features import Features
from tools.preprocessing.preprocessing import Preprocessing


class Resources:

    def __init__(self, args):

        self.signal_sample_rate = args.signal_sample_rate
        self.input_path_sound_plot = args.input_path_sound_plot
        self.model_loss = args.model_loss
        self.model_optimizer = args.model_optimizer
        self.file_load_model = args.file_load_model
        self.max_frequency = args.max_frequency
        self.min_frequency = args.min_frequency
        self.neural_network = None
        self.dataset = Dataset(args)
        self.spectrogram = Features(args)
        self.preprocessing = Preprocessing(args)

    def get_all_files(self):

        filename_list_dir = '{}/*'.format(self.input_path_sound_plot)
        list_all_filenames = glob(filename_list_dir)
        return list_all_filenames

    @staticmethod
    def get_directories(dataset_path):

        logging.debug('Get name file samples {}'.format(dataset_path))
        filename_subdirectories = '{}/*'.format(dataset_path)
        list_subdirectories = glob(filename_subdirectories)
        list_subdirectories = sorted(list_subdirectories)
        subdirectories_in_order = []

        for filename_path in list_subdirectories:
            subdirectories_in_order.append(filename_path)

        return subdirectories_in_order

    @staticmethod
    def get_file_name(data_path):

        filename = data_path.split('/')[-1]
        return filename

    def load_sound(self, file_sound):

        try:

            sound_source_signal, sample_rate = load(file_sound, sr=self.signal_sample_rate)
            return sound_source_signal

        except FileNotFoundError:

            logging.error('Path not found: {}'.format(file_sound.split('/')[:-1]))
            exit(-1)

    def load_file_model(self):

        try:

            logging.info('Loading neural network model')
            filename_architecture_model = '{}.json'.format(self.file_load_model)
            neural_model_json = open(filename_architecture_model, 'r')

            logging.debug('Loaded file {}.json'.format(self.file_load_model))
            self.neural_network = model_from_json(neural_model_json.read())
            filename_weights_model = '{}.h5'.format(self.file_load_model)
            self.neural_network.load_weights(filename_weights_model)

            logging.debug('Loaded file {}.h5'.format(self.file_load_model))
            self.neural_network.compile(loss=self.model_loss, optimizer=self.model_optimizer)
            self.neural_network.summary()
            logging.debug('Neural network compiled: {} {}'.format(self.model_loss, self.model_optimizer))
            neural_model_json.close()

        except FileNotFoundError:

            logging.error('File not found: Files {}.json {}.h5'.format(self.file_load_model, self.file_load_model))
            exit(-1)

    @staticmethod
    def applied_transformation(signal_image):

        logging.debug('Applied transformation')
        feature_signal_filtered = []

        for signal_unique_feature in signal_image.T:

            reshape_feature_array = []

            for signal_unique_time in signal_unique_feature.T:

                reshape_feature_array.extend(signal_unique_time.T)

            feature_signal_filtered.append(reshape_feature_array)

        return numpy.asarray(feature_signal_filtered)

    @staticmethod
    def create_directory(directory_name):

        try:
            logging.debug('Creating directory {}'.format(directory_name))
            os.makedirs(directory_name)

        except OSError:
            logging.debug('Directory exist {}'.format(directory_name))

