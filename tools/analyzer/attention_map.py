#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import os
import cv2
import logging
from tqdm import tqdm
from tensorflow.keras.models import Model
from tools.analyzer.resources import Resources


class AttentionMap(Resources):

    def __init__(self, args):

        super().__init__(args)
        self.output_path_heat_maps = args.output_path_heat_maps
        self.neural_network_number_layers = 12

    def generate_attention_map(self):

        logging.debug('Generating attention maps')
        self.create_directory(self.output_path_heat_maps)
        list_directory_all_file = self.get_all_files()
        self.load_file_model()

        for file_sound in tqdm(list_directory_all_file, desc='Generating attention maps'):

            self.generate_attention_map_layer(file_sound)

    def generate_attention_map_layer(self, input_file):

        self.create_directory_tree_map(self.get_file_name(input_file).split('.')[-2])
        features_extracted = self.get_spectrogram_features(input_file)

        for filter_id in range(self.neural_network_number_layers):

            image_predicted = self.get_map_predicted_model(features_extracted, filter_id)
            image_transformed = self.applied_transformation(image_predicted)
            self.save_image_attention_map(image_transformed, filter_id, self.get_file_name(input_file))

    def create_directory_tree_map(self, directory_name):

        new_path_name = '{}'.format(self.output_path_heat_maps)
        new_path_name += '/{}'.format(directory_name)

        try:

            logging.debug('Creating subdirectory {}'.format(new_path_name))
            os.mkdir(new_path_name)

        except FileExistsError:

            logging.debug('Directory exists {}'.format(new_path_name))

        finally:

            root_directory = '{}'.format(self.output_path_heat_maps)
            root_directory += '/{}'.format(directory_name)
            return root_directory

    def get_spectrogram_features(self, filename_sound):

        try:
            logging.debug('Extract feature file {}'.format(filename_sound))
            sound_feature, sample_rate = self.dataset.extract_features(filename_sound)
            return sound_feature

        except FileNotFoundError:
            logging.error('File not found {}'.format(filename_sound))
            exit(-1)

    def get_map_predicted_model(self, image_feature, identifier_layer):

        try:

            logging.debug('Loading precompiled neural network layer')
            partial_graph_input = self.neural_network.inputs
            partial_graph_output = self.neural_network.layers[identifier_layer].output
            compiled_model = Model(inputs=partial_graph_input, outputs=partial_graph_output)
            tensor_predicted = compiled_model.predict(image_feature)
            return tensor_predicted

        except SystemError:

            logging.debug('Layer {} not exiting'.format(str(identifier_layer)))
            return -1

    def save_image_attention_map(self, attention_maps, identifier_layer, filename):

        attention_map_filename = filename.split('.')[-2]
        attention_map_filename += '/Layer_{}'.format(str(identifier_layer))
        filename_output_maps = self.create_directory_tree_map(attention_map_filename)

        for channel, tensor_features in enumerate(attention_maps):

            filename_image = self.get_filename_attention_map(filename_output_maps, filename, identifier_layer, channel)

            try:

                logging.debug('Save heat map attention_maps {}'.format(filename_image))
                tensor_features_rotated = cv2.flip(tensor_features.T*8, -1)
                cv2.imwrite(filename_image, tensor_features_rotated)

            except FileExistsError:

                logging.debug('Loading precompiled neural network layer')
                exit(-1)

    @staticmethod
    def get_filename_attention_map(path_output, name_file, layer_id, conv_id):

        filename_attention_map = '{}'.format(path_output)
        filename_attention_map += '/{}_Layer'.format(name_file)
        filename_attention_map += '{}_Conv'.format(layer_id)
        filename_attention_map += '{}.jpg'.format(conv_id)

        return filename_attention_map
