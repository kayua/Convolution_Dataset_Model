#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import librosa
import numpy
import logging
from tqdm import tqdm
from matplotlib import pyplot

from tools.analyzer.resources import Resources
from tools.parameters.parameters_analyse import DEFAULT_HISTOGRAM_MIN_CUTOFF_INTENSITY
from tools.parameters.parameters_analyse import DEFAULT_HISTOGRAM_MAX_CUTOFF_INTENSITY
from tools.parameters.parameters_analyse import DEFAULT_HISTOGRAM_TITLE_AXIS_Y
from tools.parameters.parameters_analyse import DEFAULT_HISTOGRAM_TITLE


class Histogram(Resources):

    def __init__(self, args):

        super().__init__(args)

        self.output_path_histogram = args.output_path_histogram
        self.plot_histogram = args.generate_histogram
        self.applied_filters = args.applied_filters

    def generate_histogram_unique(self, signal_sound_discrete, file_output_name):

        signal_sound_discrete = self.preprocessing.amplification(signal_sound_discrete)
        logging.debug('Starting plotting histograms')

        if self.applied_filters:

            logging.debug('Applying filter in histograms')
            signal_sound_discrete = self.preprocessing.applied_frequency_filters(signal_sound_discrete)

        signal_sound_discrete = self.spectrogram.get_short_transform_fourier(signal_sound_discrete)
        signal_sound_power = librosa.amplitude_to_db(numpy.abs(signal_sound_discrete), ref=numpy.min)
        mean_distribution_frequency = numpy.mean(signal_sound_power, axis=1)
        self.generate_histogram_image(file_output_name, mean_distribution_frequency)

    def generate_histogram_image(self, file_output_name, mean_distribution_frequency):

        pyplot.bar(numpy.arange(mean_distribution_frequency.shape[0]), mean_distribution_frequency)
        pyplot.ylim([DEFAULT_HISTOGRAM_MIN_CUTOFF_INTENSITY, DEFAULT_HISTOGRAM_MAX_CUTOFF_INTENSITY])
        pyplot.xlabel('Frequency Hz')
        pyplot.ylabel(DEFAULT_HISTOGRAM_TITLE_AXIS_Y)
        pyplot.title(DEFAULT_HISTOGRAM_TITLE)
        output_filename = '{}/{}_histogram.png'.format(self.output_path_histogram, file_output_name)

        try:

            logging.debug('Saving Histogram image {}'.format(output_filename))
            pyplot.savefig(output_filename)

        except FileNotFoundError:

            logging.error('Path not found: {}'.format(self.output_path_histogram))
            exit(-1)

        finally:

            pyplot.close()

    def generate_histogram(self):

        logging.debug('Plotting Histogram and spectrogram')
        self.create_directory(self.output_path_histogram)
        directory_file = self.get_all_files()

        for file_sound in tqdm(directory_file):

            sound_source_signal = self.load_sound(file_sound)
            filename_histogram = self.get_file_name('{}'.format(file_sound))
            self.generate_histogram_unique(sound_source_signal, filename_histogram)
