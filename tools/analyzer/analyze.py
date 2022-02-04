#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = '@gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import logging

from tools.analyzer.attention_map import AttentionMap
from tools.analyzer.histogram import Histogram
from tools.analyzer.spectrograms import Spectrogram


class Analyzer:

    def __init__(self, args):

        self.option_generate_spectrogram = args.generate_spectrogram
        self.option_generate_attention_map = args.generate_attention_map
        self.option_generate_histogram = args.generate_histogram

        self.generator_attention_map = AttentionMap(args)
        self.generator_histogram = Histogram(args)
        self.generator_spectrogram = Spectrogram(args)

    def start_analyse(self):

        logging.debug('Start analyse model and files')

        if self.option_generate_spectrogram:
            self.generate_spectrograms()
            pass

        if self.option_generate_attention_map:
            self.generate_attention_maps()

        if self.option_generate_histogram:
            self.generate_histograms()
            pass

    def generate_attention_maps(self):

        logging.info('Start generating attention maps')
        self.generator_attention_map.generate_attention_map()
        logging.info('End generating attention maps')

    def generate_spectrograms(self):

        logging.info('Start generating spectrograms')
        self.generator_spectrogram.generate_spectrogram()
        logging.info('End generating spectrograms')

    def generate_histograms(self):

        logging.info('Start generating histograms')
        self.generator_histogram.generate_histogram()
        logging.info('End generating histograms')

