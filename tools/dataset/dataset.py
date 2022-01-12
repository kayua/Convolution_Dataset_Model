import logging
from subprocess import Popen, PIPE
import numpy


class Dataset:

    def __init__(self, args):

        self.snapshot_column_position = args.column_snapshot
        self.peer_column_position = args.column_peer
        self.feature_window_length = args.window_length
        self.feature_window_width = args.window_width
        self.number_block_per_samples = args.number_block
        self.input_file_swarm_sorted = args.input_swarm
        self.output_file_swarm_sorted = args.output_swarm
        self.snapshot_id = self.feature_window_length
        self.save_file_samples = args.save_file_samples
        self.load_file_samples = args.load_file_samples

        self.features = []
        self.input_feature = []
        self.feature_input = []
        self.matrix_features = []

    def allocation_matrix(self):

        size_matrix_allocation_width = self.feature_window_width * self.number_block_per_samples

        for i in range(len(self.matrix_features), size_matrix_allocation_width):

            self.matrix_features.append([0 for _ in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):

            for j in range(len(self.matrix_features[0])):

                self.matrix_features[i][j] = 0

    def insert_in_matrix(self, snapshot_id, peer_id):

        if snapshot_id > self.snapshot_id:

            self.snapshot_id = self.snapshot_id + self.feature_window_length
            self.feature_input.append(numpy.array(self.matrix_features))
            self.clean_matrix()

        if (snapshot_id % self.feature_window_length) != 0:

            self.matrix_features[peer_id][(snapshot_id % self.feature_window_length)-1] = 1

        else:

            self.matrix_features[peer_id][self.feature_window_length-1] = 1

    def load_swarm_to_feature(self):
        self.sort(self.input_file_swarm_sorted, self.input_file_swarm_sorted)
        self.allocation_matrix()
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        line_swarm_file = file_pointer_swarm.readlines()

        for i, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])
            self.insert_in_matrix(snapshot_value, peer_value)

        self.feature_input.append(numpy.array(self.matrix_features))
        self.clean_matrix()
        self.cut_features()
        self.cast_all_features_to_swarm()
        self.sort(self.output_file_swarm_sorted, self.output_file_swarm_sorted)

    def cut_features(self):

        results = []

        for i in self.feature_input:

            for j in range(0, self.feature_window_width*self.number_block_per_samples, self.feature_window_width):

                results.append(i[j:j+self.feature_window_width])

        self.features = results

    @staticmethod
    def restore_matrix(matrix):

        matrix_results = []

        for i in matrix:

            matrix_results.extend(i)
        return matrix_results

    def cast_feature_to_swarm(self, feature, position, pointer):

        results = self.restore_matrix(feature)

        for i in range(len(results)):

            for j in range(len(results[0])):

                if results[i][j]:

                    pointer.write('{} {}\n'.format(j+position+1, i))

    def cast_all_features_to_swarm(self):

        output = open(self.output_file_swarm_sorted, 'w')
        for i, j in enumerate(range(0, len(self.features), self.number_block_per_samples)):
            self.cast_feature_to_swarm(self.features[j:j+self.number_block_per_samples], i*self.feature_window_length, output)

    def sort(self, input_file, output_file):

        sequence_commands = 'sort -n -k{},{} '.format(self.snapshot_column_position, self.snapshot_column_position)
        sequence_commands += '-k{},{} '.format(self.peer_column_position, self.peer_column_position)
        sequence_commands += '{} -o {}'.format(input_file, output_file)
        external_process = Popen(sequence_commands.split(' '), stdout=PIPE, stderr=PIPE)
        external_process.communicate()

    def save_file_samples(self):

        try:

            numpy.savez(self.save_file_samples, self.features)

        except FileNotFoundError:

            logging.error('Error: File not found error')

    def load_file_samples(self):

        try:

            dataset_file = numpy.load('{}.npz'.format(self.load_file_samples), "r")
            self.features = dataset_file['arr_0']

        except FileNotFoundError:

            logging.error('Error: File not found error')

a = Dataset()
a.load_swarm_to_feature()