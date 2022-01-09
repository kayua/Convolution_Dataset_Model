import logging
from subprocess import Popen
from subprocess import PIPE
import numpy


class Dataset:

    def __init__(self, args):

        self.swarm_file_unsorted_input = args.input_unsorted_file
        self.swarm_file_sorted_output = args.output_sorted_file
        self.input_file_swarm_sorted = args.input_file_swarm_sorted
        self.output_file_samples = args.output_file_samples
        self.feature_window_width = args.width_window
        self.feature_window_length = args.length_window
        self.file_results = args.file_results
        self.threshold = 0.5

        self.snapshot_column_position = args.snapshot_position
        self.peer_column_position = args.peer_position

        self.matrix_features = []
        self.list_features = []
        self.list_features_array = None
        self.file_input_feature = None

    def sort_dataset_trace(self):

        sequence_commands = 'sort -n -k{},{} '.format(self.snapshot_column_position, self.snapshot_column_position)
        sequence_commands += '-k{},{} '.format(self.peer_column_position, self.peer_column_position)
        sequence_commands += '{} -o {}'.format(self.swarm_file_unsorted_input, self.swarm_file_sorted_output)
        external_process = Popen(sequence_commands.split(' '), stdout=PIPE, stderr=PIPE)
        command_stdout, command_stderr = external_process.communicate()

        if command_stderr.decode("utf-8") != '':
            logging.error('Error sort file')
            exit(-1)

    def allocation_matrix(self, length):

        for i in range(len(self.matrix_features), length + 1):
            self.matrix_features.append([0 for x in range(self.feature_window_width)])

    def add_peer_in_matrix(self, snapshot, peer_id):

        try:

            self.matrix_features[int(snapshot)][int(peer_id)] = 1

        except:

            self.allocation_matrix(int(snapshot))
            try:

                self.matrix_features[int(snapshot)][int(peer_id)] = 1

            except:

                pass

    def load_swarm_to_feature(self):

        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        lines = file_pointer_swarm.readlines()

        for line in lines:
            array_list = line.split(' ')
            snapshot_id = array_list[self.snapshot_column_position - 1]
            peer_id = array_list[self.peer_column_position - 1]
            self.add_peer_in_matrix(snapshot_id, peer_id)

    def create_samples(self):

        length_feature = len(self.matrix_features)
        fill_boarders = (self.feature_window_length - (length_feature % self.feature_window_length)) + length_feature
        self.allocation_matrix(fill_boarders - 1)

        for i in range(0, len(self.matrix_features), self.feature_window_length):
            self.list_features.append(self.matrix_features[i:i + self.feature_window_length])

        self.list_features_array = numpy.array(self.list_features)

    def save_file_samples(self):

        try:

            numpy.savez(self.output_file_samples, self.list_features_array)

        except FileNotFoundError:

            logging.error('Error: File not found error')

    def load_file_samples(self, input_samples):

        try:

            dataset_file = numpy.load('{}.npz'.format(input_samples), "r")
            self.file_input_feature = dataset_file['arr_0']

        except FileNotFoundError:

            logging.error('Error: File not found error')

    def get_training_features(self):

        return self.file_input_feature

    def cast_and_save_tensor_output(self, tensor_corrected):

        file_output = open(self.file_results, 'w')

        for a, b in enumerate(tensor_corrected):

            for c, d in enumerate(b):
                list_d = d.tolist()

                for f, l in enumerate(list_d):
                    if float(l) > self.threshold:

                        line_output = '{} {}\n'.format(str((a*self.feature_window_length)+c), f)
                        file_output.write(line_output)