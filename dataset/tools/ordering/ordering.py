import logging
from subprocess import Popen
from subprocess import PIPE
import numpy


class Dataset:

    def __init__(self, args):

        self.swarm_file_unsorted_input = args.input_unsorted_file
        self.swarm_file_sorted_output = args.output_sorted_file
        self.snapshot_column_position = args.snapshot_position
        self.peer_column_position = args.peer_position


    def sort_dataset_trace(self):

        sequence_commands = 'sort -n -k{},{} '.format(self.snapshot_column_position, self.snapshot_column_position)
        sequence_commands += '-k{},{} '.format(self.peer_column_position, self.peer_column_position)
        sequence_commands += '{} -o {}'.format(self.swarm_file_unsorted_input, self.swarm_file_sorted_output)
        external_process = Popen(sequence_commands.split(' '), stdout=PIPE, stderr=PIPE)
        command_stdout, command_stderr = external_process.communicate()

        if command_stderr.decode("utf-8") != '':
            logging.error('Error sort file')
            exit(-1)
