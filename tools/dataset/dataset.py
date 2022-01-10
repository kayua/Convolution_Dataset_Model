import numpy


class Dataset:

    def __init__(self):

        self.snapshot_column_position = 1
        self.peer_column_position = 2
        self.feature_window_length = 10
        self.feature_window_width = 256
        self.matrix_features = []
        self.number_block_per_samples = 16
        self.input_file_swarm_sorted = 'S4'
        self.list_features = []

    def allocation_matrix(self):

        for i in range(len(self.matrix_features), self.feature_window_width*self.number_block_per_samples):
            self.matrix_features.append([0 for x in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):
            for j in range(len(self.matrix_features[i])):
                self.matrix_features[i][j] = 0

    def add_peer_in_matrix(self, snapshot, peer_id):

        self.matrix_features[peer_id][(snapshot % self.feature_window_length)-1] = 1

    def load_swarm_to_feature(self):

        self.allocation_matrix()

        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        lines = file_pointer_swarm.readlines()

        for _, line in enumerate(lines):

            array_list = line.split(' ')
            snapshot_id = array_list[self.snapshot_column_position-1]
            peer_id = array_list[self.peer_column_position-1]
            self.add_peer_in_matrix(int(snapshot_id), int(peer_id))
            if int(snapshot_id) % self.feature_window_length+1 == 0:
                self.show_matrix()
                exit()

        self.show_matrix()
        exit()
        pass

    def show_matrix(self):

        for i in range(len(self.matrix_features)):
            print(self.matrix_features[i])





a = Dataset()

a.load_swarm_to_feature()
a.show_matrix()