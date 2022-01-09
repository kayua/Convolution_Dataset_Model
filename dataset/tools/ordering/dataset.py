

class Dataset:

    def __init__(self, args):

        self.feature_window_width = args.width_window
        self.feature_window_length = args.length_window
        self.matrix_features = []
        self.number_block_per_samples = 8

    def allocation_matrix(self):

        for i in range(self.number_block_per_samples):

            self.matrix_features.append([0 for x in range(self.feature_window_width)])

    def add_peer_in_matrix(self, snapshot, peer_id):

        while True:

            try:

                self.matrix_features[int(snapshot)*self.number_block_per_samples][int(peer_id)] = 1

            except:

                self.allocation_matrix()
                self.matrix_features[int(snapshot) * self.number_block_per_samples][int(peer_id)] = 1
                exit()


    def load_swarm_to_feature(self):

        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        lines = file_pointer_swarm.readlines()

        for line in lines:
            array_list = line.split(' ')
            snapshot_id = array_list[self.snapshot_column_position - 1]
            peer_id = array_list[self.peer_column_position - 1]
            self.add_peer_in_matrix(snapshot_id, peer_id)


