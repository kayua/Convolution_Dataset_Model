

class Dataset:

    def __init__(self):

        self.snapshot_column_position = 0
        self.peer_column_position = 1
        self.feature_window_width = 16
        self.feature_window_length = 16
        self.matrix_features = []
        self.number_block_per_samples = 4
        self.input_file_swarm_sorted = 'S4'
        self.list_features = []

    def allocation_matrix(self, length):

        for i in range(len(self.matrix_features), length + 1):
            self.matrix_features.append([0 for x in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):
            for j in range(len(self.matrix_features[i])):
                self.matrix_features[i][j] = 0

    def create_features(self):

        for i in range(0, self.feature_window_width, self.number_block_per_samples):

            self.list_features.append(self.matrix_features[i:i+self.feature_window_width])

    def add_peer_in_matrix(self, snapshot, peer_id):

        self.matrix_features[int(snapshot) % self.feature_window_length][int(peer_id) % self.feature_window_width] = 1

    def load_swarm_to_feature(self):

        self.allocation_matrix(self.feature_window_width*self.number_block_per_samples)
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        lines = file_pointer_swarm.readlines()

        for snapshot_position, line in enumerate(lines):

            array_list = line.split(' ')
            snapshot_id = array_list[self.snapshot_column_position - 1]
            peer_id = array_list[self.peer_column_position - 1]
            self.add_peer_in_matrix(snapshot_id, peer_id)

            if snapshot_position % self.feature_window_length == 0:
                #self.create_features()
                pass

    def show_matrix(self):

        for i in range(len(self.matrix_features)):

            print(self.matrix_features[i])


a = Dataset()
a.load_swarm_to_feature()
a.show_matrix()