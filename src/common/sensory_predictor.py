import numpy as np

class SensoryPredictor():
    def __init__(self, permutation_maps, side_size):
        self.__permutation_maps = permutation_maps
        self.__side_size = side_size

    @property
    def side_size(self):
        return self.__side_size
    
    def predict(self, observation, action):
        permutation_map = self.__permutation_maps[action]
        observation_predicted = np.zeros(observation.shape)
        observation_predicted[:] = np.nan

        for i in range(permutation_map.shape[0]):
            for j in range(permutation_map.shape[1]):
                if permutation_map[i, j] == 1:
                    pre_sensel_index = np.array((i // self.side_size , i % self.side_size)) 
                    post_sensel_index = np.array((j // self.side_size, j % self.side_size)) 
                    observation_predicted[post_sensel_index[0], post_sensel_index[1]] = \
                    observation[pre_sensel_index[0], pre_sensel_index[1]]
        return observation_predicted