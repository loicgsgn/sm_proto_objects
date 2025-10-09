import numpy as np

class Object2D():
    def __init__(self, position, appearance):
        self._position = position # in the environment's reference
        self._appearance = appearance
        self._size = appearance.shape

class GridWorldEnvironment():
    def __init__(self,
                 action_dict,
                 grid_size = np.array([500, 500]),
                 sensor_size = np.array([10, 10]), 
                 environment_range = np.array([0, 255]),
                 sensor_position = np.array([250, 250]),
                 grid = np.array([]),
                 ):
        self.__grid_size = grid_size
        self.__environment_range = environment_range

        self.__grid = grid.astype(int)
        self.__grid_size = self.__grid.shape
        
        self.__sensor_size = sensor_size
        self.sensor_position = sensor_position
        self.__action_dict = action_dict

    @property
    def grid_size(self):
        return self.__grid_size

    @property
    def environment_range(self):
        return self.__environment_range
    
    @property
    def grid(self):
        return self.__grid
    
    @property
    def sensor_size(self):
        return self.__sensor_size

    @property
    def sensor_position(self):
        return self.__sensor_position   

    @property
    def observation(self):
        return self.__observation   
      
    def get_observation(self):
        return self.__observation   

    @sensor_position.setter
    def sensor_position(self, sensor_position):
        """ Posistion setter automaticly ensures position within the grid and
        updates the observation state
        """
        self.__sensor_position = sensor_position


        self.observation = self.__grid[self.sensor_position[0]:self.sensor_position[0] + self.sensor_size[0],
                          self.sensor_position[1]: self.sensor_position[1] + self.sensor_size[1]]

    @observation.setter
    def observation(self, observation):
        self.__observation = observation



    def agent_step(self, action):
        """
        Returns False if the agent could not move because out of bounds
        """
        action_array = self.__action_dict[action]
        self.sensor_position += action_array
        if not self.__is_position_inside_limits():
            self.sensor_position -= action_array # cancels action
            return False
        else:
            return True

    def environment_step(self, transformation):
        """
        Returns False if the transformation could not be applied because
        out of bounds sensor
        """
        transformation_array = self.__action_dict[transformation]
        # "-" because performing the transformation results as if the agent 
        # had performed the inverse action
        self.sensor_position -= transformation_array 
        if not self.__is_position_inside_limits():
            self.sensor_position += transformation_array # cancels transformation
            return False
        else:
            return True
    
    def __is_position_inside_limits(self):
        return self.sensor_position[0] >= 0 and self.sensor_position[1] >= 0 and \
            self.sensor_position[0] + self.sensor_size[0] < self.grid_size[0] \
                and self.sensor_position[1] + self.sensor_size[1] < self.grid_size[1]

    def reset_position_to_center(self):
        self.sensor_position[0] = int(self.__grid_size[0] / 2)
        self.sensor_position[1] = int(self.__grid_size[1] / 2)