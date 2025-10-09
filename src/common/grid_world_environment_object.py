import numpy as np

class Object2D(object):
    def __init__(self, position, appearance):
        self._position = position # in the environment's reference
        self._appearance = appearance
        self._size = np.array([appearance.shape[0], appearance.shape[1] ])

    @property
    def position(self):
        return self._position
    
    @property
    def appearance(self):
        return self._appearance    
    
    @property
    def size(self):
        return self._size   

    @position.setter
    def position(self, position):
        self._position = position

    @appearance.setter
    def appearance(self, new_appearance):
        self._appearance = new_appearance

    def move_object(self, vector):
        """
        does not account for out of bounds errors
        """
        self._position = self._position + vector



class GridWorldEnvironment(object):
    def __init__(self,
                 action_dict,
                 sensor_size = np.array([10, 10]), 
                 grid = np.array([]),
                 sensor_position = np.array([250, 250]),
                 ):

        self.__grid = grid.astype(int)
        self.__grid_size = self.__grid.shape
        
        self.__sensor_size = sensor_size
        self.__sensor_position = sensor_position
        self.__action_dict = action_dict

        self.__list_of_objects = []

    @property
    def grid_size(self):
        return self.__grid_size
    
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
        # render objects on a single plane
        object_grid = np.zeros(self.__grid_size)
        for obj in self.__list_of_objects:
            # print(f"obj.position[0] : {obj.position[0]}, obj.size[0] : {obj.size[0]}")
            # print(f"obj.position[1] : {obj.position[1]}, obj.size[1] : {obj.size[1]}")
            # print(f"Shape of appearance : {obj.appearance.shape}")
            object_grid[obj.position[0]:obj.position[0]+obj.size[0],
                        obj.position[1]:obj.position[1]+obj.size[1]] =\
                        obj.appearance
 
        # overlay objects and background
        new_grid = np.where(object_grid!=0, object_grid, self.__grid)

        # render image
        s_p = self.sensor_position 
        self.__observation = new_grid[s_p[0]:s_p[0] + self.sensor_size[0],
                                      s_p[1]: s_p[1] + self.sensor_size[1]]

        return self.__observation   

    def change_object_appearance(self, object_index, new_appearance):
        if object_index >= len(self.__list_of_objects):
            print("Object not defined")
            return False
        self.__list_of_objects[object_index].appearance = new_appearance
        return True

    @sensor_position.setter
    def sensor_position(self, sensor_position):
        """ Posistion setter automaticly ensures position within the grid and
        updates the observation state
        """
        self.__sensor_position = sensor_position

    @observation.setter
    def observation(self, observation):
        self.__observation = observation
    
    def add_object(self, objects_position_agents_frame, objects_appearance):
        self.__list_of_objects.append(
            Object2D(objects_position_agents_frame + self.sensor_position, 
                     objects_appearance)
            )

    def move_object(self, object_index, transformation):
        if object_index >= len(self.__list_of_objects):
            print("Object not defined")
            return False
        self.__list_of_objects[object_index].move_object(self.__action_dict[transformation])
        return True

    def set_object_position(self, object_index, position):
        """
        Position in grid frame of reference
        """
        if object_index >= len(self.__list_of_objects):
            print("Object not defined")
            return False
        self.__list_of_objects[object_index].position = position
        return True
    
    def get_object_position_agents_pov(self, object_index):
        """
        Position in grid frame of reference
        """
        if object_index >= len(self.__list_of_objects):
            print("Object not defined")
            return False
        return self.__list_of_objects[object_index].position - self.sensor_position
    
    def set_object_position_agents_pov(self, object_index, position_agents_frame):
        """
        Position in agent's sensor frame of reference
        """
        if object_index >= len(self.__list_of_objects):
            print("Object not defined")
            return False
        
        self.__list_of_objects[object_index].position = \
            position_agents_frame + self.sensor_position
        return True
    
    def pop_object(self, object_index = -1):
        if object_index == -1:
            object_index = len(self.__list_of_objects) - 1
        if object_index >= len(self.__list_of_objects) and object_index >= 0:
            print("Object not defined")
            return False
        
        self.__list_of_objects.pop(object_index)
        return True
    
    def empty_object_list(self):
        self.__list_of_objects = []

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
        for obj in self.__list_of_objects:
            obj.move_object(-transformation_array)
        if not self.__is_position_inside_limits():
            self.sensor_position += transformation_array # cancels transformation
            return False
        else:
            return True
    
    def __is_position_inside_limits(self):
        return self.sensor_position[0] >= 0 and self.sensor_position[1] >= 0 and \
            self.sensor_position[0] + self.sensor_size[0] < self.grid_size[0] \
                and self.sensor_position[1] + self.sensor_size[1] < self.grid_size[1]

    def reset_agents_position_to_center(self):
        self.sensor_position[0] = int(self.__grid_size[0] / 2)
        self.sensor_position[1] = int(self.__grid_size[1] / 2)