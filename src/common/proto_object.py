import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))  # get current script directory
utils_dir = os.path.join(script_dir,'..', "utils")  # adjust the relative path to reach /utils/
sys.path.append(utils_dir)

# from utils import *

from src.common.utils import *

def remove_doubles_list_of_ndarrays(list_of_ndarrays):
    unique = []
    for i, a in enumerate(list_of_ndarrays):
        if not any(np.array_equal(a, b) for b in unique):
            unique.append(a)
    return unique


class ProtoObject:
    def __init__(self, init_elements_list, update_rate, rejection_threshold, initial_objecthood_probability = 1.):
        # self.__elements_list = init_elements_list # elements ndarray np.array([x, y])
        # self.__elements_list = [init_elements_list[i] for i in range(init_elements_list.shape[0])]
        self.__elements_list = remove_doubles_list_of_ndarrays(init_elements_list)
        self.__update_rate = update_rate
        self.__rejection_threshold = rejection_threshold
        self.__initial_objecthood_probability = initial_objecthood_probability
        self.__objecthood_prob_list = [self.__initial_objecthood_probability]*len(self.__elements_list)

    @property
    def elements_list(self):
        return self.__elements_list
    
    @property
    def update_rate(self):
        return self.__update_rate

    @property
    def rejection_threhold(self):
        return self.__rejection_threshold

    @property
    def objecthood_prob_list(self):
        return self.__objecthood_prob_list
    
    @elements_list.setter
    def elements_list(self, elements_list):
        self.__elements_list = elements_list
    
    @objecthood_prob_list.setter
    def objecthood_prob_list(self, objecthood_prob_list):
        self.__objecthood_prob_list = objecthood_prob_list

    def __add_element(self, element_position):
        self.__elements_list.append(element_position)
        self.__objecthood_prob_list.append(self.__initial_objecthood_probability)

    def __remove_element(self, index):
        self.__elements_list.pop(index)
        self.__objecthood_prob_list.pop(index)

    # def matchElements()
    def move_elements(self, permutation_map):
        """Disaplce each element of proto object according to permutation map
        TODO : if permutation map has no successor, discard the element (no object permanence)
        """
        # print(f"Action delta {permutation_map}")
        # new_postition = [element - action_delta for element in self.__elements_list]
        new_positions = []
        for element in self.elements_list:
            new_coordinate, new_exists  = displace_coord_of_permutation(element, permutation_map)
            # Remove if there is no sucessor
            if new_exists: 
                new_positions.append(new_coordinate)
            # print(f"Element : {element}")
        # print(f"Old positions {self.elements_list}")
        # print(f"New positions {new_positions}")
        # plt.imshow(permutation_map)
        # plt.show()

        #TODO Move elements according to permutation map
        # If element is not described then we remove it
        # self.__remove_element(index)
        self.elements_list = new_positions
        return new_positions

    def __compare(self, list_1, list_2):
        return [any(np.array_equal(a, b) for b in list_2) for a in list_1]



    def update(self, new_elements, permutation_map):
        # self.__element_list qu'on déplace de l'action
        new_elements_list = remove_doubles_list_of_ndarrays(new_elements)
        # print(self.elements_list)
        old_elements_displacement = self.move_elements(permutation_map)
        # print(old_elements_displacement)

        # new_elements_list = [new_elements[i] for i in range(new_elements.shape[0])]
        is_element_matched = self.__compare(old_elements_displacement, new_elements_list)

        for i, el in enumerate(is_element_matched):
            if el == True:
                self.objecthood_prob_list[i] = self.objecthood_prob_list[i] * self.update_rate + (1 - self.update_rate)
            else:
                self.objecthood_prob_list[i] = self.objecthood_prob_list[i] * self.update_rate

        self.elements_list = old_elements_displacement.copy()
        # print(self.elements_list)

        # Remove under threshold
        temp_elements_list = []
        temp_objecthood_prob_list = []
        for i, el in enumerate(self.elements_list):
            if self.objecthood_prob_list[i] > self.rejection_threhold:
                temp_elements_list.append(el)
                temp_objecthood_prob_list.append(self.objecthood_prob_list[i])
        self.elements_list = []
        self.objecthood_prob_list = []                
        self.elements_list = temp_elements_list
        self.objecthood_prob_list = temp_objecthood_prob_list


        # Add new elements
        reverse_element_match =  self.__compare(new_elements_list, old_elements_displacement)
        for i, el in enumerate(reverse_element_match):
            if el == False:
                self.__add_element(new_elements_list[i])





