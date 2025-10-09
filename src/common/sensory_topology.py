#!/usr/bin/python3

import numpy as np
import scipy
import sklearn.manifold
import cv2 

class SensoryTopology:
    def __init__(
        self, 
        frame,
        dim,
        patch_size = None,
        patch_loc = None # Set to note to randomly attribute loc
    ):

        self._resolution= frame.shape[0:2]

        if patch_size is not None and patch_loc is None:
            self._patch_size = patch_size
            self._patch_indices = [
                np.random.randint(0, self._resolution[0] - self._patch_size[0]),
                np.random.randint(0, self._resolution[1] - self._patch_size[1])
            ]

        elif patch_size is not None and patch_loc is not None:
            self._patch_size = patch_size
            self._patch_indices = [patch_loc[0], patch_loc[1]]
        else:
            self._patch_size = frame.shape[0:2]
            self._patch_indices = [0, 0]
        
        self._update_patch(frame)

        self._dim = dim
        self._previous_patch = self._patch.copy()
        
        # Association between symbol (attributed by order of apparition)
        self._symbol_to_value_dict= {}
        self._value_to_symbol_dict = {}
        self._init_association_symbol_value()
        self._transition_iter = 0
        self._transition_dict = {}
        # self.__distance_matrix = np.array([])


    def _init_association_symbol_value(self):
        """Initialize the dictionnary value/symbol with the first input frame
        The symbol is the integer associated to the order of apparition
        (starting at 0)
        """
        if self._dim == 1:
            for i in range(self._previous_patch.shape[0]):
                for j in range(self._previous_patch.shape[1]):
                    if self._previous_patch[i,j] not in self._value_to_symbol_dict:
                        new_symbol = len(self._symbol_to_value_dict) 
                        self._symbol_to_value_dict[new_symbol] = self._previous_patch[i,j]
                        self._value_to_symbol_dict[self._previous_patch[i,j]] = new_symbol
        elif self._dim > 1: # If data are not scalars, treat them as tuples
            for i in range(self._previous_patch.shape[0]):
                for j in range(self._previous_patch.shape[1]):
                    if tuple(self._previous_patch[i,j]) not in self._value_to_symbol_dict:
                        new_symbol = len(self._symbol_to_value_dict) 
                        self._symbol_to_value_dict[new_symbol] = tuple(self._previous_patch[i,j])
                        self._value_to_symbol_dict[tuple(self._previous_patch[i,j])] = new_symbol   

    def _value_to_symbol(self, value):
        """If a value is seen then it increase the dict with a new symbol
        """
        if self._dim == 1:
            if value not in self._value_to_symbol_dict:
                new_symbol = len(self._symbol_to_value_dict) 
                self._symbol_to_value_dict[new_symbol] = value
                self._value_to_symbol_dict[value] = new_symbol
                return new_symbol 
            else:
                return self._value_to_symbol_dict[value]
        elif self._dim > 1:
            if tuple(value) not in self._value_to_symbol_dict:
                new_symbol = len(self._symbol_to_value_dict)
                self._symbol_to_value_dict[new_symbol] = tuple(value)
                self._value_to_symbol_dict[tuple(value)] = new_symbol 
                return new_symbol  
            else:
                return self._value_to_symbol_dict[tuple(value)]

    def update(self, next_frame):
        """Takes the next frame and update according to standard rule
        """
        self._update_patch(next_frame)
        self._transition_iter += 1

        for i in range(self._patch_size[0]):
            for j in range(self._patch_size[1]):
                previous_symbol = self._value_to_symbol(self._previous_patch[i,j])
                new_symbol = self._value_to_symbol(self._patch[i,j])
                if previous_symbol not in self._transition_dict:
                    self._transition_dict[previous_symbol] = {}
                self._transition_dict[previous_symbol][new_symbol] = \
                    self._transition_dict[previous_symbol].get(new_symbol, 0) + 1
        self._previous_patch = self._patch
        
    def update_distance_matrix(self):
        self._compute_distance_matrix()
        

    def _update_patch(self, frame):
        """Crop frame according to patch size"""
        self._patch = frame[self._patch_indices[0]:self._patch_indices[0] + self._patch_size[0],\
                            self._patch_indices[1]:self._patch_indices[1] + self._patch_size[1]]

    def _transition_dict_to_matrix(self):
        """Explicit functionning of unique symbols"""
        #TODO may be useless as the symbol dicto_to_values automatically has every symbol seen
        unique_symbols = list(set(self._transition_dict.keys()).union(*self._transition_dict.values())) 
        matrix_size = len(unique_symbols)
        transition_matrix = np.zeros((matrix_size, matrix_size))
        for row, cols in self._transition_dict.items():
            for col, value in cols.items():
                transition_matrix[row, col] = value
        return transition_matrix
    
    def _compute_probability_matrix(self):
        trans_mat = self._transition_dict_to_matrix()
        # # probability_matrix = trans_mat / np.sum(trans_mat, axis = 1, keepdims = True) if np.sum(trans_mat, axis = 1, keepdims = True) else 0
        # b = np.sum(trans_mat, axis = 1, keepdims = True)
        # probability_matrix = np.divide(trans_mat, b, out=np.zeros_like(trans_mat), where=b!=0)
        return trans_mat / np.sum(trans_mat, axis = 1, keepdims = True)
    
    def _compute_distance_matrix(self):
        probability_matrix = self._compute_probability_matrix()
        mask = (probability_matrix != 0)   # création du mask pour prendre les valeurs non nulle
        adjacency_matrix = np.zeros(probability_matrix.shape)
        adjacency_matrix[mask] = -np.log(probability_matrix[mask]) # calcule de la disimilarité
        adjacency_matrix[~mask] = np.inf # prends la valeur la plus faible        
        distance_matrix = scipy.sparse.csgraph.dijkstra(csgraph=adjacency_matrix, directed=True)
        distance_matrix[distance_matrix == np.inf] = distance_matrix[distance_matrix != np.inf].max()/4 # Supress naively the presence of np.nan
        # self.__distance_matrix = distance_matrix
        return distance_matrix
    
    @property
    def transition_matrix(self):
        return self._transition_dict_to_matrix()

    @property
    def probability_matrix(self):
        return self._compute_probability_matrix()
    
    @property
    def distance_matrix(self):
        return self._compute_distance_matrix()
    
    @property
    def patch(self):
        return self._patch
    
    @property
    def symbol_to_value_dict(self):
        return self._symbol_to_value_dict
    
    def _compute_embedding_error(self, m_a, m_b, norm = "fro"):
        if hasattr(self, '_previous_embedding'): 
            # no comparison with potential new points 
            max_nb_points = np.maximum(m_a.shape[0], m_b.shape[0])
            if norm == "fro":
                return np.linalg.norm(
                        m_b[::max_nb_points, :]
                        - m_a[::max_nb_points, :], 
                        ord="fro"
                    )
            
    def _reflect_x(self, data):
        """"""
        reflected_points_x = data.copy()
        reflection_matrix = np.array([[1, 0], [0, -1]])
        for pp in range(data.shape[0]):
            reflected_points_x[pp, :] = np.dot(reflected_points_x[pp, :], 
                                               reflection_matrix
                                            )
        return reflected_points_x
    
    def _reflect_y(self, data):
        """"""
        reflected_points_y = data.copy()
        reflection_matrix = np.array([[-1, 0], [0, 1]])
        for pp in range(data.shape[0]):
            reflected_points_y[pp, :] = np.dot(reflected_points_y[pp, :],
                                               reflection_matrix
                                            )    
        return reflected_points_y
            
    def _perform_homography(self):
        """Method used to perform an homography with previous embedding to 
        create a more readable continuity between the visualization 
        """
        data = self._current_embedding.copy()
        if hasattr(self, '_previous_embedding'): 
            global_error = self._compute_embedding_error(self._previous_embedding,
                                                  self._current_embedding,
                                                  norm = "fro")
            data_reflect_x = self._reflect_x(data)
            error_reflect_x = self._compute_embedding_error(
                self._previous_embedding,
                data_reflect_x
            ) 
            if error_reflect_x < global_error:
                data = data_reflect_x.copy()
            
            data_reflect_y = self._reflect_y(data)
            error_reflect_y = self._compute_embedding_error(
                self._previous_embedding,
                data_reflect_y
            ) 
            if error_reflect_y < np.min([global_error, error_reflect_x]):
                data = data_reflect_y.copy()        
        
            num_points = min(255, self._previous_embedding.shape[0])
            #prends des nombre au hasard dans l'index
            idx_homo = np.random.choice(self._previous_embedding.shape[0],num_points, replace=False) 
            
            M, _ = cv2.estimateAffinePartial2D(
                    self._previous_embedding[idx_homo, :], 
                    self._current_embedding[idx_homo, :]
                )
            
            if M is not None:
                # Extract the rotation/reflection part of the transformation
                # matrix
                rot_scale_mat = M[:, 0:2] 

                # Use Singular Value Decomposition (SVD) to obtain the rotation
                # or reflection matrix
                U, _, VT = np.linalg.svd(rot_scale_mat)
                R = np.dot(U, VT) # Compute the rotation 

                # Construct the "rigidified" transformation matrix
                M_rigid = np.zeros_like(M)
                M_rigid[:, 0:2] = R
                M_rigid[:, 2] = M[:, 2]
                
                # Convert to 3x3 matrix
                M_rigid_homog = np.vstack([M_rigid, [0, 0, 1]]) 
                # Invert the transformation
                h_inv = np.linalg.inv(M_rigid_homog) 
                
                data_avec_rotation = data.copy()
                for pp in range(data.shape[0]):
                    # Rajoute une dimension avec 1 (coordonnées homogènes)
                    point_homo = np.hstack((self._current_embedding[pp, :], 1))
                    point_transform = h_inv @ point_homo
                    # Normalize
                    point_transform = point_transform / point_transform[-1]
                    # Projette avec h_inv
                    data_avec_rotation[pp, :] = point_transform[:2]
       
                # reflexion X + reflexion Y + rotation
                # erreur juste avec les rotations
                error_avec_rotation = self._compute_embedding_error(
                        self._previous_embedding, 
                        data_avec_rotation
                    ) 
                if error_avec_rotation < np.min([error_reflect_x, error_reflect_y, global_error ]):
                    data = data_avec_rotation.copy()
            self._current_embedding = data.copy()
            

    def get_mds_embedding(self, dim = 2):
        symmetric_distance_matrix = 0.5 * (self.distance_matrix + self.distance_matrix.T)
        mds_embedding_obj = sklearn.manifold.MDS(n_components = dim, dissimilarity = "precomputed")
        colors = [tuple([_/255., _/255., _/255.]) for _ in self._symbol_to_value_dict.values()]
        mds_embedding = mds_embedding_obj.fit_transform(symmetric_distance_matrix)
        self._current_embedding = mds_embedding
        self._perform_homography()
        self._previous_embedding = mds_embedding
        return mds_embedding, colors
    
    def get_isomap_embedding(self, dim = 2):
        symmetric_distance_matrix = 0.5 * (self.distance_matrix + self.distance_matrix.T)
        mds_embedding = sklearn.manifold.Isomap(n_components = dim, dissimilarity = "precomputed")
        colors = [tuple([_/255., _/255., _/255.]) for _ in self._symbol_to_value_dict.values()]

        return mds_embedding.fit_transform(symmetric_distance_matrix), colors
    
    def v2s(self, value):
        return self._value_to_symbol(value)
    def s2v(self, symbol):
        return self._symbol_to_value(symbol)
    
    def _symbol_to_value(self, symbol):
        """If a value is seen then it increase the dict with a new symbol
        """
        if self._dim == 1:

            if symbol not in self._symbol_to_value_dict:
                print("Erros symbol not known encountered")
                exit()
            else:
                return self._symbol_to_value_dict[symbol]
        elif self._dim > 1:
            if symbol not in self._symbol_to_value_dict:
                print("Erros symbol not known encountered")
                exit()
            else:
                return self._symbol_to_value_dict[symbol]
        
    def get_symbols_distance(self, s1, s2):
        return self.distance_matrix[s1, s2]
    


    