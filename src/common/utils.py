import numpy as np
from scipy.stats import gaussian_kde
from src.common.sensory_topology import SensoryTopology

# =============================================================================
# Data helpers
# =============================================================================


class ConfigArgs:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __repr__(self):
        return str(self.__dict__)

# =============================================================================
# Configuration & Constants
# =============================================================================


def get_actions_dicts():
    """Returns dictionaries for action vectors and their names."""
    global_action_dict = {
        0: np.array([0, 0]),   # identity
        1: np.array([0, 1]),   # right
        2: np.array([0, -1]),  # left
        3: np.array([1, 0]),   # down
        4: np.array([-1, 0]),  # up
        5: np.array([1, 1]),   # down-right
        6: np.array([-1, 1]),  # up-right
        7: np.array([1, -1]),  # down-left
        8: np.array([-1, -1])  # up-left
    }

    global_action_names_dict = {
        0: "identity",
        1: "right",
        2: "left",
        3: "down",
        4: "up",
        5: "d down right",
        6: "d right up",
        7: "d left down",
        8: "d left up"
    }
    return global_action_dict, global_action_names_dict


def return_probability_strategy(strategy_name: str):
    """
    Returns the action probability distribution for a given strategy name.
    Covers actions: still, up, down, left, right.
    """
    if strategy_name == "random":
        return np.full(5, 1.0 / 5)
    if strategy_name == "mostly_still":
        return np.array([0.5, 0.125, 0.125, 0.125, 0.125])
    if strategy_name == "still":
        return np.array([1., 0., 0., 0., 0.])
    # Default to random if strategy not found
    return np.full(5, 1.0 / 5)

# =============================================================================
# Coordinate & Index Helpers
# =============================================================================


def coordinate_2_index(coordinate, sensor_side_size):
    """Converts a (row, col) coordinate to a flattened index."""
    return int(coordinate[0] * sensor_side_size + coordinate[1])


def index_2_coordinate(index, sensor_side_size):
    """Converts a flattened index to a (row, col) coordinate."""
    return np.array([int(index // sensor_side_size),
                     int(index % sensor_side_size)])

# =============================================================================
# Permutation Map Generation & Usage
# =============================================================================


def generate_permutation_maps_diag(side_size):
    """
    Generate permutation maps for all 8 directions plus identity for
    a square sensor.

    Rows represent predecessors and columns successors. For example, if
    permutation_map[i, j] = 1, it means the sensory value shifted from
    sensel i to j.

    Args:
        side_size (int): The side length of the square sensor.

    Returns:
        np.ndarray: A stack of 9 permutation maps.
    """
    sensor_size = np.array([side_size, side_size])
    total_sensels = side_size ** 2

    id_permutation_map = np.identity(total_sensels)
    downward_permutation_map = np.diag(np.ones(total_sensels - side_size),
                                       k=-side_size)
    upward_permutation_map = downward_permutation_map.T

    # To get right/left shifts, we transpose the coordinate system
    up_2_left_index_swap =\
        np.arange(total_sensels).reshape(sensor_size).T.flatten()
    temp_map = np.diag(np.ones(total_sensels - side_size), k=-side_size)
    rightward_permutation_map =\
        temp_map[:, up_2_left_index_swap][up_2_left_index_swap, :]
    leftward_permutation_map = rightward_permutation_map.T

    # Diagonals are compositions of cardinal movements
    diag_right_down_map =\
        np.matmul(rightward_permutation_map, downward_permutation_map)
    diag_right_up_map =\
        np.matmul(rightward_permutation_map, upward_permutation_map)
    diag_left_down_map =\
        np.matmul(leftward_permutation_map, downward_permutation_map)
    diag_left_up_map =\
        np.matmul(leftward_permutation_map, upward_permutation_map)

    return np.array([
        id_permutation_map, rightward_permutation_map,
        leftward_permutation_map, downward_permutation_map,
        upward_permutation_map, diag_right_down_map,
        diag_right_up_map, diag_left_down_map, diag_left_up_map
    ])


def displace_coord_of_permutation(coordinate, permutation_map):
    """
    Calculates the successor coordinate after applying a permutation map.

    Args:
        coordinate (tuple or np.ndarray): The starting (row, col) coordinate.
        permutation_map (np.ndarray): The permutation map to apply.

    Returns:
        tuple: A tuple containing (successor_coordinate, success_flag).
               Returns (np.empty, False) if no successor exists.
    """
    sensor_side_size = np.sqrt(permutation_map.shape[0])
    index = coordinate_2_index(coordinate, sensor_side_size)

    successor_indices = np.where(permutation_map[index] == 1)[0]
    if successor_indices.size > 0:
        return index_2_coordinate(successor_indices[0], sensor_side_size), True
    return np.empty, False

# =============================================================================
# Image & Noise Manipulation
# =============================================================================


def compute_power_image(img_array):
    """Computes the average power of an image array."""
    return np.mean(img_array.astype(np.float64) ** 2)


def add_noise_on_image(img_array, snr_db):
    """
    Adds Gaussian noise to an image to achieve a target Signal-to-Noise Ratio (SNR).

    """

    img_array_safe = np.nan_to_num(img_array.copy(), nan=0).astype(np.float64)

    image_power = compute_power_image(img_array_safe)
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power, handling the zero-power image case.
    if image_power == 0:
        return img_array_safe.astype(img_array.dtype)

    noise_power_target = image_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power_target), img_array.shape)

    noisy_image = img_array_safe + noise

    return np.clip(noisy_image, 0, 255).astype(img_array.dtype)

# =============================================================================
# Environment Interaction & Object Generation
# =============================================================================


def generate_random_sensor_position(env, border_limit):
    """Generates a random valid sensor position within environment borders."""
    max_pos = np.array(env.grid_size) - border_limit - 1
    min_pos = np.array([border_limit, border_limit])
    return np.random.randint(low=min_pos, high=max_pos)


def generate_random_object(side_size):
    """Generates a square object with random pixel values between 100 and 150."""
    return np.random.randint(100, 150, size=(side_size, side_size))


def generate_random_object_metric(side_size, sensory_distance):
    """Generates a square object using pixel values present in a sensory distance matrix."""
    valid_indices = [
        i for i in range(sensory_distance.shape[0])
        if not np.all(np.isnan(sensory_distance[i, :]))
    ]
    return np.random.choice(valid_indices, size=(side_size, side_size))


def generate_object_from_environment(env, side_size):
    """Extracts a square patch of a given size from a
    random location in the environment."""
    max_pos = np.array(env.grid_size) - side_size
    pos = np.random.randint(low=0, high=max_pos)
    return env.grid[pos[0]:pos[0] + side_size, pos[1]:pos[1] + side_size]

# =============================================================================
# Error & Metric Computations
# =============================================================================


def compute_errors(matrix_1, matrix_2):
    """Computes the absolute element-wise difference between two matrices."""
    return np.abs(matrix_1 - matrix_2)


def compute_prediction_error(prediction, ground_truth, sensory_metric):
    """
    Computes the prediction error using a given sensory metric.

    Args:
        prediction (np.ndarray): The predicted sensory input.
        ground_truth (np.ndarray): The actual sensory input.
        sensory_metric (np.ndarray): A distance matrix for sensory values.

    Returns:
        tuple: (error_map, mean_error_rate)
    """
    non_nan_mask = ~np.isnan(prediction) & ~np.isnan(ground_truth)
    pred_int = prediction[non_nan_mask].astype(int)
    truth_int = ground_truth[non_nan_mask].astype(int)

    error_map = sensory_metric[pred_int, truth_int]
    error_map /= (np.std(ground_truth) + 1e-12)
    error_rate = np.nanmean(error_map)

    return error_map, error_rate


def compute_entropy_prediction_error(prediction, ground_truth, sensory_metric, nb_samples):
    """Computes the entropy of the normalized prediction error distribution."""
    error_map, _ = compute_prediction_error(prediction, ground_truth, sensory_metric)
    error_vector = error_map.flatten()

    if np.all(error_vector == 0) or error_vector.size == 0:
        return 0.0

    kde = gaussian_kde(error_vector)
    x_grid = np.linspace(min(error_vector), max(error_vector), nb_samples)
    pdf = kde(x_grid)
    pdf = np.where(pdf > 0, pdf, 1e-12)  # Ensure pdf values are positive
    pdf /= np.sum(pdf)

    entropy = -np.sum(pdf * np.log(pdf)) *\
        ((max(error_vector) - min(error_vector)) / nb_samples)
    return entropy


def compute_trust_error(prediction, ground_truth,
                        sensory_metric, error_threshold):
    """
    Computes a trust score based on the proportion of well-predicted pixels.
    A high score means low trust (high error).
    """
    error_map, _ = compute_prediction_error(prediction, ground_truth, sensory_metric)
    error_vector = error_map.flatten()

    if error_vector.size == 0:
        return 0.0  # No error if there's nothing to compare

    well_predicted_count = np.sum(error_vector < error_threshold)
    trust_error_score = 1 - (well_predicted_count / len(error_vector))
    return trust_error_score

# =============================================================================
# Main Workflow Functions
# =============================================================================


def compute_metric(grid_world,
                   nb_iter=100,
                   actions_picks=np.arange(5),
                   action_picks_proba=np.full(5, 1.0/5),
                   transformation_picks_proba=np.full(5, 1.0/5)
                   ):
    """
    Computes a sensory distance metric by exploring the environment.
    TODO: Allow for incomplete matrices.
    TODO: Continue to update the matrix as the experience continues => allow to handle new symbols.
    """
    init_frame = grid_world.get_observation_with_noise()
    sensory_topology = SensoryTopology(init_frame, 1)

    for _ in range(nb_iter):
        action = np.random.choice(actions_picks, p=action_picks_proba)
        grid_world.step(action)

        transformation = np.random.choice(actions_picks, p=transformation_picks_proba)
        grid_world.environment_step(transformation)
        sensory_topology.update(grid_world.get_observation_with_noise())

    symbol_distance_matrix = sensory_topology.distance_matrix
    s2v_dict = sensory_topology._symbol_to_value_dict
    value_sensory_metric = np.full((256, 256), np.nan)

    for s1, v1 in s2v_dict.items():
        for s2, v2 in s2v_dict.items():
            value_sensory_metric[v1, v2] = symbol_distance_matrix[s1, s2]

    return value_sensory_metric


def compute_sensory_metric_with_adjusted_snr(grid_world,
                                             nb_iter,
                                             action_prob,
                                             transformation_prob,
                                             snr_db_goal):
    """
    Computes a sensory distance metric while controlling
    the SNR of observations.
    """
    action_pool = np.arange(len(action_prob))
    init_frame = add_noise_on_image(grid_world.observation, snr_db_goal)
    sensory_topology = SensoryTopology(init_frame, 1)

    for i in range(nb_iter):
        if i % 100 == 0:
            grid_world.sensor_position = generate_random_sensor_position(grid_world, 50)

        action = np.random.choice(action_pool, p=action_prob)
        if not grid_world.agent_step(action):
            grid_world.reset_agents_position_to_center()
            grid_world.agent_step(action)

        transformation = np.random.choice(action_pool, p=transformation_prob)
        grid_world.environment_step(transformation)

        new_frame = add_noise_on_image(grid_world.observation, snr_db_goal)
        sensory_topology.update(new_frame)

    symbol_sensory_distance = sensory_topology.distance_matrix
    s2v_dict = sensory_topology._symbol_to_value_dict
    value_sensory_distance = np.full((256, 256), np.nan)

    for s1, v1 in s2v_dict.items():
        for s2, v2 in s2v_dict.items():
            value_sensory_distance[int(v1), int(v2)] = symbol_sensory_distance[s1, s2]

    return value_sensory_distance
