#!/usr/bin/env python3

# Local imports
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from datetime import datetime
import argparse
import os


import src.common.utils
from src.common.grid_world_environment_object import GridWorldEnvironment
from src.common.sensory_predictor import SensoryPredictor
from src.common.proto_object import ProtoObject

# def generate_random_object_metric(side_size, sensory_distance):

#     valid_indices = [i for i in range(sensory_distance.shape[0]) if not all([np.isnan(sensory_distance[i, j]) for j in range(sensory_distance.shape[1])])]
#     object_appearance = np.random.choice(valid_indices, size=(side_size, side_size))
#     return object_appearance


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2*(precision*recall)/(precision + recall)


def get_extended_coordinates(coordinates):
    list_of_dir = [np.array([0, 1]),
                   np.array([0, -1]),
                   np.array([1, 0]),
                   np.array([-1, 0])]
    new_coordinates = []
    for c in coordinates:
        new_coordinates.append(c)
        for dir in list_of_dir:
            if not any([np.array_equal(c + dir, coo) for coo in coordinates]):
                new_coordinates.append(c + dir)
    return new_coordinates

# def compute_power_image(img_array):
#     squarred_image = img_array**2
#     return np.nanmean(squarred_image)

# def add_noise_on_image(img_array, snr_db):

#     img_array_safe = np.nan_to_num(img_array, nan=0)
#     image_power = compute_power_image(img_array_safe)
#     snr_target = np.power(10, snr_db/10)

#     if image_power == 0:
#         noise_power_target = 0
#     else:
#         noise_power_target = image_power / snr_target

#     if noise_power_target < 0:
#         noise_power_target = 0

#     noise = np.int16(np.random.randn(
#         img_array.shape[0], img_array.shape[1]) * np.sqrt(noise_power_target))

#     return np.clip(img_array + noise, a_min=0, a_max=255)


def parse_arguments():

    global TOTAL_NUMBER_OF_TRANSFORMATIONS
    parser = argparse.ArgumentParser(description="This script computes error of inference with an"
                                     " object with increasing object size.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")

    parser.add_argument("--nb_iter_metric", type=int, default=1000,
                        help="Number of steps to compute metric")
    # parser.add_argument("--snr", type=int, default=40, help="Signal to noise ratio to apply to the image")

    parser.add_argument(
        '--snr_array',
        type=int,
        nargs='+',
        choices=range(0, 105, 10),
        default=np.array([0, 25, 50]),
        help=f"List of SNR to apply the inference process for each object size)"
    )

    parser.add_argument("--nb_iter_inference_per_object", type=int, default=100,
                        help="Number of steps to infer per object")
    parser.add_argument("--camera_side_size", type=int, default=10,
                        help="Size of the side of the square camera sensor")
    parser.add_argument(
        '--actions',
        type=int,
        nargs='+',
        choices=range(TOTAL_NUMBER_OF_TRANSFORMATIONS),
        default=list(range(TOTAL_NUMBER_OF_TRANSFORMATIONS)),
        help=f"List of actions indices to perform (0-{TOTAL_NUMBER_OF_TRANSFORMATIONS-1})"
    )
    parser.add_argument(
        '--transformations',
        type=int,
        nargs='+',
        choices=range(TOTAL_NUMBER_OF_TRANSFORMATIONS),
        default=list(range(TOTAL_NUMBER_OF_TRANSFORMATIONS)),
        help=f"List of actions indices to perform (0-{TOTAL_NUMBER_OF_TRANSFORMATIONS-1})"
    )

    parser.add_argument("--object_side_size", type=int, default=2,
                        help="Size in pixels of each side of the object")

    parser.add_argument("--rejection_threshold", type=float, default=2,
                        help="")
    parser.add_argument("--update_rate", type=float, default=2,
                        help="")
    parser.add_argument("--initial_likelihood", type=float, default=2,
                        help="")
    parser.add_argument("--object_rejection_threshold", type=float, default=2,
                        help="")

    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed")

    parser.add_argument("--save_dir", type=str,
                        help="Directory to save everything")

    return parser.parse_args()


def get_save_path(args, HEAD):
    """Make save path for whatever results.
    """
    # date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    date = '{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    seedstr = str(args.seed).zfill(3)
    # suffix = "globalInference_{}dB_ actions {}_ trans {}_{}".format(args.snr, args.actions, args.transformations, date, seedstr)
    suffix = "singleObjectInference_{}pxl_{}iter_{}_seed{}".format(
        args.camera_side_size, args.nb_iter_inference_per_object, date, seedstr)

    result_path = os.path.join(HEAD, suffix)
    return result_path


def turn_rgb_imagio_into_grayscale_ndarray(image):
    return np.asarray(image).mean(axis=2)


def get_center_coordinates_of_image(shape):
    return np.array([int(shape[0]/2), int(shape[1]/2)])


def object_is_in_margins(object_current_pos, camera_side_size, object_side_size, margin):
    return object_current_pos[0] <= margin \
        or object_current_pos[0] >= camera_side_size - margin - object_side_size \
        or object_current_pos[1] <= margin \
        or object_current_pos[1] >= camera_side_size - margin - object_side_size


def object_inside_margins(object_position, camera_side_size, object_side_size, margin):
    return margin <= object_position[0] <= camera_side_size - object_side_size - margin \
        and margin <= object_position[1] <= camera_side_size - object_side_size - margin


def compute_error_map_with_sensory_distance(previous_observation, observation, sensory_distance):
    # Sanitize the inputs immediately by replacing any NaNs with a safe integer value.
    # We use a value that will not be a valid index, like -1, to be safe.
    # The reshape(-1) flattens the array.
    prev_flat_safe = np.nan_to_num(
        previous_observation.reshape(-1), nan=-1).astype(np.int16)
    next_flat_safe = np.nan_to_num(
        observation.reshape(-1), nan=-1).astype(np.int16)

    # We now have two arrays of type int16 with no NaNs.
    # But, what about the values? If they are -1, they are not valid indices.
    # We must create a new mask to handle these sentinel values.
    valid_mask = (prev_flat_safe >= 0) & (next_flat_safe >= 0)

    # Initialize the results array. It should be a floating-point array
    # to be able to hold NaN values.
    results = np.full_like(prev_flat_safe, np.nan, dtype=np.float64)

    # Use the valid_mask to apply the sensory distance lookup only for valid indices
    results[valid_mask] = sensory_distance[
        prev_flat_safe[valid_mask],
        next_flat_safe[valid_mask]
    ]

    return results


def object_detection_metrics(relevant_elements_list, proto_objets_detected):
    # relevant_elements_list = coordinates_list
    true_positives_among_relevant = \
        [any(np.array_equal(a, b) for b in proto_objets_detected.elements_list)
         for a in relevant_elements_list]
    true_positives_among_detected = \
        [any(np.array_equal(a, b) for b in relevant_elements_list)
         for a in proto_objets_detected.elements_list]
    # true_positives = \
    #     true_positives_among_detected[np.where(true_positives_among_detected == True)]
    # likelihood_sum_of_true_positives = 0
    # for i, el in enumerate(true_positives_among_detected):
    #     if el:
    #         likelihood_sum_of_true_positives += proto_objets_detected.likelihood_list[i]
    # weighted_recall = likelihood_sum_of_true_positives / len(relevant_elements_list)
    # weighted_precision = likelihood_sum_of_true_positives / np.sum(proto_objets_detected.likelihood_list)
    # weighted_f1 = f1_score(weighted_precision, weighted_recall)

    # Check if the likelihood list is empty to prevent division by zero
    if not proto_objets_detected.objecthood_prob_list:
        return 0, 0, 0  # Return a default value for metrics

    objecthood_prob_sum_of_true_positives = 0
    for i, el in enumerate(true_positives_among_detected):
        if el:
            objecthood_prob_sum_of_true_positives += proto_objets_detected.objecthood_prob_list[i]

    weighted_recall = objecthood_prob_sum_of_true_positives / \
        len(relevant_elements_list)

    # Check if the sum of likelihoods is zero
    objecthood_prob_sum = np.sum(proto_objets_detected.objecthood_prob_list)
    if objecthood_prob_sum == 0:
        weighted_precision = 0
    else:
        weighted_precision = objecthood_prob_sum_of_true_positives / objecthood_prob_sum

    weighted_f1 = f1_score(weighted_precision, weighted_recall)

    return weighted_recall, weighted_precision, weighted_f1


# GLOBAL VARIABLE
TOTAL_NUMBER_OF_TRANSFORMATIONS = 9


def run_experiment(args):
    print("-- Experiment Script is Running --")

    # SETUPS
    np.random.seed(seed=args.seed)
    sensor_size = np.array([args.camera_side_size, args.camera_side_size])
    nb_iter_inference_per_object = args.nb_iter_inference_per_object
    snr_db_array = args.snr_array
    global_action_dict, _ = src.common.utils.get_actions_dicts()
    global TOTAL_NUMBER_OF_TRANSFORMATIONS
    image_grid = np.asarray(imageio.imread(args.image_env_path)).mean(axis=2)
    image_center = get_center_coordinates_of_image(image_grid.shape)

    nb_actions = len(args.actions)
    action_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    action_strategy[args.actions] = 1./nb_actions

    nb_transformations = len(args.transformations)
    transformation_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    transformation_strategy[args.transformations] = 1./nb_transformations

    object_shape = np.array([args.object_side_size, args.object_side_size])
    rejection_threshold = args.rejection_threshold
    update_rate = args.update_rate
    object_rejection_threshold = args.object_rejection_threshold
    initial_objecthood_probability = args.initial_objecthood_probability

    # CONSTANTS
    border_limit = 5
    action_2_transformation = np.array([0, 2, 1, 4, 3])
    margin_object_boundaries = 1

    global_permutation_maps = src.common.utils.generate_permutation_maps_diag(
        args.camera_side_size)
    sensory_predictor = SensoryPredictor(
        global_permutation_maps, side_size=args.camera_side_size)
    environment = GridWorldEnvironment(global_action_dict,
                                       sensor_size=sensor_size,
                                       grid=image_grid,
                                       sensor_position=image_center
                                       )

    # FLAGS
    is_object_detected = False

    # RESULTS VARIABLES
    object_transformation_list = []
    action_inferred_list = []
    w_recall_list = []
    w_precision_list = []
    f1_score_list = []

    ext_w_recall_list = []
    ext_w_precision_list = []
    ext_f1_score_list = []

    mixed_f1_score_list = []

    agents_actions_array = np.full(nb_iter_inference_per_object, -1, dtype=int)

    for i_snr, snr_db in enumerate(snr_db_array):

        sensory_distance = np.load(os.path.join(os.getcwd(),
                                                "data",
                                                "sensory_metrics",
                                                f"sensory_distance_{snr_db}db.npy"))
        initial_object_position = get_center_coordinates_of_image(sensor_size)
        environment.empty_object_list()
        environment.sensor_position = src.common.utils.generate_random_sensor_position(environment,
                                                                                       border_limit=args.object_side_size*border_limit)
        object_generated = src.common.utils.generate_random_object_metric(
            args.object_side_size, sensory_distance)
        environment.add_object(initial_object_position, object_generated)
        environment.set_object_position_agents_pov(object_index=0,
                                                   position_agents_frame=initial_object_position)

        for iter in range(nb_iter_inference_per_object):

            previous_observation = src.common.utils.add_noise_on_image(
                environment.observation, snr_db)

            agents_action = np.random.choice(np.arange(len(action_strategy)),
                                             p=action_strategy)
            agents_actions_array[iter] = agents_action

            if is_object_detected:
                proto_object.move_elements(
                    global_permutation_maps[agents_action])

            # PREDICTION
            predicted_observation = sensory_predictor.predict(
                previous_observation, agents_action)

            # OBJECT TRANSFORMATION
            # Assure transformation maintain objects within margin_object_boundaries of sensor
            object_current_pos = environment.get_object_position_agents_pov(
                object_index=0)
            if object_is_in_margins(object_current_pos,
                                    args.camera_side_size,
                                    args.object_side_size,
                                    margin_object_boundaries):
                available_transformations = []
                available_transformations_prob = []

                for transformation, transformation_prob in enumerate(transformation_strategy):
                    hypothetical_object_pos = object_current_pos + \
                        global_action_dict[transformation]
                    if transformation_strategy[transformation] != 0 \
                        and object_inside_margins(hypothetical_object_pos,
                                                  args.camera_side_size,
                                                  args.object_side_size,
                                                  margin_object_boundaries):
                        available_transformations.append(transformation)
                        available_transformations_prob.append(
                            transformation_prob)
                if available_transformations:
                    # print("#STOP#")
                    # print(transformation_strategy)
                    # print(available_transformations)

                    adjusted_transformation_prob = \
                        transformation_strategy[available_transformations] / np.sum(
                            transformation_strategy[available_transformations])

                    object_transformation = np.random.choice(available_transformations,
                                                             p=adjusted_transformation_prob)

                    environment.move_object(
                        object_index=0, transformation=object_transformation)
                else:
                    print(f"Error: object blocked")
            else:
                object_transformation = np.random.choice(args.transformations,
                                                         p=transformation_strategy[args.transformations])
                environment.move_object(
                    object_index=0, transformation=object_transformation)

            object_transformation_list.append(object_transformation)

            environment.agent_step(agents_action)
            observation = src.common.utils.add_noise_on_image(
                environment.observation, snr_db)
            error_map = compute_error_map_with_sensory_distance(predicted_observation,
                                                                observation,
                                                                sensory_distance)

            # HYPOTHESIS EVALUATION
            error_improvements = []
            for i in range(len(args.actions)):
                hypothetical_prediction = sensory_predictor.predict(
                    predicted_observation, i)
                hypothetical_error_map = compute_error_map_with_sensory_distance(observation,
                                                                                 hypothetical_prediction,
                                                                                 sensory_distance)
                mask = error_map > rejection_threshold
                if np.any(mask):
                    error_improvement_map = error_map[mask] - \
                        hypothetical_error_map[mask]
                    error_improvements.append(np.nanmean(
                        np.nan_to_num(error_improvement_map, nan=0)))
                else:
                    error_improvements.append(0)

            error_improvements_safe = [np.nan_to_num(
                x, nan=-1e10) for x in error_improvements]
            inferred_action = np.argmax(error_improvements_safe)
            action_inferred_list.append(inferred_action)

            # OBJECT IDENTIFICATION THROUGH SUCCESSOR MAPS

            irregularities = (error_map > rejection_threshold).flatten()

            score = 0
            object_pos = []

            for irr in np.where(irregularities)[0]:
                # If there exists a successor in the permutation map
                irr_index = np.array(
                    (irr // args.camera_side_size, irr % args.camera_side_size))
                if np.where(global_permutation_maps[inferred_action, irr, :])[0].size != 0:
                    post_irr = np.squeeze(
                        np.where(global_permutation_maps[inferred_action, irr, :]))
                    post_irr_index = np.array(
                        (post_irr // args.camera_side_size, post_irr % args.camera_side_size))

                    predicted_val = predicted_observation[irr_index[0],
                                                          irr_index[1]]
                    observation_val = observation[post_irr_index[0],
                                                  post_irr_index[1]]

                    # Use np.isfinite to check for NaNs and Infs before comparison
                    if np.isfinite(predicted_val) and np.isfinite(observation_val):
                        if np.abs(predicted_val - observation_val) < rejection_threshold:
                            score += 1
                            object_pos.append(
                                np.array([post_irr_index[0], post_irr_index[1]]))
                # If there exists a predecessor in the permutation map
                if np.where(global_permutation_maps[inferred_action, :, irr])[0].size != 0:
                    pre_irr = np.squeeze(
                        np.where(global_permutation_maps[inferred_action, :, irr]))
                    pre_irr_index = np.array(
                        (pre_irr // args.camera_side_size, pre_irr % args.camera_side_size))

                    predicted_val = predicted_observation[pre_irr_index[0],
                                                          pre_irr_index[1]]
                    observation_val = observation[irr_index[0], irr_index[1]]

                    if np.isfinite(predicted_val) and np.isfinite(observation_val):
                        if np.abs(predicted_val - observation_val) < rejection_threshold:
                            score += 1
                            object_pos.append(
                                np.array([irr_index[0], irr_index[1]]))

            # OBJECT
            # Verify if the object has been detected
            if object_pos:
                pos_stacks = np.stack(object_pos)
            if 'pos_stacks' not in locals():
                print("\'pos_stacks\' not assigned yet")
                # Fill with nan to have a single size of results
                w_recall_list.append(np.nan)
                w_precision_list.append(np.nan)
                f1_score_list.append(np.nan)
                ext_w_recall_list.append(np.nan)
                ext_w_precision_list.append(np.nan)
                ext_f1_score_list.append(np.nan)
                mixed_f1_score_list.append(np.nan)
            else:
                if not is_object_detected:
                    is_object_detected = True
                    proto_object = ProtoObject(pos_stacks,
                                               update_rate=update_rate,
                                               rejection_threshold=object_rejection_threshold,
                                               initial_objecthood_probability=initial_objecthood_probability)
                else:
                    proto_object.update(
                        pos_stacks, global_permutation_maps[inferred_action])

                true_object_position = environment.get_object_position_agents_pov(
                    object_index=0)
                true_objects_elements = [
                    np.array([0, 0])]*(args.object_side_size**2)
                true_objects_elements[0] = true_object_position
                rows = np.arange(
                    true_object_position[0], true_object_position[0] + args.object_side_size)
                cols = np.arange(
                    true_object_position[1], true_object_position[1] + args.object_side_size)

                rr, cc = np.meshgrid(rows, cols, indexing='ij')
                coordinates = np.stack([rr.ravel(), cc.ravel()], axis=1)
                coordinates_list = [coordinates[i]
                                    for i in range(len(coordinates))]

                # UPDATE RESULT METRICS

                weighted_recall, weighted_precision, f1_score_value = \
                    object_detection_metrics(coordinates_list, proto_object)

                w_recall_list.append(weighted_recall)
                w_precision_list.append(weighted_precision)
                f1_score_list.append(f1_score_value)

                extended_coordinates = get_extended_coordinates(
                    coordinates_list)

                ext_weighted_recall, ext_weighted_precision, ext_f1_score_value = \
                    object_detection_metrics(
                        extended_coordinates, proto_object)

                ext_w_recall_list.append(ext_weighted_recall)
                ext_w_precision_list.append(ext_weighted_precision)
                ext_f1_score_list.append(ext_f1_score_value)

                mixed_f1_score_list.append(
                    f1_score(ext_weighted_precision, weighted_recall))

    inferred_transformation_array = action_2_transformation[np.asarray(
        action_inferred_list)]
    accuracy = len(np.where(object_transformation_list == inferred_transformation_array)[0])\
        / nb_iter_inference_per_object

    df_rows = []

    for trial in range(nb_iter_inference_per_object):
        df_rows.append({
            "snr": snr_db_array[0],
            "object_side_size": args.object_side_size,
            "trial": trial,
            "accuracy": accuracy,
            "recall": w_recall_list[trial],
            "precision": w_precision_list[trial],
            "f1": f1_score_list[trial],
            "ext_precision": ext_w_precision_list[trial],
            "ext_recall": ext_w_recall_list[trial],
            "ext_f1": ext_f1_score_list[trial],
            "mixed_f1": mixed_f1_score_list[trial],
            "object_inferred_transformation_array": inferred_transformation_array[trial],
            "agents_action": agents_actions_array[trial],
        })

    df = pd.DataFrame(df_rows)
    df_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(df_path, index=False)
    print("-- Experiment Script Finished --")


def main():
    # INPUTS
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
