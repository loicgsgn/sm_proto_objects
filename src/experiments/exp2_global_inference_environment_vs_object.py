#!/usr/bin/env python3
# Basic Modules
import os
import argparse


import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd


import src.common.utils
from src.common.grid_world_environment_object import GridWorldEnvironment
from src.common.sensory_predictor import SensoryPredictor

TOTAL_NUMBER_OF_TRANSFORMATIONS = 9


def parse_arguments():
    global TOTAL_NUMBER_OF_TRANSFORMATIONS
    parser = argparse.ArgumentParser(description="This script computes error of inference with an"
                                     " object with increasing object size.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")

    parser.add_argument("--nb_iter_metric", type=int, default=1000,
                        help="Number of steps to compute metric")

    parser.add_argument(
        '--snr_array',
        type=int,
        nargs='+',
        choices=range(0, 105, 5),
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

    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed")

    parser.add_argument("--save_dir", type=str,
                        help="Directory to save everything")
    return parser.parse_args()


def run_experiment(args):
    print("-- Experiment Script is Running --")

    # SETUPS
    np.random.seed(seed=args.seed)
    sensor_size = np.array([args.camera_side_size, args.camera_side_size])
    image_grid = np.asarray(imageio.imread(args.image_env_path)).mean(axis=2)
    global_actions_dict, _ = src.common.utils.get_actions_dicts()
    global_permutations_maps = src.common.utils.generate_permutation_maps_diag(
        args.camera_side_size)

    image_center = np.array([int(image_grid.shape[0]/2),
                             int(image_grid.shape[1]/2)])
    snr_db_array = args.snr_array
    object_side_size_array = np.arange(0, args.camera_side_size + 1)

    global TOTAL_NUMBER_OF_TRANSFORMATIONS

    nb_actions = len(args.actions)
    action_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    action_strategy[args.actions] = 1./nb_actions

    nb_transformations = len(args.transformations)
    transformation_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    transformation_strategy[args.transformations] = 1./nb_transformations

    border_limit = 5

    environment = GridWorldEnvironment(global_actions_dict,
                                       sensor_size=sensor_size,
                                       grid=image_grid,
                                       sensor_position=image_center
                                       )

    sensory_predictor = SensoryPredictor(
        global_permutations_maps, side_size=args.camera_side_size)
    # RESULTS ARRAYS

    agents_actions_array = np.full((len(snr_db_array),
                                    len(object_side_size_array),
                                    args.nb_iter_inference_per_object), -1, dtype=int)

    environments_transformations_array = np.full((len(snr_db_array),
                                                 len(object_side_size_array),
                                                 args.nb_iter_inference_per_object), -1, dtype=int)

    objects_transformations_array = np.full((len(snr_db_array),
                                             len(object_side_size_array),
                                             args.nb_iter_inference_per_object), -1, dtype=int)

    hypothetical_prediction_error_per_action = np.full((len(snr_db_array),
                                                        len(object_side_size_array),
                                                        args.nb_iter_inference_per_object,
                                                        nb_actions
                                                        ), -1, dtype=np.float64)

    inferred_agents_actions_array = np.full((len(snr_db_array),
                                             len(object_side_size_array),
                                             args.nb_iter_inference_per_object), -1, dtype=int)

    # EXPERIENCE
    for i_snr, snr_db in enumerate(snr_db_array):

        sensory_distance = np.load(os.path.join(os.getcwd(),
                                                "data",
                                                "sensory_metrics",
                                                f"sensory_distance_{snr_db}db.npy"))

        for i_object_side_size, object_side_size in enumerate(object_side_size_array):
            print(f"SNR : {snr_db}, Object side size : {object_side_size}")
            initial_object_position = np.array([int(sensor_size[0]/2 - object_side_size/2),
                                                int(sensor_size[1]/2 - object_side_size/2)])
            environment.empty_object_list()

            if object_side_size != 0:
                environment.add_object(initial_object_position,
                                       src.common.utils.generate_random_object_metric(object_side_size,
                                                                                      sensory_distance)
                                       )
            for iter in range(args.nb_iter_inference_per_object):
                environment.sensor_position = src.common.utils.generate_random_sensor_position(environment,
                                                                                               border_limit=object_side_size*border_limit)

                if object_side_size != 0:
                    environment.set_object_position_agents_pov(
                        0, initial_object_position)
                    object_appearance = src.common.utils.generate_random_object_metric(object_side_size,
                                                                                       sensory_distance)
                    environment.change_object_appearance(0, object_appearance)

                previous_observation = src.common.utils.add_noise_on_image(
                    environment.observation, snr_db)

                agents_action = np.random.choice(np.arange(len(action_strategy)),
                                                 p=action_strategy)
                # agents_actions_array[i_snr, i_object_side_size, iter] = agents_action

                if not environment.agent_step(agents_action):
                    # In case agent's out of bound reset agent's pos to center of environment
                    environment.reset_agents_position_to_center()
                    if object_side_size != 0:
                        environment.set_object_position_agents_pov(
                            0, initial_object_position)
                    previous_observation = src.common.utils.add_noise_on_image(
                        environment.observation, snr_db)
                    environment.agent_step(agents_action)

                agents_actions_array[i_snr,
                                     i_object_side_size, iter] = agents_action

                # PREDICTION
                predicted_observation = sensory_predictor.predict(
                    previous_observation, agents_action)

                # ENVIRONMENT TRANSFORMATION
                environment_transformation = np.random.choice(np.arange(len(transformation_strategy)),
                                                              p=transformation_strategy)
                environment.environment_step(environment_transformation)
                environments_transformations_array[i_snr, i_object_side_size, iter] = \
                    environment_transformation

                # OBJECT TRANSFORMATION
                if object_side_size != 0:
                    object_transformation = np.random.choice(np.arange(len(transformation_strategy)),
                                                             p=transformation_strategy)
                    environment.move_object(
                        object_index=0, transformation=object_transformation)
                    objects_transformations_array[i_snr, i_object_side_size, iter] = \
                        object_transformation

                observation = src.common.utils.add_noise_on_image(
                    environment.observation, snr_db)

                # INFERENCE
                hypothetical_predictions = np.zeros((nb_actions,
                                                     args.camera_side_size,
                                                     args.camera_side_size))

                # Compute hypothesis of predictions
                for i_action, action in enumerate(args.actions):
                    hypothetical_predictions[i_action] = \
                        sensory_predictor.predict(
                            predicted_observation, action)

                    _, error_rate = src.common.utils.compute_prediction_error(
                        hypothetical_predictions[i_action],
                        observation,
                        sensory_distance)

                    hypothetical_prediction_error_per_action[i_snr, i_object_side_size, iter, i_action] = \
                        error_rate
                # Inference
                i_action_with_min_pred_error = \
                    np.argmin(
                        hypothetical_prediction_error_per_action[i_snr, i_object_side_size, iter])

                action_with_min_pred_error = args.actions[i_action_with_min_pred_error]
                inferred_agents_actions_array[i_snr,
                                              i_object_side_size, iter] = action_with_min_pred_error

    # SAVING
    action_2_transformation = np.array([0, 2, 1, 4, 3])
    inferred_transformations_array = action_2_transformation[inferred_agents_actions_array]

    df_rows = []

    for i_snr, snr in enumerate(snr_db_array):
        for i_object_size, object_side_size in enumerate(object_side_size_array):
            for step in range(args.nb_iter_inference_per_object):
                df_rows.append({
                    "snr": snr,
                    "object_side_size": object_side_size,
                    "trial": step,
                    "environment_transformation": environments_transformations_array[i_snr, i_object_size, step],
                    "object_transformation": objects_transformations_array[i_snr, i_object_size, step],
                    "transformation_inferred": inferred_transformations_array[i_snr, i_object_size, step],
                    "agents_action": agents_actions_array[i_snr, i_object_size, step],
                })
    df = pd.DataFrame(df_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    print("-- End of Experiment --")


def main():
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
