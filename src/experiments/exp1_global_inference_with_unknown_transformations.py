#!/usr/bin/python3
import os
import argparse
import yaml

import numpy as np
import imageio.v2 as imageio
import pandas as pd

import src.common.utils
from src.common.grid_world_environment import GridWorldEnvironment
from src.common.sensory_predictor import SensoryPredictor


TOTAL_NUMBER_OF_TRANSFORMATIONS = 9


def parse_arguments():
    global TOTAL_NUMBER_OF_TRANSFORMATIONS
    parser = argparse.ArgumentParser(
        description="This script computes error of inference.")

    parser.add_argument("--nb_iter_metric", type=int, default=1000,
                        help="Number of steps to compute metric")
    parser.add_argument("--snr", type=int, default=40,
                        help="Signal to noise ratio to apply to the image")
    parser.add_argument("--nb_iter_inference", type=int, default=10,
                        help="Number of steps to infer")
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
    parser.add_argument("--image_env_path", type=str,
                        help="Path to image used for the environment")

    parser.add_argument("--output_dir", type=str,
                        help="Directory to save everything")

    return parser.parse_args()


def run_experiment(args):
    print("-- Experiment Script is Running --")

    global TOTAL_NUMBER_OF_TRANSFORMATIONS
    np.random.seed(seed=args.seed)
    global_action_dict, _ = src.common.utils.get_actions_dicts()
    sensor_size = np.array([args.camera_side_size, args.camera_side_size])
    image_grid = np.asarray(imageio.imread(args.image_env_path)).mean(axis=2)
    image_center = np.array([int(image_grid.shape[0]/2),
                             int(image_grid.shape[1]/2)])
    index_actions = args.actions
    nb_actions = len(index_actions)
    action_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    action_strategy[index_actions] = 1./nb_actions

    index_transformations = args.transformations
    nb_transformations = len(index_transformations)
    transformation_strategy = np.full(TOTAL_NUMBER_OF_TRANSFORMATIONS, 0.)
    transformation_strategy[index_transformations] = 1./nb_transformations

    global_permutation_maps = src.common.utils.generate_permutation_maps_diag(
        args.camera_side_size)

    sensory_predictor = SensoryPredictor(
        global_permutation_maps, side_size=args.camera_side_size)

    environment = GridWorldEnvironment(
        global_action_dict,
        sensor_size=sensor_size,
        sensor_position=image_center,
        grid=image_grid,
    )
    print("-- Sensory Distance Building Process --")
    sensory_distance = src.common.utils.compute_sensory_metric_with_adjusted_snr(environment,
                                                                                 args.nb_iter_metric,
                                                                                 action_strategy,
                                                                                 transformation_strategy,
                                                                                 args.snr)

    # Uncomment to load a metric instead of computing it
    # sensory_distance =np.load(os.path.join(os.getcwd(),
    #                                         "data",
    #                                         "sensory_metrics",
    #                                         f"sensory_distance_{args.snr}db.npy"))

    print("-- Sensory Distance Building Process Finished --")

    # Results Arrays
    environments_transformations_array = np.full(
        args.nb_iter_inference, -1, dtype=int)
    agents_actions_array = np.full(args.nb_iter_inference, -1, dtype=int)
    inferred_actions_array = np.full(args.nb_iter_inference, -1, dtype=int)
    counterfactual_prediction_error_per_action = np.full(
        (args.nb_iter_inference, nb_actions), -1, dtype=float)

    # Experience
    print("-- Mean Prediction Error Computation Process --")
    for i in range(args.nb_iter_inference):
        if args.verbose:
            print(f"step : {i}/{args.nb_iter_inference}")
        agents_action = np.random.choice(
            np.arange(len(action_strategy)), p=action_strategy)
        agents_actions_array[i] = agents_action

        if i == 0:
            previous_observation = src.common.utils.add_noise_on_image(
                environment.get_observation(), args.snr)

        predicted_observation = sensory_predictor.predict(
            previous_observation, agents_action)

        if not environment.agent_step(agents_action):
            environment.reset_position()
            previous_observation = src.common.utils.add_noise_on_image(
                environment.get_observation(), args.snr)
            environment.agent_step(agents_action)

        environments_transformation = np.random.choice(np.arange(len(transformation_strategy)),
                                                       p=transformation_strategy)

        environment.environment_step(environments_transformation)
        environments_transformations_array[i] = environments_transformation

        observation = src.common.utils.add_noise_on_image(
            environment.get_observation(), args.snr)

        counterfactual_predictions = np.zeros(
            (nb_actions, args.camera_side_size, args.camera_side_size))
        for idx_action, action in enumerate(index_actions):

            counterfactual_predictions[idx_action] = sensory_predictor.predict(
                predicted_observation, action)
            _, error_rate = src.common.utils.compute_prediction_error(
                counterfactual_predictions[idx_action], observation, sensory_distance)
            counterfactual_prediction_error_per_action[i,
                                                       idx_action] = error_rate

        idx_with_min_cntr_pred_error = np.argmin(
            counterfactual_prediction_error_per_action[i])
        action_with_min_cntr_pred_error = index_actions[idx_with_min_cntr_pred_error]
        inferred_actions_array[i] = action_with_min_cntr_pred_error

        previous_observation = observation

    print("-- Mean Prediction Error Computation Process Finished --")

    df = pd.DataFrame({
        "environment_transformation": environments_transformations_array,
        "inferred_actions": inferred_actions_array,
        "counterfactual_prediction_error_per_action":
            [row.tolist() for row in counterfactual_prediction_error_per_action]
    })

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    with open(os.path.join(args.output_dir, "experiment_args.yaml"), 'w') as f:
        yaml.dump(vars(args), f, indent=4)

    print("-- End of Experiment --")


def main():
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
