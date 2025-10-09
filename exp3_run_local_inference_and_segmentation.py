import os
import yaml
import sys
import src.experiments.exp3_local_inference_and_segmentation
import src.common.utils
import itertools


script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

with open(os.path.join(".",
                       "config",
                       "exp3_local_inference_and_segmentation.yaml")) as f:
    config = yaml.safe_load(f)

# Save Parameter In Experience Folder
os.makedirs(config["output_dir"], exist_ok=True)
with open(os.path.join(config["output_dir"], "experiment_args.yaml"), 'w') as f:
    yaml.dump(src.common.utils.ConfigArgs(config), f, indent=4)


fixed_params = config["fixed_params"]
varying_params = config["varying_param"]

fixed_params['experiment_name'] = config['experiment_name']
fixed_params['output_dir'] = config['output_dir']


nb_seeds = varying_params.pop("nb_seeds")
seed_iterator = range(nb_seeds)


varying_params_lists = []
varying_params_keys = []

for key, value in varying_params.items():
    varying_params_keys.append(key)
    if not isinstance(value, list):
        varying_params_lists.append([value])
    else:
        varying_params_lists.append(value)

param_comb = list(itertools.product(*varying_params_lists))

total_runs = len(param_comb)*nb_seeds


run_counter = 0
for comb in param_comb:
    for seed in seed_iterator:
        run_counter += 1
        current_var_params = dict(zip(varying_params_keys, comb))
        run_name = f"exp_{run_counter:03d}"

        current_params = {
            **fixed_params,
            **current_var_params,
            "seed": seed
        }

        exp_output_dir = os.path.join(current_params["output_dir"], run_name)
        current_params["output_dir"] = exp_output_dir
        os.makedirs(exp_output_dir, exist_ok=True)

        args = src.common.utils.ConfigArgs(current_params)
        print(f"Parameters : {args}")
        with open(os.path.join(exp_output_dir, "run_args.yaml"), 'w') as f:
            yaml.dump(args, f, indent=4)
        src.experiments.exp3_local_inference_and_segmentation.run_experiment(
            args)

        del args
