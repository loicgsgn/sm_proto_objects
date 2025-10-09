import os
import yaml
import sys
import src.experiments.exp2_global_inference_environment_vs_object
import src.common.utils

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

with open(os.path.join(".",
                       "config",
                       "exp2_global_inference_environment_vs_object.yaml")) as f:
    config = yaml.safe_load(f)

results_dir = config["output_dir"]

os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "experiment_args.yaml"), 'w') as f:
    yaml.dump(src.common.utils.ConfigArgs(config), f, indent=4)

for seed in range(config["nb_seeds"]+1):
    config["seed"] = seed
    print(f"Running Experiment with seed = {config["seed"]}")
    run_folder_name = f"exp_{seed:03d}"
    config["output_dir"] = os.path.join(results_dir, run_folder_name)
    args = src.common.utils.ConfigArgs(config)
    src.experiments.exp2_global_inference_environment_vs_object.run_experiment(args)
    del args
