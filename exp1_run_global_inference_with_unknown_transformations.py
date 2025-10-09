import os
import yaml
import sys
import src.experiments.exp1_global_inference_with_unknown_transformations
import src.common.utils


script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

with open(os.path.join(".",
                       "config",
                       "exp1_global_inference_with_unknown_transformations.yaml")) as f:
    config = yaml.safe_load(f)

args = src.common.utils.ConfigArgs(config)

src.experiments.exp1_global_inference_with_unknown_transformations.run_experiment(args)
