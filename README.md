# Sensorimotor Proto-Objects: Bootstrapping Perception from Compensable Actions

## 📝 Description

This folder contains the source code for the paper "Sensorimotor Proto-Objects: Bootstrapping Perception from Compensable Actions". It provides the necessary scripts to reproduce the experiments and generate the figures presented in the publication.

---

## ⚙️ Installation

Follow these steps to set up the project environment. Developed and tested with Python `3.13.3`. The code is expected to be compatible with Python `3.9` and newer

**1. Copy the repository in your file system.**
<!-- ```bash
git clone TODO
cd TODO
``` -->

**2. Create and activate a virtual environment:**
This ensures that all dependencies are isolated from your system's Python installation.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (on macOS/Linux)
source .venv/bin/activate

# Activate it (on Windows)
.venv\Scripts\activate
```

You should now see (.venv) at the beginning of your terminal prompt.

**3. Install the required packages:**

```bash
pip install -r requirements.txt
```

## 🚀 Usage

The experiments can be run using the scripts in the root directory. The results
are saved to the ```data/results/``` folder by default.

**Running Experiments**

To run each of the main experiments presented in the paper, execute the corresponding script from the project's root directory:

**Experiment 1: Global Inference with Unknown Transformations**
```bash
python ./exp1_run_global_inference_with_unksown_transformations.py
```
**Experiment 2: Global Inference (Environment vs. Object)**
```bash
python ./exp2_run_global_inference_environment_vs_object.py
```
**Experiment 3: Local Inference and Segmentation**
```bash
python ./exp3_run_local_inference_and_segmentation
```
**Note**: each scripts refers to a ```.yaml``` configuration file in the ```config/``` folder. You can modify this file to change the experiment's configuration.

**Vizualizing the Results:**
The results and figures from the paper can be reproduced using the Jupyter 
noteboods in the ```notebooks/``` directory. The figure produced are stored 
in the ```data/figures/``` directory.

By default, each notebook displays the pre-generated results from the paper. If you want to visualize your own results uncomment the ```# YOUR OWN DATA``` section in the run python file in the root directory and specify the experiment name the you selected in the corresponding ```config/``` directory.

## 📄 Citation

If you use this code in your research, please consider citing our paper:

```bibtex
@article{,
  author  = {},
  title   = {},
  journal = {},
  year    = {}
}
```

