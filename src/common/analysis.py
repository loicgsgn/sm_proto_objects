import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ast

TEXTWIDTH_IN_INCH = 7.1413
LINEWIDTH_IN_INCH = 3.48761


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    'lines.markersize': 2.0,
})


def proper_args_print(args):
    for key, value in args.items():
        print(f"{key} : {value}")


def load_exp_args(results_path):
    with open(os.path.join(results_path, "experiment_args.yaml")) as f:
        args = yaml.safe_load(f)
    return args

########################
#   EXPERIENCE 1
########################


def load_exp1_data(results_path):
    df = pd.read_csv(os.path.join(results_path, "results.csv"))
    return df


def generate_figure_exp1(df, args, figure_output_path):
    df["counterfactual_prediction_error_per_action"] =\
          df["counterfactual_prediction_error_per_action"].apply(ast.literal_eval)  # Safe string to list

    environment_transformation = df["environment_transformation"].to_numpy()
    # inferred_actions = df["inferred_actions"].to_numpy()
    counterfactual_prediction_error_per_action =\
        np.array(df["counterfactual_prediction_error_per_action"].tolist())
    nb_transformations = len(args["transformations"])
    nb_actions = len(args["actions"])

    # nb_transformations = len(args.transformations)
    # nb_actions = len(args.actions)

    ascii_direction_actions = np.array(["id", "→", "←", "↓", "↑", "↘", "↗", "↙", "↖"])
    latex_direction_actions_map = {
        "id": "id",
        "→": r"$\rightarrow$",
        "←": r"$\leftarrow$",
        "↓": r"$\downarrow$",
        "↑": r"$\uparrow$",
        "↘": r"$\searrow$",
        "↗": r"$\nearrow$",
        "↙": r"$\swarrow$",
        "↖": r"$\nwarrow$",
    }
    latex_direction_actions = np.array([latex_direction_actions_map[char] for char in ascii_direction_actions])
    unique_transformations = np.unique(environment_transformation)

    mean_prediction_error_per_trans = {}
    for trsf in unique_transformations:
        group = counterfactual_prediction_error_per_action[environment_transformation == trsf]
        mean_prediction_error_per_trans[trsf] = np.mean(group, axis=0)

    mean_prediction_errors_matrix = np.zeros((nb_transformations, nb_actions))
    for i in range(len(mean_prediction_error_per_trans)):
        mean_prediction_errors_matrix[i, :] = mean_prediction_error_per_trans[i]

    # DISPLAY FIGURE
    fig = plt.figure(figsize=(LINEWIDTH_IN_INCH, LINEWIDTH_IN_INCH/2))
    ax = fig.add_subplot(111)

    colors = np.array(['#ff595e', '#ff924c', '#8ac926', "#1982c4", "#6a4c93"])[::-1]
    my_cmap = LinearSegmentedColormap.from_list('my_custom_colormap', colors)

    im = ax.imshow(mean_prediction_errors_matrix.T,
                   cmap=my_cmap, vmin=0, vmax=1.0)

    fig.colorbar(im, fraction=0.026, pad=0.04)

    # First axis : grid
    ax.grid(which="both", color="#292929", linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, nb_transformations, 1), [])
    ax.set_yticks(np.arange(-0.5, nb_actions, 1), [])
    ax.tick_params(which="both", length=0)

    # Second axis : labels
    sec_x_ax = ax.secondary_xaxis(location=1)
    sec_x_ax.set_xlabel("Environment transformation")
    sec_x_ax.set_xticks(np.arange(0, nb_transformations), latex_direction_actions[0:nb_transformations])
    sec_x_ax.tick_params(which="both", length=0)

    sec_y_ax = ax.secondary_yaxis(location=0)
    sec_y_ax.set_ylabel("Hypothesized transformation")
    sec_y_ax.set_yticks(np.arange(0, nb_actions), latex_direction_actions[0:nb_actions])
    sec_y_ax.tick_params(which="both", length=0)

    # Matrix values
    for i in range(nb_transformations):
        for j in range(nb_actions):
            ax.text(i, j, f"{mean_prediction_errors_matrix[i, j]:.2f}",
                    ha="center", va="center", color="w", fontsize=5)

    plt.savefig(os.path.join(figure_output_path, "exp1_global_inference.pdf"),
                format='pdf',
                dpi=300,
                transparent=True)

    plt.show()


########################
#   EXPERIENCE 2
########################


def load_exp2_data(results_path):
    experiments = []
    for exp_name in sorted(os.listdir(results_path)):
        if exp_name != "experiment_args.yaml":
            folder_experience_path = os.path.join(results_path, exp_name)
            if not os.path.exists(folder_experience_path):
                continue
            try:
                df = pd.read_csv(os.path.join(folder_experience_path, "results.csv"))
                df['exp_name'] = exp_name
                experiments.append(df)
            except Exception as e:
                print(f"Skipping {exp_name}: {e}")
    all_data = pd.concat(experiments, ignore_index=True)
    return all_data


def plot_accuracy_exp2(ax, data_to_plot,
                       metric_column, colors,
                       linestyles, show_indiv_accuracies):
    for i_snr, snr_val in enumerate(data_to_plot["snr"].unique()):
        sub_data_snr = data_to_plot[data_to_plot["snr"] == snr_val]
        mean_accuracy_by_size = sub_data_snr.groupby("object_side_size")[metric_column].mean()
        std_accuracy_by_size = sub_data_snr.groupby("object_side_size")[metric_column].std()

        index = mean_accuracy_by_size.index

        ax.plot(index,
                mean_accuracy_by_size,
                marker='s',
                markersize=2.5,
                color=colors[i_snr],
                linestyle=linestyles[i_snr],
                label=f"{snr_val} dB"
                )

        ax.fill_between(index,
                        mean_accuracy_by_size - std_accuracy_by_size,
                        mean_accuracy_by_size + std_accuracy_by_size,
                        alpha=0.2,
                        color=colors[i_snr]
                        )
        if show_indiv_accuracies:
            for obj_size in index:
                accuracies_for_obj_size = sub_data_snr[sub_data_snr["object_side_size"] == obj_size][metric_column]
                jitter_amount = 0.2
                x_jittered = [obj_size + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(accuracies_for_obj_size))]
                ax.scatter(x_jittered,
                           accuracies_for_obj_size,
                           color=colors[i_snr],
                           marker='o',
                           alpha=0.4,
                           s=30,
                           zorder=5)

    ax.set_xlabel("Object side size (pixels)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.)
    unique_obj_sizes = data_to_plot["object_side_size"].unique()
    ax.set_xlim(min(unique_obj_sizes), max(unique_obj_sizes))
    ax.set_xticks(np.arange(min(unique_obj_sizes), max(unique_obj_sizes)+1, 2))
    ax.legend()
    ax.grid(True, which="major", linestyle='-', linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle='-', linewidth=0.25, alpha=0.15)


def generate_figure_exp2(df, figures_folder, show_indiv_accuracies):
    df['object_transformation_match'] = df.transformation_inferred.eq(df.object_transformation)
    df['environment_transformation_match'] = df.transformation_inferred.eq(df.environment_transformation)
    accuracy_per_seed = \
        df.groupby(['snr', 'object_side_size', 'exp_name']).agg(environment_accuracy_per_seed=('environment_transformation_match', 'mean'),  # 'mean' gives True proportion
                                                                object_accuracy_per_seed=('object_transformation_match', 'mean')
                                                                ).reset_index()

    colors = np.array(['#ff595e', '#ff924c', '#ffca3a',
                       '#8ac926', "#1982c4", "#6a4c93"])
    linestyles = [(0, ()), (0, (1, 5)), (0, (1, 1)), (5, (10, 3)),
                  (0, (3, 5, 1, 5)), (0, (5, 1))]

    fig, axes = plt.subplots(2, 1,
                             figsize=(LINEWIDTH_IN_INCH, LINEWIDTH_IN_INCH))

    plot_accuracy_exp2(axes[0], accuracy_per_seed,
                       "environment_accuracy_per_seed",
                       colors, linestyles, show_indiv_accuracies)

    plot_accuracy_exp2(axes[1], accuracy_per_seed,
                       "object_accuracy_per_seed",
                       colors, linestyles, show_indiv_accuracies)

    plt.savefig(os.path.join(figures_folder,
                             "exp2_global_inference_object.pdf"),
                format='pdf',
                dpi=300,
                bbox_inches='tight',
                transparent=True)
    plt.tight_layout()
    plt.show()


########################
#   EXPERIENCE 3
########################

def load_exp3_data(results_path):
    return load_exp2_data(results_path)


def print_accuracy_tab_latex(df):
    print(r"\\hline")
    for object_side_size in df["object_side_size"].unique():
        for i, snr in enumerate(df["snr"].unique()):
            # print(snr, end="")
            sub = df[
                    (df["object_side_size"] == object_side_size) &
                    (df["snr"] == snr)
                    ]
            mean = sub["accuracy"].mean()
            if i == 0:
                print(object_side_size, end="&")
            print(f"{mean:.3f}", end="")
            if i != len(df["snr"].unique()) - 1:
                print("&", end="")
        print(r"\\\ \n", end="")
        print(r"\\hline")


def print_accuracy_tab(df):
    for i, snr in enumerate(df["snr"].unique()):
        if i == 0:
            print("      |", end="")
        print(f" {snr} db |", end="")
    print("\n", end="")
    for object_side_size in df["object_side_size"].unique():

        for i, snr in enumerate(df["snr"].unique()):
            sub = df[
                    (df["object_side_size"] == object_side_size) &
                    (df["snr"] == snr)
                    ]
            mean = sub["accuracy"].mean()
            if i == 0:
                print(f"{object_side_size} x {object_side_size}", end=" | ")
            print(f"{mean:.3f}", end=" | ")
        print("\n", end="")


def generate_figure_exp3_1(df, figure_folder, snr_selected):
    plt.figure(figsize=(LINEWIDTH_IN_INCH, 2))
    list_param = ["recall",  "ext_precision"]
    list_param_name = [r"WR", r"WEP"]

    linestyle_list = [
        {'linestyle': '-', 'marker': 'o', 'color': '#1982c4ff'},
        {'linestyle': '--', 'marker': '^', 'color': '#ff924cff'},
        {'linestyle': '-.', 'marker': 's', 'color': '#8ac926ff'},
        {'linestyle': ':', 'marker': 'D', 'color': '#ff595eff'},
        {'linestyle': '-', 'marker': 'x', 'color': 'purple'},
        {'linestyle': '--', 'marker': '+', 'color': 'brown'},
        {'linestyle': '-.', 'marker': '*', 'color': 'cyan'},
        {'linestyle': ':', 'marker': 'p', 'color': 'magenta'},
        {'linestyle': '-', 'marker': 'h', 'color': 'orange'},
        {'linestyle': '--', 'marker': 'v', 'color': 'gray'},
        {'linestyle': '-.', 'marker': '<', 'color': 'lime'},
        {'linestyle': ':', 'marker': '>', 'color': 'teal'}
    ]

    for i, param in enumerate(list_param):
        plt.subplot(1, 2, i + 1)

        for i_oss, object_side_size in enumerate([1, 2, 4, 6]):

            sub = df[(df["snr"] == snr_selected) & (df["object_side_size"] == object_side_size)
                     ]
            grouped = sub.groupby("trial")[param]
            mean = grouped.mean()
            index = mean.index
            std = grouped.std()

            plt.plot(index,
                     mean,
                     linestyle=linestyle_list[i_oss]['linestyle'],
                     c=linestyle_list[i_oss]['color'],
                     markersize=2.5,
                     label=f"{object_side_size} pxl",
                     markevery=10)
            plt.fill_between(index, mean - std,
                             mean + std,
                             color=linestyle_list[i_oss]['color'],
                             alpha=0.2)

        if i == 0:
            plt.legend()
        plt.grid()
        plt.ylim(0, 1.1)
        plt.xlim(0, 200)
        plt.ylabel(list_param_name[i])
        plt.xlabel("Steps")

    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "exp3_1.pdf"), dpi=300)
    plt.show()


def generate_figure_exp3_2(df, figure_folder):

    plt.figure(figsize=(TEXTWIDTH_IN_INCH, TEXTWIDTH_IN_INCH/4))

    colors_list = ["#8ac926ff", "#1982c4ff", "#6a4c93ff"]

    linestyle_list = [
        {'linestyle': '-'},
        {'linestyle': '--'},
        {'linestyle': '-.'}
    ]

    for i, object_side_size in enumerate([1, 2, 4, 6]):

        plt.subplot(1, 4, i + 1)
        for i_snr, snr in enumerate(df["snr"].unique()):
            sub = df[(df["object_side_size"] == object_side_size) &
                     (df["snr"] == snr)
                     ]
            grouped = sub.groupby("trial")["mixed_f1"]
            mean = grouped.mean()
            index = mean.index
            std = grouped.std()

            plt.plot(index,
                     mean,
                     linestyle=linestyle_list[i_snr]['linestyle'],
                     label=f"{snr} dB",
                     c=colors_list[i_snr])

            plt.fill_between(index,
                             mean - std,
                             mean + std,
                             color=colors_list[i_snr],
                             alpha=0.2)
        plt.grid()
        plt.ylim(0, 1.1)
        plt.xlim(0, 199)

        if object_side_size == 1:
            plt.title(f"{object_side_size}" + r"$\times$" + f"{object_side_size}")
            plt.legend()
        else:
            plt.title(f"{object_side_size}" + r"$\times$" + f"{object_side_size}")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "exp3_2.pdf"), dpi=300)
    plt.show()
