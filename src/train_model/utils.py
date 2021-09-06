import os
import itertools
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shap
import copy


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names, rotation=90)

    # Normalize the confusion matrix.
    ncm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, ncm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout(pad=1.8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_shap_values(policy, sample, save_folder=None):
    # TODO perhaps move this to the dataset definitions
    feature_labels = [
        "user_obs_age_group",
        "user_obs_add_0_RO",
        "user_obs_add_1_TO",
        "user_obs_add_2_RR",
        "user_obs_add_3_AC",
        "user_obs_add_4_AP",
        "user_obs_add_5_AL",
        "user_obs_add_6_AM",
        "user_obs_add_7_BA",
        "user_obs_add_8_CE",
        "user_obs_add_9_DF",
        "user_obs_add_10_ES",
        "user_obs_add_11_GO",
        "user_obs_add_12_MA",
        "user_obs_add_13_MG",
        "user_obs_add_14_MS",
        "user_obs_add_15_MT",
        "user_obs_add_16_PA",
        "user_obs_add_17_PB",
        "user_obs_add_18_PE",
        "user_obs_add_19_PI",
        "user_obs_add_20_PR",
        "user_obs_add_21_RJ",
        "user_obs_add_22_RN",
        "user_obs_add_23_RS",
        "user_obs_add_24_SC",
        "user_obs_add_25_SE",
        "user_obs_add_26_SP",
        "user_obs_add_27_NC",
        "user_obs_add_28_MI",
        "user_obs_add_29_ZZ",
        "user_obs_add_30_BN",
        "user_obs_add_31_NF",
        "user_obs_gen_0_FEMALE",
        "user_obs_gen_1_MALE",
        "user_obs_gen_2_OTHER",
        "obs_objective_agreement",
        "obs_objective_signup",
        "obs_score",
        "obs_user_activated",
        "obs_user_registered",
        "obs_debt_located",
        "obs_debt_visualized",
        "obs_deals",
        "obs_last_activation",
        "obs_last_activation_week",
        "obs_last_activation_month",
        "obs_last_activation_year",
        "obs_last_location",
        "obs_last_location_week",
        "obs_last_location_month",
        "obs_last_location_year",
        "obs_last_visualization",
        "obs_last_visualization_week",
        "obs_last_visualization_month",
        "obs_last_visualization_year",
        "obs_last_deal",
        "obs_last_deal_week",
        "obs_last_deal_month",
        "obs_last_deal_year",
        "obs_total_original_value",
        "obs_total_current_value",
        "obs_max_days_of_delay",
    ]

    # manipulate the model input/output
    def model(obs, *args):
        actions = policy.compute_actions(obs)
        return (
            (1 - actions[0]) * (1 - actions[2]["action_prob"])
            + actions[0] * actions[2]["action_prob"]
        )  # this outputs low probs (1 - no activation prob) for actions 0 and high probs for actions 1 (activation prob)

    # create shap explainers
    kernel_explainer = shap.KernelExplainer(model, sample)
    shap_values = kernel_explainer.shap_values(sample)

    explainer = shap.Explainer(model, sample, feature_names=feature_labels)
    shap_exp = explainer(sample)

    # create plots
    plt.figure(0)
    shap.plots.bar(shap_exp, show=False)
    plt.tight_layout(pad=1.2)
    bar = plt.figure(0)

    plt.figure(1)
    shap.plots.beeswarm(shap_exp, show=False)
    plt.tight_layout(pad=1.2)
    beeswarm = plt.figure(1)

    plt.figure(2)
    # force plots return a figure
    individual_force = shap.force_plot(
        kernel_explainer.expected_value,
        shap_values[0],
        feature_names=feature_labels,
        show=False,
    )

    plt.figure(3)
    # force plots return a figure
    global_force = shap.force_plot(
        kernel_explainer.expected_value,
        shap_values,
        feature_names=feature_labels,
        show=False,
    )

    if save_folder is not None:
        bar.savefig(os.path.join(save_folder, "shap_bar.png"))
        beeswarm.savefig(os.path.join(save_folder, "shap_beeswarm.png"))
        shap.save_html(
            os.path.join(save_folder, "shap_individual_force.html"), individual_force
        )
        shap.save_html(
            os.path.join(save_folder, "shap_global_force.html"), global_force
        )

    return bar, beeswarm

