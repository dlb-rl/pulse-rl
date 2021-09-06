import os
import copy
import luigi
import json
import pickle
from tqdm import tqdm
import pandas as pd
import shap
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import ray
from ray.tune.utils import merge_dicts
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.marwil.bc import BCTrainer
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator
from ray.rllib.offline.is_estimator import ImportanceSamplingEstimator

from src.task_timer import TaskTimer
from src.train_model import utils
from src.train_model.custom_estimators import (
    CustomImportanceSamplingEstimator,
    get_registry_reward,
    get_deal_reward,
    get_ltv_reward,
)

import torch
import torch.nn as nn
import math
import numpy as np
from src.running_stats import RunningMeanStd

N_SHAP_SAMPLES = 5
TRAINER = dict(
    DQN_model=DQNTrainer, SAC_model=SACTrainer, PPO_model=PPOTrainer, BC_model=BCTrainer
)


class Evaluation(luigi.Task):
    train_output = luigi.Parameter(default="")
    model_version = luigi.Parameter()
    dataset_version = luigi.Parameter()

    datasets_name = luigi.ListParameter()
    num_workers = luigi.IntParameter()
    rewards_shift = luigi.ListParameter(default=[])
    evaluation_episodes = luigi.IntParameter(default=0)
    is_gamma = luigi.FloatParameter(default=0.99)
    wis_gamma = luigi.FloatParameter(default=0.99)

    local_eval = luigi.BoolParameter()
    train_id = luigi.Parameter(default="")

    use_direct_estimator = luigi.BoolParameter(default=False)
    predictor_version = luigi.Parameter(default="reward_pred_v0")
    use_custom_estimator = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(Evaluation, self).__init__(*args, **kwargs)

        ray.init(address="auto", dashboard_host="0.0.0.0")

        self.timer = TaskTimer()

        if self.local_eval:
            self.model_path = "models/{}_model/{}/{}/".format(
                self.model_version, self.model_version.split("_")[0], self.train_id
            )
        else:
            with open(self.train_output, "r") as f:
                train_output = json.load(f)
            self.model_path = str(train_output["logdir"] + "/")

        self.save_folder = os.path.join(self.model_path, "evaluation")

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.save_folder,
                "evaluation_scalars.txt",
            )
        )

    def run(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Create agent
        with open(os.path.join(self.model_path, "params.pkl"), "rb") as f:
            config = pickle.load(f)
        evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
        config = merge_dicts(config, evaluation_config)
        config["evaluation_num_workers"] = self.num_workers
        del config["input"]
        agent = TRAINER[self.model_version](env=config["env"], config=config)

        # Get list of checkpoints
        checkpoints = sorted(
            [
                int(f.split("_")[1])
                for f in os.listdir(self.model_path)
                if f.startswith("checkpoint")
            ]
        )

        if len(self.rewards_shift) <= 0:
            self.rewards_shift = []
            for dataset_name in self.datasets_name:
                dataset_path = "data/processed/{}_dataset/{}".format(
                    self.dataset_version, dataset_name
                )
                dataset = []
                for path in os.listdir(dataset_path):
                    if os.path.isfile(os.path.join(dataset_path, path)):
                        dataset.append(os.path.join(dataset_path, path))
                dataset = sorted(dataset)

                rewards = []
                for n_eps in range(len(dataset)):
                    reader = JsonReader(dataset[n_eps])

                    with open(dataset[n_eps], "r") as f:
                        sb = f.readlines()

                    for _ in range(len(sb)):
                        n = reader.next()
                        batch = reader.next()
                        for episode in batch.split_by_episode():
                            for r in episode["rewards"]:
                                rewards.append(r)

                self.rewards_shift.append(
                    (round(min(rewards), 10) * -1) + 1e-6
                    if round(min(rewards), 10) <= 0
                    else 0
                )

        for dataset_name, reward_shift in zip(self.datasets_name, self.rewards_shift):
            # create dirs
            os.makedirs(
                os.path.join(self.save_folder, dataset_name, "charts"), exist_ok=True
            )
            # Create Tensorboard Writer
            writer = SummaryWriter(os.path.join(self.save_folder, dataset_name))

            # Get validation dataset
            dataset_path = "data/processed/{}_dataset/{}/".format(
                self.dataset_version, dataset_name
            )
            validation_dataset = [
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f))
            ]

            if self.use_direct_estimator:
                # Get input size
                r = JsonReader(validation_dataset[0])
                b = r.next()
                input_size = b["obs"].shape[-1]

                # Get checkpoint dir
                chkpt_path = "models/{}_model/release/{}_input".format(
                    self.predictor_version, input_size
                )

                # Load reward predictor
                rew_pred = RewardPredictor(input_size, os.path.abspath(chkpt_path))
                del r, b

            evaluation_scalars = {}
            for checkpoint in checkpoints:
                self.timer.start(
                    "Evaluating model for checkpoint {}".format(checkpoint)
                )

                policy = self.get_policy(agent, checkpoint)

                wis_estimator = WeightedImportanceSamplingEstimator(policy, gamma=0.99)
                is_estimator = ImportanceSamplingEstimator(policy, gamma=0.99)

                if self.use_custom_estimator:
                    custom_estimator = CustomImportanceSamplingEstimator(
                        policy, gamma=0.99
                    )

                if self.use_direct_estimator:
                    direct_estimator = DirectEstimator(policy, rew_pred)

                obs_samples = []  # for shap calculation
                actions = []
                true_actions = []

                estimation = {
                    "wis/V_prev": [],
                    "wis/V_step_WIS": [],
                    "wis/V_gain_est": [],
                    "is/V_prev": [],
                    "is/V_step_IS": [],
                    "is/V_gain_est": [],
                    "register/is/V_prev": [],
                    "register/is/V_step_IS": [],
                    "register/is/V_gain_est": [],
                    "deal/is/V_prev": [],
                    "deal/is/V_step_IS": [],
                    "deal/is/V_gain_est": [],
                    "ltv/is/V_prev": [],
                    "ltv/is/V_step_IS": [],
                    "ltv/is/V_gain_est": [],
                }

                for n_eps in tqdm(range(len(validation_dataset))):
                    reader = JsonReader(validation_dataset[n_eps])
                    batch = reader.next()
                    ep_obs_samples = []

                    for episode in batch.split_by_episode():
                        # potentially sample N_SHAP_SAMPLES observations for later (it can be less)
                        ep_obs_samples.extend(
                            shap.sample(episode["obs"], N_SHAP_SAMPLES)
                        )

                        true_actions.extend(episode["actions"])
                        computed_actions = policy.compute_actions(episode["obs"])[0]
                        actions.extend(computed_actions.tolist())

                        wis_estimation = dict(
                            wis_estimator.estimate(
                                episode, reward_shift=reward_shift
                            )._asdict()
                        )
                        is_estimation = dict(
                            is_estimator.estimate(
                                episode, reward_shift=reward_shift
                            )._asdict()
                        )

                        if self.use_custom_estimator:
                            # Custom business metrics --------------------
                            register_reward = get_registry_reward(episode)
                            custom_register_estimate = custom_estimator.estimate(
                                episode, register_reward
                            )

                            deal_reward = get_deal_reward(episode)
                            custom_deal_estimate = custom_estimator.estimate(
                                episode, deal_reward
                            )

                            ltv_reward = get_ltv_reward(
                                episode, reward_shift=reward_shift
                            )
                            custom_ltv_estimate = custom_estimator.estimate(
                                episode, ltv_reward
                            )

                            # Custom Estimations -----------------------
                            if custom_register_estimate:
                                custom_register_estimate = dict(
                                    custom_register_estimate._asdict()
                                )
                                estimation["register/is/V_prev"].append(
                                    custom_register_estimate["metrics"]["V_prev"]
                                )
                                estimation["register/is/V_step_IS"].append(
                                    custom_register_estimate["metrics"]["V_step_IS"]
                                )
                                estimation["register/is/V_gain_est"].append(
                                    custom_register_estimate["metrics"]["V_gain_est"]
                                )

                            if custom_deal_estimate:
                                custom_deal_estimate = dict(
                                    custom_deal_estimate._asdict()
                                )
                                estimation["deal/is/V_prev"].append(
                                    custom_deal_estimate["metrics"]["V_prev"]
                                )
                                estimation["deal/is/V_step_IS"].append(
                                    custom_deal_estimate["metrics"]["V_step_IS"]
                                )
                                estimation["deal/is/V_gain_est"].append(
                                    custom_deal_estimate["metrics"]["V_gain_est"]
                                )

                            if custom_ltv_estimate:
                                custom_ltv_estimate = dict(
                                    custom_ltv_estimate._asdict()
                                )
                                estimation["ltv/is/V_prev"].append(
                                    custom_ltv_estimate["metrics"]["V_prev"]
                                )
                                estimation["ltv/is/V_step_IS"].append(
                                    custom_ltv_estimate["metrics"]["V_step_IS"]
                                )
                                estimation["ltv/is/V_gain_est"].append(
                                    custom_ltv_estimate["metrics"]["V_gain_est"]
                                )

                        # Direct Estimator
                        if self.use_direct_estimator:
                            DM_score = direct_estimator.estimate(episode["obs"])

                        # WIS Estimation -----------------------
                        estimation["wis/V_prev"].append(
                            wis_estimation["metrics"]["V_prev"]
                        )
                        estimation["wis/V_step_WIS"].append(
                            wis_estimation["metrics"]["V_step_WIS"]
                        )
                        estimation["wis/V_gain_est"].append(
                            wis_estimation["metrics"]["V_gain_est"]
                        )

                        # IS Estimation -----------------------
                        estimation["is/V_prev"].append(
                            is_estimation["metrics"]["V_prev"]
                        )
                        estimation["is/V_step_IS"].append(
                            is_estimation["metrics"]["V_step_IS"]
                        )
                        estimation["is/V_gain_est"].append(
                            is_estimation["metrics"]["V_gain_est"]
                        )

                    # from the transitions sampled above, sample again to reduce samples
                    obs_samples.extend(
                        shap.sample(np.array(ep_obs_samples), N_SHAP_SAMPLES)
                    )

                # from the transitions sampled above, sample again to reduce samples further more
                # these subsequent samplings ensures uniform sampling
                obs_samples = shap.sample(np.array(obs_samples), N_SHAP_SAMPLES)

                est_mean = {}
                for k in estimation.keys():
                    est_mean[k] = np.array(estimation[k]).mean()
                # est_mean = pd.DataFrame.from_dict(estimation).mean(axis=0)
                evaluation_scalars[checkpoint] = est_mean

                # WIS Estimation -----------------------
                writer.add_scalar(
                    "evaluation/wis/V_prev", est_mean["wis/V_prev"], checkpoint
                )
                writer.add_scalar(
                    "evaluation/wis/V_step_WIS",
                    est_mean["wis/V_step_WIS"],
                    checkpoint,
                )
                writer.add_scalar(
                    "evaluation/wis/V_gain_est",
                    est_mean["wis/V_gain_est"],
                    checkpoint,
                )

                # IS Estimation -----------------------
                writer.add_scalar(
                    "evaluation/is/V_prev", est_mean["is/V_prev"], checkpoint
                )
                writer.add_scalar(
                    "evaluation/is/V_step_IS", est_mean["is/V_step_IS"], checkpoint
                )
                writer.add_scalar(
                    "evaluation/is/V_gain_est",
                    est_mean["is/V_gain_est"],
                    checkpoint,
                )

                if self.use_custom_estimator:
                    # Custom Estimations -----------------------
                    writer.add_scalar(
                        "evaluation/register/is/V_prev",
                        est_mean["register/is/V_prev"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/register/is/V_step_IS",
                        est_mean["register/is/V_step_IS"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/register/is/V_gain_est",
                        est_mean["register/is/V_gain_est"],
                        checkpoint,
                    )

                    writer.add_scalar(
                        "evaluation/deal/is/V_prev",
                        est_mean["deal/is/V_prev"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/deal/is/V_step_IS",
                        est_mean["deal/is/V_step_IS"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/deal/is/V_gain_est",
                        est_mean["deal/is/V_gain_est"],
                        checkpoint,
                    )

                    writer.add_scalar(
                        "evaluation/ltv/is/V_prev",
                        est_mean["ltv/is/V_prev"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/ltv/is/V_step_IS",
                        est_mean["ltv/is/V_step_IS"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/ltv/is/V_gain_est",
                        est_mean["ltv/is/V_gain_est"],
                        checkpoint,
                    )

                if self.use_direct_estimator:
                    # DM Estimation ------------------------
                    writer.add_scalar(
                        "evaluation/dm/score", DM_score["score"], checkpoint
                    )
                    writer.add_scalar(
                        "evaluation/dm/pred_reward_mean",
                        DM_score["pred_reward_mean"],
                        checkpoint,
                    )
                    writer.add_scalar(
                        "evaluation/dm/pred_reward_total",
                        DM_score["pred_reward_total"],
                        checkpoint,
                    )

                # Action
                writer.add_scalar(
                    "evaluation/actions_prob",
                    float(actions.count(1)) / len(actions),
                    checkpoint,
                )

                # Accuracy, Precision, Recall, F1
                true_actions = np.array(true_actions)
                pred_actions = np.array(actions)

                accuracy = (pred_actions == true_actions).sum() / len(true_actions)
                true_positives = ((pred_actions == 1) & (true_actions == 1)).sum()
                false_positives = ((pred_actions == 1) & (true_actions == 0)).sum()
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / true_actions.sum()
                f1 = (2 * precision * recall) / (precision + recall)

                writer.add_scalar(
                    "evaluation/accuracy",
                    accuracy,
                    checkpoint,
                )
                writer.add_scalar(
                    "evaluation/precision",
                    precision,
                    checkpoint,
                )
                writer.add_scalar(
                    "evaluation/recall",
                    recall,
                    checkpoint,
                )
                writer.add_scalar(
                    "evaluation/f1",
                    f1,
                    checkpoint,
                )

                # Confusion Matrix
                cm = confusion_matrix(true_actions, pred_actions)
                figure = utils.plot_confusion_matrix(
                    cm, class_names=["Don't activate", "Activate"]
                )
                writer.add_figure("Confusion Matrix", figure, checkpoint)

                # SHAP values
                shap_bar, shap_beeswarm = utils.plot_shap_values(
                    policy,
                    obs_samples,
                    os.path.join(self.save_folder, dataset_name, "charts"),
                )
                writer.add_figure("SHAP Bar", shap_bar, checkpoint)
                writer.add_figure("SHAP Bee Swarm", shap_beeswarm, checkpoint)

                self.timer.end("Evaluating model for checkpoint {}".format(checkpoint))

            self.timer.save(self.save_folder)

            writer.close()

            output_file = os.path.join(
                self.save_folder,
                "evaluation_scalars_{}.json".format(dataset_name),
            )
            with open(output_file, "w+") as f:
                json.dump(evaluation_scalars, f)

            with self.output().open("w") as f:
                f.write(output_file)

    def get_policy(self, agent, checkpoint):
        agent.restore(
            os.path.join(
                self.model_path,
                "checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint),
            )
        )
        policy = agent.get_policy()

        return policy


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, bias=True)
        # nn.init.zeros_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 256, bias=True)
        # nn.init.zeros_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc4 = nn.Linear(128, 1, bias=True)
        # nn.init.zeros_(self.fc4.weight)
        self.tanh = torch.nn.Tanh()
        self.softp = torch.nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        # x = self.softp(x)
        return x


class RewardPredictor:
    def __init__(self, input_size, checkpoint_dir):

        self.model = NN(input_size + 1)

        self.running_stats = RunningMeanStd()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state, scheduler_state, running_stats_state = torch.load(
            checkpoint, map_location=torch.device(device)
        )
        self.model.load_state_dict(model_state)
        self.running_stats.load_dict(running_stats_state)

    def predict(self, x):
        scores = self.model(x)
        scores_raw = (torch.exp(scores) - 1 + 0.003) * math.sqrt(
            (self.running_stats.var)
        )  # just the inverse transofrmation for the predicted rewards
        return scores_raw


class DirectEstimator:
    def __init__(self, policy, predictor):
        self.predictor = predictor
        self.policy = policy

    def estimate(self, obs):
        actions = self.policy.compute_actions(obs)
        action_probs = actions[2]["action_prob"][0]  # get probabilities
        obs = torch.Tensor(
            np.concatenate(
                (obs, np.reshape(actions[0], (actions[0].shape[0], 1))), axis=1
            )
        )  # concatenate actions and observations for input obs are usually [[obs1],[obs2],[obs3]] and
        # actions are usually [1,0,1,0] so the goal is to make actions like this: [[1],[0],[1]]
        scores_raw = self.predictor.predict(obs)
        results = {}
        results["score"] = (scores_raw * action_probs).mean()
        results["pred_reward_mean"] = scores_raw.mean()
        results["pred_reward_total"] = scores_raw.sum()
        return results
