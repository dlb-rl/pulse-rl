# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import time

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from statistics import mean
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.train_model import utils
from ray.rllib.offline.json_reader import JsonReader
from src.train_model.batch_rl.fixed_replay.reward_predictor import RewardPredictor
from src.train_model.batch_rl.fixed_replay.is_estimator import (
    ImportanceSamplingEstimator,
)
from src.train_model.batch_rl.fixed_replay.custom_estimators import (
    CustomImportanceSamplingEstimator,
    get_registry_reward,
    get_deal_reward,
    get_ltv_reward,
)

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""

    def _initialize_checkpointer_and_maybe_resume(
        self, checkpoint_file_prefix, checkpoint_number=None
    ):
        super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
            checkpoint_file_prefix
        )

        # Code for the loading a checkpoint at initialization
        init_checkpoint_dir = self._agent._init_checkpoint_dir
        if (self._start_iteration == 0) and (init_checkpoint_dir is not None):
            if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir, checkpoint_file_prefix
                )
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir
                )
                print("Latest init checkpoint ", latest_init_checkpoint)

                if latest_init_checkpoint >= 0:
                    experiment_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint
                    )
                    if self._agent.unbundle(
                        init_checkpoint_dir, latest_init_checkpoint, experiment_data
                    ):
                        if experiment_data is not None:
                            assert "logs" in experiment_data
                            assert "current_iteration" in experiment_data
                            self._logger.data = experiment_data["logs"]
                            self._start_iteration = (
                                experiment_data["current_iteration"] + 1
                            )
                        print(
                            "Reloaded checkpoint from {} and will start from iteration {}".format(
                                init_checkpoint_dir, self._start_iteration
                            )
                        )
        elif self._start_iteration > 0:
            print(
                "Reloaded checkpoint from {} and will start from iteration {}".format(
                    self._checkpoint_dir, self._start_iteration
                )
            )

        if checkpoint_number:
            init_checkpointer = checkpointer.Checkpointer(
                init_checkpoint_dir, checkpoint_file_prefix
            )
            if checkpoint_number >= 0:
                experiment_data = init_checkpointer.load_checkpoint(checkpoint_number)
                if self._agent.unbundle(
                    init_checkpoint_dir, checkpoint_number, experiment_data
                ):
                    if experiment_data is not None:
                        assert "logs" in experiment_data
                        assert "current_iteration" in experiment_data
                        self._logger.data = experiment_data["logs"]
                        self._start_iteration = experiment_data["current_iteration"] + 1
                    print(
                        "Reloaded checkpoint from {} and will start from iteration {}".format(
                            init_checkpoint_dir, self._start_iteration
                        )
                    )

    def _run_train_phase(self):
        """Run training phase."""
        self._agent.eval_mode = False
        start_time = time.time()
        for _ in range(self._training_steps):
            self._agent._train_step()
        time_delta = time.time() - start_time
        print(
            "Average training steps per second: {}".format(
                str(self._training_steps / time_delta)
            )
        )

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        print("Starting iteration {}".format(iteration))

        if not self._agent._replay_suffix:
            # Reload the replay buffer
            self._agent._replay.memory.reload_buffer(num_buffers=5)

        self._run_train_phase()
        self.offline_evaluation(iteration)

        return statistics.data_lists

    def set_offline_evaluation(self, dataset_path, chkpt_path):
        # Get validation dataset
        self.dataset_path = dataset_path
        input_size = self._environment.observation_shape[0]

        self.predictor = RewardPredictor(input_size, os.path.abspath(chkpt_path))
        self.is_estimator = ImportanceSamplingEstimator()
        self.custom_estimator = CustomImportanceSamplingEstimator()

    def offline_evaluation(self, iteration):
        self._agent.eval_mode = True

        validation_dataset = [
            os.path.join(self.dataset_path, f)
            for f in os.listdir(self.dataset_path)
            if os.path.isfile(os.path.join(self.dataset_path, f))
        ]
        validation_dataset = sorted(validation_dataset)

        rewards = []
        for n_eps in range(len(validation_dataset)):
            reader = JsonReader(validation_dataset[n_eps])

            with open(validation_dataset[n_eps], "r") as f:
                sb = f.readlines()

            for _ in range(len(sb)):
                n = reader.next()
                batch = reader.next()
                for episode in batch.split_by_episode():
                    for r in episode["rewards"]:
                        rewards.append(r)

        rewards_shift = (
            (round(min(rewards), 10) * -1) + 1e-6 if round(min(rewards), 10) <= 0 else 0
        )

        actions = []
        true_actions = []

        estimation = {
            "dm/score": [],
            "dm/pred_reward_mean": [],
            "dm/pred_reward_total": [],
            "is/V_prev": [],
            "is/V_step_IS": [],
            "is/V_gain_est": [],
        }
        custom_register = {
            "register/is/V_prev": [],
            "register/is/V_step_IS": [],
            "register/is/V_gain_est": [],
        }
        custom_deal = {
            "deal/is/V_prev": [],
            "deal/is/V_step_IS": [],
            "deal/is/V_gain_est": [],
        }
        custom_ltv = {
            "ltv/is/V_prev": [],
            "ltv/is/V_step_IS": [],
            "ltv/is/V_gain_est": [],
        }

        for n_eps in range(len(validation_dataset)):
            reader = JsonReader(validation_dataset[n_eps])
            batch = reader.next()

            for episode in batch.split_by_episode():
                true_actions.extend(episode["actions"])

                action, selected_action_prob, all_actions_prob = [], [], []
                for i in range(len(episode["eps_id"])):
                    _action, _action_prob, _ = self._agent.step(episode["obs"][i])
                    action.append(_action)
                    selected_action_prob.append(_action_prob[_action])
                    all_actions_prob.append(_action_prob)

                is_estimation = self.is_estimator.estimate(
                    episode, all_actions_prob, rewards_shift
                )

                # Custom business metrics --------------------
                register_reward = get_registry_reward(episode)
                custom_register_estimate = self.custom_estimator.estimate(
                    episode, all_actions_prob, register_reward
                )

                deal_reward = get_deal_reward(episode)
                custom_deal_estimate = self.custom_estimator.estimate(
                    episode, all_actions_prob, deal_reward
                )

                ltv_reward = get_ltv_reward(episode, reward_shift=rewards_shift)
                custom_ltv_estimate = self.custom_estimator.estimate(
                    episode, all_actions_prob, ltv_reward
                )

                # Custom Estimations -----------------------
                if custom_register_estimate:
                    custom_register["register/is/V_prev"].append(
                        custom_register_estimate["V_prev"]
                    )
                    custom_register["register/is/V_step_IS"].append(
                        custom_register_estimate["V_step_IS"]
                    )
                    custom_register["register/is/V_gain_est"].append(
                        custom_register_estimate["V_gain_est"]
                    )

                if custom_deal_estimate:
                    custom_deal["deal/is/V_prev"].append(custom_deal_estimate["V_prev"])
                    custom_deal["deal/is/V_step_IS"].append(
                        custom_deal_estimate["V_step_IS"]
                    )
                    custom_deal["deal/is/V_gain_est"].append(
                        custom_deal_estimate["V_gain_est"]
                    )

                if custom_ltv_estimate:
                    custom_ltv["ltv/is/V_prev"].append(custom_ltv_estimate["V_prev"])
                    custom_ltv["ltv/is/V_step_IS"].append(
                        custom_ltv_estimate["V_step_IS"]
                    )
                    custom_ltv["ltv/is/V_gain_est"].append(
                        custom_ltv_estimate["V_gain_est"]
                    )

                # Direct Estimator -----------------------
                actions.extend(action)
                action = np.array([action])
                action_prob = np.array([selected_action_prob])

                obs = torch.Tensor(
                    np.concatenate(
                        (episode["obs"], np.reshape(action, (action[0].shape[0], 1))),
                        axis=1,
                    )
                )  # concatenate actions and observations for input obs are usually [[obs1],[obs2],[obs3]] and
                # actions are usually [1,0,1,0] so the goal is to make actions like this: [[1],[0],[1]]
                scores_raw = self.predictor.predict(obs).detach().numpy()
                scores = {}
                scores["score"] = (scores_raw * action_prob).mean()
                scores["pred_reward_mean"] = scores_raw.mean()
                scores["pred_reward_total"] = scores_raw.sum()

                # DM Estimation ------------------------
                estimation["dm/score"].append(scores["score"])
                estimation["dm/pred_reward_mean"].append(scores["pred_reward_mean"])
                estimation["dm/pred_reward_total"].append(scores["pred_reward_total"])

                # IS Estimation -----------------------
                estimation["is/V_prev"].append(is_estimation["V_prev"])
                estimation["is/V_step_IS"].append(is_estimation["V_step_IS"])
                estimation["is/V_gain_est"].append(is_estimation["V_gain_est"])

        est_mean = pd.DataFrame.from_dict(estimation).mean(axis=0)
        custom_register_mean = pd.DataFrame.from_dict(custom_register).mean(axis=0)
        custom_deal_mean = pd.DataFrame.from_dict(custom_deal).mean(axis=0)
        custom_ltv_mean = pd.DataFrame.from_dict(custom_ltv).mean(axis=0)

        # Accuracy, Precision, Recall, F1
        true_actions = np.array(true_actions)
        pred_actions = np.array(actions)

        accuracy = (pred_actions == true_actions).sum() / len(true_actions)
        true_positives = ((pred_actions == 1) & (true_actions == 1)).sum()
        false_positives = ((pred_actions == 1) & (true_actions == 0)).sum()
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / true_actions.sum()
        f1 = (2 * precision * recall) / (precision + recall)

        # Confusion Matrix
        cm = confusion_matrix(true_actions, pred_actions)

        figure = utils.plot_confusion_matrix(
            cm, class_names=["Don't activate", "Activate"]
        )
        buf = io.BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)

        height, width, channel = np.asarray(im).shape
        image = Image.fromarray(np.asarray(im))

        output = io.BytesIO()
        image.save(output, format="PNG")
        image_string = output.getvalue()
        output.close()
        image_summary = tf.Summary.Image(
            height=height,
            width=width,
            colorspace=channel,
            encoded_image_string=image_string,
        )

        summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag="Evaluation/DM/score", simple_value=est_mean["dm/score"]
                ),
                tf.Summary.Value(
                    tag="Evaluation/DM/pred_reward_mean",
                    simple_value=est_mean["dm/pred_reward_mean"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/DM/pred_reward_total",
                    simple_value=est_mean["dm/pred_reward_total"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/is/V_prev", simple_value=est_mean["is/V_prev"]
                ),
                tf.Summary.Value(
                    tag="Evaluation/is/V_step_IS",
                    simple_value=est_mean["is/V_step_IS"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/is/V_gain_est",
                    simple_value=est_mean["is/V_gain_est"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/actions_prob",
                    simple_value=float(actions.count(1)) / len(actions),
                ),
                tf.Summary.Value(
                    tag="Evaluation/register/is/V_prev",
                    simple_value=custom_register_mean["register/is/V_prev"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/register/is/V_step_IS",
                    simple_value=custom_register_mean["register/is/V_step_IS"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/register/is/V_gain_est",
                    simple_value=custom_register_mean["register/is/V_gain_est"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/deal/is/V_prev",
                    simple_value=custom_deal_mean["deal/is/V_prev"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/deal/is/V_step_IS",
                    simple_value=custom_deal_mean["deal/is/V_step_IS"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/deal/is/V_gain_est",
                    simple_value=custom_deal_mean["deal/is/V_gain_est"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/ltv/is/V_prev",
                    simple_value=custom_ltv_mean["ltv/is/V_prev"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/ltv/is/V_step_IS",
                    simple_value=custom_ltv_mean["ltv/is/V_step_IS"],
                ),
                tf.Summary.Value(
                    tag="Evaluation/ltv/is/V_gain_est",
                    simple_value=custom_ltv_mean["ltv/is/V_gain_est"],
                ),
                tf.Summary.Value(tag="Evaluation/accuracy", simple_value=accuracy),
                tf.Summary.Value(tag="Evaluation/precision", simple_value=precision),
                tf.Summary.Value(tag="Evaluation/recall", simple_value=recall),
                tf.Summary.Value(tag="Evaluation/f1", simple_value=f1),
                tf.Summary.Value(
                    tag="Evaluation/Confusion Matrix", image=image_summary
                ),
            ],
        )
        self._summary_writer.add_summary(summary, iteration)
