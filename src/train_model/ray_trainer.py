import os
import sys
import ast
import json
import luigi
import pprint
import datetime
import numpy as np
from tqdm import tqdm

import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.dqn as dqn
from sklearn.model_selection import train_test_split

from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.agents.marwil.bc import BCTrainer


from src.train_model.algorithms import CustomBCTrainer
from src.train_model.environments import ACPulse

from src.task_timer import TaskTimer

ENVS = {"ac-pulse": ACPulse}
ALGORITHMS = {"DQN": "DQN", "PPO": "PPO", "BC": CustomBCTrainer, "CQL": "CQL"}


class RayTrainer(luigi.Task):
    """
    Use processed dataset to train an Agent using RLlib Offline Datasets
    """

    dtime = luigi.Parameter()
    model_version = luigi.Parameter()
    dataset_version = luigi.Parameter()
    dataset_name = luigi.Parameter(default="dataset")

    algorithm = luigi.Parameter()

    env = luigi.Parameter()
    training_iteration = luigi.IntParameter()
    time_total = luigi.IntParameter()
    checkpoint_frequency = luigi.IntParameter()
    checkpoint_at_end = luigi.BoolParameter()
    load_experiment_checkpoint = luigi.BoolParameter(default=False)
    eval_during_train = luigi.BoolParameter(default=False)
    eval_dataset_name = luigi.Parameter(default="test_dataset")

    train_id = luigi.Parameter(default="")

    analysis_metric = luigi.Parameter()
    analysis_mode = luigi.Parameter()

    config = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(RayTrainer, self).__init__(*args, **kwargs)

        save_folder = os.path.abspath("models/{}_model".format(self.model_version))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder

        self.timer = TaskTimer()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.save_folder, "result_{}.json".format(self.dtime))
        )

    def run(self):
        config = ast.literal_eval(self.config)
        config["env"] = ENVS[self.env]
        config["input"] = self.get_dataset(self.dataset_name)

        if self.eval_during_train:
            config["evaluation_config"]["input"] = self.get_dataset(
                self.eval_dataset_name
            )

        pprint.pprint(config)

        result = self.train(config)

        with self.output().open("w") as f:
            json.dump(result, f)

    def get_dataset(self, dataset_name):
        dataset_path = "data/processed/{}_dataset/{}/".format(
            self.dataset_version, dataset_name
        )

        dataset = []
        for path in os.listdir(dataset_path):
            for f in os.listdir(os.path.join(dataset_path, path)):
                if os.path.isfile(os.path.join(dataset_path, path, f)):
                    dataset.append(os.path.abspath(os.path.join(dataset_path, path, f)))
        dataset = sorted(dataset)
        return dataset

    def get_experiment_checkpoint(self):
        experiment_path = "models/{}_model/{}/{}/".format(
            self.model_version, self.algorithm, self.train_id
        )

        # Get list of checkpoints
        checkpoints = sorted(
            [
                int(f.split("_")[1])
                for f in os.listdir(experiment_path)
                if f.startswith("checkpoint")
            ]
        )
        experiment_checkpoint = "{}checkpoint_{}/checkpoint-{}".format(
            experiment_path, checkpoints[-1], checkpoints[-1]
        )

        return experiment_checkpoint

    def train(self, config):
        checkpoint = None
        if self.load_experiment_checkpoint:
            checkpoint = self.get_experiment_checkpoint()

        print("----- ", checkpoint)
        analysis = tune.run(
            run_or_experiment=ALGORITHMS[self.algorithm],
            name=self.algorithm,
            config=config,
            stop={
                "training_iteration": self.training_iteration,
                "time_total_s": self.time_total,
            },
            checkpoint_freq=self.checkpoint_frequency,
            checkpoint_at_end=self.checkpoint_at_end,
            local_dir=self.save_folder,
            restore=checkpoint,
        )

        logdir = analysis.get_best_logdir(
            metric=self.analysis_metric, mode=self.analysis_mode
        )
        result = {
            "datetime": self.dtime,
            "logdir": os.path.relpath(logdir),
            "checkpoint": os.path.relpath(
                analysis.get_best_checkpoint(
                    logdir, metric=self.analysis_metric, mode=self.analysis_mode
                )
            ),
        }
        return result
