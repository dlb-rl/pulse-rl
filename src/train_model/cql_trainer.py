import os
import sys
import ast
import json
import luigi
import pprint
import datetime
import numpy as np
from tqdm import tqdm
import functools

import gin.tf

from dopamine.discrete_domains import run_experiment as base_run_experiment

from src.train_model.batch_rl.fixed_replay import run_experiment
from src.train_model.batch_rl.fixed_replay.agents.quantile_agent import (
    FixedReplayQuantileAgent,
)
from src.train_model.environments import ACPulse
from src.task_timer import TaskTimer
from src.data.cql_replay_dataset import CQLReplayDataset

ENVS = {"ac-pulse": ACPulse}


class CQLTrainer(luigi.Task):

    dtime = luigi.Parameter()
    model_version = luigi.Parameter()

    algorithm = luigi.Parameter()
    env = luigi.Parameter()
    env_config = luigi.DictParameter()

    dataset_version = luigi.Parameter()
    dataset_name = luigi.Parameter(default="dataset")
    base_dataset_version = luigi.Parameter()
    evaluation_dataset_name = luigi.Parameter()
    dataset_initial_day = luigi.Parameter()
    dataset_final_day = luigi.Parameter()

    init_checkpoint_dir = luigi.Parameter(default="")
    gin_files = luigi.ListParameter()
    gin_bindings = luigi.ListParameter(default=[])
    load_experiment_checkpoint = luigi.BoolParameter(default=False)
    train_id = luigi.Parameter(default="")
    config = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(CQLTrainer, self).__init__(*args, **kwargs)

        save_folder = os.path.abspath("models/{}_model/CQL".format(self.model_version))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder

        self.timer = TaskTimer()

    def requires(self):
        return CQLReplayDataset(
            self.dataset_version,
            self.dataset_name,
            self.dataset_initial_day,
            self.dataset_final_day,
            self.base_dataset_version,
            self.env_config["obs_shape"],
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.save_folder, "result_{}.json".format(self.dtime))
        )

    def run(self):

        num_iterations = None
        if self.load_experiment_checkpoint:
            self.init_checkpoint_dir = os.path.join(self.save_folder, self.train_id)
            num_iterations = self.get_training_iterations()

        else:
            self.init_checkpoint_dir = os.path.join(
                self.save_folder,
                "CQL_ACPulse_{}".format(self.dtime),
            )
        print("----- Init Checkpoint Dir: ", self.init_checkpoint_dir)

        if num_iterations:
            gin_bindings = list(self.gin_bindings)
            for idx in range(len(gin_bindings)):
                if gin_bindings[idx].startswith("FixedReplayRunner.num_iterations"):
                    gin_bindings[idx] = "FixedReplayRunner.num_iterations = {}".format(
                        num_iterations
                    )
            self.gin_bindings = gin_bindings

        self.dataset_path = self.get_dataset()

        result = self.train()

        with self.output().open("w") as f:
            json.dump(result, f)

    def get_dataset(self):
        dataset_path = "data/processed/{}_dataset/{}/replay_logs/".format(
            self.dataset_version, self.dataset_name
        )
        print("----- ", dataset_path)

        return dataset_path

    def get_training_iterations(self):
        with open(
            os.path.join(self.init_checkpoint_dir, "checkpoints/checkpoint")
        ) as f:
            num_iterations = (
                int(f.readline().split(" ")[-1].split("/")[-1].split("-")[-1][:-2]) + 2
            )

        return num_iterations

    def train(self):
        base_run_experiment.load_gin_configs(self.gin_files, self.gin_bindings)
        create_agent_fn = functools.partial(
            self.create_agent,
            replay_data_dir=self.dataset_path,
            init_checkpoint_dir=self.init_checkpoint_dir,
        )
        create_environment_fn = functools.partial(
            self.create_environment, self.env_config
        )

        runner = run_experiment.FixedReplayRunner(
            self.init_checkpoint_dir,
            create_agent_fn,
            create_environment_fn=create_environment_fn,
        )

        print("----- Initialize checkpointer and maybe resume")
        runner._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix="ckpt")

        eval_dataset_path = "data/processed/{}_dataset/{}/".format(
            self.base_dataset_version, self.evaluation_dataset_name
        )
        dm_chkpt_path = "models/reward_pred_v0_model/release/{}_input".format(
            self.env_config["obs_shape"]
        )

        runner.set_offline_evaluation(
            os.path.abspath(eval_dataset_path), os.path.abspath(dm_chkpt_path)
        )

        runner.run_experiment()

        with open(
            os.path.join(self.init_checkpoint_dir, "checkpoints/checkpoint")
        ) as f:
            checkpoint = f.readline().split(" ")[-1]

        result = {
            "datetime": self.dtime,
            "logdir": os.path.relpath(self.init_checkpoint_dir),
            "checkpoint": checkpoint,
        }
        return result

    def create_agent(
        self,
        sess,
        environment,
        replay_data_dir,
        init_checkpoint_dir,
        summary_writer=None,
    ):
        return FixedReplayQuantileAgent(
            sess,
            num_actions=environment.action_space.n,
            observation_shape=environment.observation_shape,
            observation_dtype=environment.observation_dtype,
            replay_data_dir=replay_data_dir,
            init_checkpoint_dir=init_checkpoint_dir,
            summary_writer=summary_writer,
            replay_scheme="uniform",
        )

    def create_environment(self, env_config):
        return ACPulse(env_config)
