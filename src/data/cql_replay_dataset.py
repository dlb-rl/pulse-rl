import os
import gym
import json
import luigi

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime as dt

from src.task_timer import TaskTimer

from ray.rllib.offline.json_reader import JsonReader
from src.train_model.batch_rl.baselines.replay_memory import WrappedLoggedReplayBuffer


class CQLReplayDataset(luigi.Task):
    """
    Given a Ray-compatible dataset, iterate over transitions, having a CQL-compatible dataset as output.
    """

    ## Configuration file parameters
    dataset_version = luigi.Parameter()
    dataset_name = luigi.Parameter()
    dataset_initial_day = luigi.Parameter()
    dataset_final_day = luigi.Parameter()
    base_dataset_version = luigi.Parameter()
    observation_shape = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super(CQLReplayDataset, self).__init__(*args, **kwargs)

        print("----- Initializing Generate Dataset")

        self.timer = TaskTimer()

        self.dataset_folder = "data/processed/{}_dataset/".format(self.dataset_version)
        self.meta_folder = os.path.join(self.dataset_folder, "metadata")

        self.ds_folder = {
            "name": self.dataset_name,
            "path": os.path.join(self.dataset_folder, self.dataset_name),
        }

        os.makedirs(os.path.join(self.dataset_folder, self.dataset_name), exist_ok=True)

        os.makedirs(self.meta_folder, exist_ok=True)

        self.dataset_meta_file_path = os.path.join(
            self.meta_folder, "dataset_metadata.csv"
        )

        self.output_file_path = os.path.join(self.meta_folder, "data_paths.json")

    def output(self):
        return luigi.LocalTarget(self.output_file_path)

    def run(self):
        print("----- Generating dataset")

        replay_log_path = os.path.join(self.ds_folder["path"], "replay_logs")

        # create a logged out-of-graph Replay Buffer
        self.replay_buffer = WrappedLoggedReplayBuffer(
            log_dir=replay_log_path,
            observation_shape=(self.observation_shape,),
            stack_size=1,
            use_staging=False,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.float32,
        )

        delta = dt.timedelta(days=1)
        initial_day = dt.datetime.strptime(self.dataset_initial_day, "%Y-%m-%d").date()
        final_day = dt.datetime.strptime(self.dataset_final_day, "%Y-%m-%d").date()

        current_day = initial_day
        while current_day <= final_day:

            self.generate_dataset(current_day.strftime("%Y-%m-%d"))

            os.makedirs(
                os.path.join(self.ds_folder["path"], current_day.strftime("%Y-%m-%d")),
                exist_ok=True,
            )
            with open(
                os.path.join(
                    self.ds_folder["path"], current_day.strftime("%Y-%m-%d"), "Ok"
                ),
                "w",
            ) as fp:
                json.dump({}, fp)

            current_day += delta

        self.replay_buffer.memory.log_final_buffer()

        data_paths = {
            "dataset": self.ds_folder,
            "dataset_meta": self.dataset_meta_file_path,
        }
        with open(self.output_file_path, "w") as fp:
            json.dump(data_paths, fp)

    def generate_dataset(self, current_day):
        self.timer.start("Generating " + self.ds_folder["name"])

        # get ray dataset current date files
        dataset_path = "data/processed/{}_dataset/{}/{}".format(
            self.base_dataset_version, self.ds_folder["name"], current_day
        )
        try:
            dataset_files = []
            for files in os.listdir(os.path.join(dataset_path)):
                if os.path.isfile(os.path.join(dataset_path, files)):
                    dataset_files.append(os.path.join(files))
            dataset_files = sorted(dataset_files)

        except:
            print("No such file or directory: {}".format(dataset_path))

        else:
            for dataset_file in tqdm(dataset_files):
                reader = JsonReader(os.path.join(dataset_path, dataset_file))

                with open(os.path.join(dataset_path, dataset_file), "r") as f:
                    sb = f.readlines()

                for _ in range(len(sb)):
                    batch = reader.next()

                    for i in range(len(batch["eps_id"])):
                        self.replay_buffer.memory.add(
                            batch["obs"][i],
                            batch["actions"][i],
                            batch["rewards"][i],
                            batch["dones"][i],
                        )

        print(
            "-----\nReplay Buffer number {}.\nCurrent Size: {}\nTotal size: {}\n-----".format(
                self.replay_buffer.memory._log_count,
                self.replay_buffer.memory.add_count
                % self.replay_buffer.memory._replay_capacity,
                self.replay_buffer.memory.add_count,
            )
        )

        self.timer.end("Generating " + self.ds_folder["name"])
