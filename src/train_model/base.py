import os
import sys
import yaml
import copy
import json
import luigi
import random
import datetime

import ray

from .ray_trainer import RayTrainer
from .cql_trainer import CQLTrainer
from .evaluation import Evaluation


TRAINER = {"RAY": RayTrainer, "CQL": CQLTrainer}


class TrainModel(luigi.Task):
    model_version = luigi.ChoiceParameter(choices=["CQL", "DQN", "BC"])

    is_cron = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(TrainModel, self).__init__(*args, **kwargs)

        ray.init(address="auto", dashboard_host="0.0.0.0")

        self.dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def model_config(self):
        path = os.path.abspath("models/{}_model.yaml".format(self.model_version))
        with open(path) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)

        return model_config

    def run(self):

        # Training Task
        print("----- Training model")
        train_output = yield TRAINER[self.model_config["trainer_version"]](
            dtime=self.dtime,
            model_version=self.model_version,
            algorithm=self.model_config["algorithm"],
            env=self.model_config["env"],
            dataset_version=self.model_config["dataset_version"],
            config=self.model_config["config"],
            **self.model_config["train"],
        )

        if (
            "only_during_train" not in self.model_config["evaluation"]
            or not self.model_config["evaluation"]["only_during_train"]
        ):
            print("----- Evaluating model")
            yield Evaluation(
                train_output=train_output.path,
                model_version=self.model_version,
                algorithm=self.model_config["algorithm"],
                dataset_version=self.model_config["dataset_version"],
                **self.model_config["evaluation"],
            )
