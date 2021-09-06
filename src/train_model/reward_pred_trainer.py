# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    IterableDataset,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from datetime import datetime
import random
import math
import luigi
import yaml
import os
import numpy as np

import subprocess

import ray
from ray import tune
from ray.rllib.offline.json_reader import JsonReader

from src.task_timer import TaskTimer
from src import utils

from src.running_stats import RunningMeanStd


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc4 = nn.Linear(128, 1, bias=True)
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
        return x


class DataReader(IterableDataset):
    def __init__(self, data_path, batch_size=0):
        self.reader = JsonReader(data_path)
        self.batch_size = batch_size
        self.length = len(data_path)

    def __iter__(self):
        return self

    def __next__(self):
        sample_batch = self.reader.next()
        while True:
            if sample_batch.count >= self.batch_size:
                sample_batch.shuffle()
                return (
                    np.concatenate(
                        (
                            sample_batch["obs"],
                            np.reshape(
                                sample_batch["actions"],
                                (sample_batch["actions"].shape[0], 1),
                            ),
                        ),
                        axis=1,
                    ),
                    sample_batch["rewards"],
                )
            else:
                sample_batch = sample_batch.concat(self.reader.next())


def evaluate(testing_part, model, device, running_stats):
    model.eval()
    errors = []
    percent_errors = []
    with torch.no_grad():
        all_targets = torch.Tensor([])
        all_scores = torch.Tensor([])
        for p in testing_part:
            loader = DataReader([p])
            data, targets = next(loader)

            data = torch.Tensor(data).to(device)

            targets = torch.Tensor(
                np.log1p((targets - 0.003).reshape((targets.shape[0], 1)))
                / math.sqrt(running_stats.var)
            ).to(device)

            scores = model(data)

            targets_cpu = targets.cpu()
            scores_cpu = scores.cpu()

            targets_raw = (torch.exp(targets_cpu) - 1) * math.sqrt((running_stats.var))
            scores_raw = (torch.exp(scores_cpu) - 1) * math.sqrt((running_stats.var))
            all_targets = torch.cat(
                (
                    all_targets,
                    targets_raw,
                )
            )
            all_scores = torch.cat(
                (
                    all_scores,
                    scores_raw,
                )
            )
            scores = model(data)
            error = (abs(targets_raw - scores_raw)).mean()
            percent_error = (
                abs(targets_raw - scores_raw) / (abs(targets_raw) + 1e-7)
            ).mean()
            errors.append(error.mean())
            percent_errors.append(percent_error.mean())

    with tune.checkpoint_dir(step="hist") as checkpoint_dir:
        bins = np.linspace(-1, 30, 100)
        plt.hist(all_scores.numpy(), bins, log=True, alpha=0.5, label="x")
        plt.hist(all_targets.numpy(), bins, log=True, alpha=0.5, label="y")
        plt.legend(loc="upper right")
        plt.xlabel("Rewards")
        plt.savefig(
            checkpoint_dir
            + "/{}_{}.png".format("hist", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        plt.clf()

    print("--- Test results ---")
    mean_err = 0
    for err in errors:
        mean_err += err
    mean_err = mean_err / len(errors)
    meanp_err = 0
    for err in percent_errors:
        meanp_err += err
    meanp_err = meanp_err / len(percent_errors)

    model.train()


def weighted_loss(loss, targets):
    if targets.std() == 0.0:
        return loss.mean()
    else:
        w = abs(targets)
        loss = loss + loss * w
        return loss.mean()


def shrinkage_loss(loss, config):
    a = config["a"]
    c = config["c"]
    loss = loss / (1 + torch.exp(a * (c - torch.sqrt(loss))))
    return loss.mean()


def train(config, checkpoint_dir=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = config["input_size"] + 1  # +1 for the actions that are input
    learning_rate = config["learning_rate"]  # 0.001
    num_epochs = config["num_epochs"]  # 1000
    batch_size = config.get("batch_size", 20000)  # 50000
    momentum = config.get("momentum", 0.0)  # 0.9
    lr_decay = config.get("learning_rate_decay", 1.0)  # 0.99

    # validation_dataset = config.get("validation_dataset", False)
    opt = config["optimizer"]
    checkpoint_freq = config["checkpoint_freq"]
    eval_freq = config["eval_freq"]

    train_path = config["dataset_path"]

    random.shuffle(train_path)
    training_part = train_path[: int(len(train_path) * 0.8)]
    testing_part = train_path[int(len(train_path) * 0.8) :]

    epoch_size = config.get("epoch_size", 1)

    train_loader = DataReader(training_part, batch_size)

    # Initialize network
    model = NN(input_size=input_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction="none")
    if opt == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, momentum=momentum
        )
    elif opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise Exception("Invalid optimizer for 'opt'")

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    running_stats = RunningMeanStd()

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state, scheduler_state, running_stats_state = torch.load(
            checkpoint
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        running_stats.load_dict(running_stats_state)

    # Train Network
    losses = 0
    stats = {"loss": [], "error": [], "percent_error": []}
    for epoch in range(num_epochs):
        batches = 0
        error = 0
        losses = 0
        percent_error = 0
        for data, targets in train_loader:
            data = torch.Tensor(data).to(device)
            running_stats.update(targets - 0.003)
            targets = torch.Tensor(
                np.log1p((targets - 0.003).reshape((targets.shape[0], 1)))
                / math.sqrt(running_stats.var)
            ).to(device)

            # forward
            scores = model(data)
            targets_cpu = targets.cpu()
            scores_cpu = scores.cpu()
            error += (
                abs(
                    ((torch.exp(targets_cpu) - 1) * math.sqrt(running_stats.var))
                    - ((torch.exp(scores_cpu) - 1) * math.sqrt(running_stats.var))
                )
            ).mean()
            percent_error += (
                abs(
                    ((torch.exp(targets_cpu) - 1) * math.sqrt(running_stats.var))
                    - ((torch.exp(scores_cpu) - 1) * math.sqrt(running_stats.var))
                )
                / (
                    abs(((torch.exp(targets_cpu) - 1) * math.sqrt(running_stats.var)))
                    + 1e-7
                )
            ).mean()
            loss = criterion(scores, targets)
            loss = weighted_loss(loss, targets)
            losses += loss

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            batches += 1

            if batches >= epoch_size:
                break

        scheduler.step()

        stats["loss"].append(losses / batches)
        stats["error"].append(error / batches)
        stats["percent_error"].append(percent_error / batches)

        if epoch % eval_freq == 0:
            evaluate(testing_part, model, device, running_stats)

        if epoch % checkpoint_freq == 0 or epoch == max(range(num_epochs)):
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        running_stats.state_dict(),
                    ),
                    path,
                )

        tune.report(
            loss=stats["loss"][-1].item(),
            error=stats["error"][-1].item(),
            percent_error=stats["percent_error"][-1].item(),
            learning_rate=scheduler.get_last_lr()[0],
        )


def get_dataset(config):
    dataset_path = config["path"]
    dataset = []
    for path in os.listdir(dataset_path):
        for f in os.listdir(os.path.join(dataset_path, path)):
            if os.path.isfile(os.path.join(dataset_path, path, f)):
                dataset.append(os.path.abspath(os.path.join(dataset_path, path, f)))
    dataset = sorted(dataset)
    return dataset


class TrainPredictor(luigi.Task):

    model_version = luigi.ChoiceParameter(choices=["reward_pred"])

    def __init__(self, *args, **kwargs):
        super(TrainPredictor, self).__init__(*args, **kwargs)

        self.dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_folder = os.path.abspath("models/{}_model".format(self.model_version))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder

        self.timer = TaskTimer()

        self.config = self.model_config

        ray.init(address="auto", dashboard_host="0.0.0.0")

    @property
    def model_config(self):
        path = os.path.abspath("models/{}_model.yaml".format(self.model_version))
        with open(path) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)

        return model_config

    def run(self):
        print("----- Training model")

        # Configure search space
        for k in self.config.keys():
            if isinstance(self.config[k], list):
                self.config[k] = tune.grid_search(self.config[k])
            elif isinstance(self.config[k], dict):
                for q in self.config[k].keys():
                    if isinstance(self.config[k][q], list):
                        self.config[k][q] = tune.grid_search(self.config[k][q])

        dataset_path = get_dataset(self.config["dataset"])
        self.config["dataset_path"] = dataset_path

        stop = self.config["stop"]
        tune.run(
            train,
            name=self.config["name"],
            config=self.config,
            local_dir=self.save_folder,
            stop=stop,
            resources_per_trial=self.config["resources_per_trial"],
            resume=self.config.get("resume", False),
        )
