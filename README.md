# PulseRL

Code for PulseRL: Enabling Offline Reinforcement Learning for Digital Marketing Systems via Conservative Q-Learning.

In this repository we provide code for PulseRL offline experiments described in the paper. Part of our code is built on top of the [CQL](https://github.com/aviralkumar2907/CQL) repository.

# Installation

First install install Python 3.x. The remaining dependencies can be installed by executing the following command:

    sh dependencies.sh

We use a customized version of ray, to fix some bugs in the desired features. The above script will build Ray from source available [here](https://github.com/dlb-rl/ray), among others dependencies.

# Usage

Download and extract the provided processed [dataset](https://bit.ly/ac-dataset) into the folder `data`. The experiments featured in the paper can be executed by running the provided bash script, as follows:

    sh train_model.sh -m <MODEL>

Where available models are: "BC", "DQN", "CQL". The respective models configuration files can be found at `models/<MODEL>_model.yaml`.

If you want to evaluate an already trained model, you can run the following:

    sh evaluation.sh -m <MODEL> -d <DATASET_NAME> -v <DATASET_VERSION> -i <TRAIN_ID> -w <NUM_WORKERS>

To train the reward predictor model, you can run:

    sh train_reward_pred_model.sh

To view the training and evaluation graphs, just run:

    tensorboard --logdir models/

In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
