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

# Lint as: python3
"""Helper functions for multi head/network (Ensemble-DQN and REM) agents."""

import collections
import numpy as np
import tensorflow.compat.v1 as tf

MultiHeadNetworkType = collections.namedtuple(
    "multi_head_dqn_network", ["q_heads", "unordered_q_heads", "q_values"]
)
DQNNetworkType = collections.namedtuple("dqn_network", ["q_values"])
MultiNetworkNetworkType = collections.namedtuple(
    "multi_network_dqn_network", ["q_networks", "unordered_q_networks", "q_values"]
)
QuantileNetworkType = collections.namedtuple(
    "qr_dqn_network", ["q_values", "logits", "probabilities"]
)


class QuantileNetwork(tf.keras.Model):
    """Keras network for QR-DQN agent.

    Attributes:
      num_actions: An integer representing the number of actions.
      num_atoms: An integer representing the number of quantiles of the value
        function distribution.
      conv1: First convolutional tf.keras layer with ReLU.
      conv2: Second convolutional tf.keras layer with ReLU.
      conv3: Third convolutional tf.keras layer with ReLU.
      flatten: A tf.keras Flatten layer.
      dense1: Penultimate fully-connected layer with ReLU.
      dense2: Final fully-connected layer with `num_actions` * `num_atoms` units.
    """

    def __init__(
        self, num_actions: int, num_atoms: int, name: str = "quantile_network"
    ):
        """Convolutional network used to compute the agent's Q-value distribution.

        Args:
          num_actions: int, number of actions.
          num_atoms: int, the number of buckets of the value function distribution.
          name: str, used to create scope for network parameters.
        """
        super(QuantileNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        activation_fn = tf.keras.activations.relu  # ReLU activation.
        self._kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode="fan_in", distribution="uniform"
        )
        # Defining layers.
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=8,
            strides=4,
            padding="same",
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.dense2 = tf.keras.layers.Dense(
            units=num_actions * num_atoms,
            kernel_initializer=self._kernel_initializer,
            activation=None,
        )

    def call(self, state):
        """Calculates the distribution of Q-values using the input state tensor."""
        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.0)
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.flatten(net)
        net = self.dense1(net)
        net = self.dense2(net)
        logits = tf.reshape(net, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(tf.zeros_like(logits))
        q_values = tf.reduce_mean(logits, axis=2)
        return QuantileNetworkType(q_values, logits, probabilities)