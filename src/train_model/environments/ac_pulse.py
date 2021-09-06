import gym
import numpy as np
from ray.rllib.env import VectorEnv


class ACPulse(VectorEnv):
    def __init__(self, env_config):
        self.observation_shape = (env_config["obs_shape"],)
        self.observation_dtype = np.float32

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self.observation_shape, dtype=self.observation_dtype
        )
        self.num_envs = 1

    def get_unwrapped(self):
        return [self for _ in range(self.num_envs)]

    def stop(self):
        return