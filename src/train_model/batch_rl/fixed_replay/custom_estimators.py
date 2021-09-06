from ray.rllib.offline.off_policy_estimator import OffPolicyEstimator, OffPolicyEstimate
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import SampleBatchType

import numpy as np


def get_registry_reward(batch, event_registry_position=-23):

    rewards = []
    for obs in batch["new_obs"]:
        r = obs[-event_registry_position]
        rewards.append(r)

    return np.asarray(rewards)


def get_deal_reward(batch, event_deal_position=-20):

    rewards = []
    for obs in batch["new_obs"]:
        r = obs[-event_deal_position]
        rewards.append(r)

    return np.asarray(rewards)


def get_ltv_reward(batch, reward_shift, event_activated_position=-24):

    rewards = []
    for obs, rew in zip(batch["new_obs"], batch["rewards"]):
        r = rew + reward_shift + 0.001 + (0.033 * obs[event_activated_position])
        rewards.append(r)

    return np.asarray(rewards)


class CustomImportanceSamplingEstimator:
    """The step-wise IS estimator.
    Step-wise IS estimator described in https://arxiv.org/pdf/1511.03722.pdf"""

    def estimate(self, batch, actions_prob, rewards):
        old_prob = batch["action_prob"]
        new_prob = actions_prob

        # calculate importance ratios
        p = []
        for t in range(batch.count):
            if t == 0:
                pt_prev = 1.0
            else:
                pt_prev = p[t - 1]
            p.append(pt_prev * new_prob[t][batch["actions"][t]] / old_prob[t])

        if (np.asarray(rewards) == np.zeros(np.asarray(rewards).shape)).all():
            estimation = None

        else:
            # calculate stepwise IS estimate
            V_prev, V_step_IS = 0.0, 0.0
            for t in range(batch.count):
                # reward = rewards[t] + reward_shift
                V_prev += rewards[t] * 0.99 ** t
                V_step_IS += p[t] * rewards[t] * 0.99 ** t

            estimation = {
                "V_prev": V_prev,
                "V_step_IS": V_step_IS,
                "V_gain_est": V_step_IS / max(1e-8, V_prev),
            }
        return estimation