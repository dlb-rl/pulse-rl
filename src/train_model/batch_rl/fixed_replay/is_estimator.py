class ImportanceSamplingEstimator:
    """The step-wise IS estimator.

    Step-wise IS estimator described in https://arxiv.org/pdf/1511.03722.pdf"""

    def estimate(self, batch, actions_prob, reward_shift):
        rewards = batch["rewards"]
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

        # calculate stepwise IS estimate
        V_prev, V_step_IS = 0.0, 0.0
        for t in range(batch.count):
            reward = rewards[t] + reward_shift
            V_prev += reward * 0.99 ** t
            V_step_IS += p[t] * reward * 0.99 ** t

        estimation = {
            "V_prev": V_prev,
            "V_step_IS": V_step_IS,
            "V_gain_est": V_step_IS / max(1e-8, V_prev),
        }
        return estimation
