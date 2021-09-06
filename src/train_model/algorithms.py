import numpy as np
from ray.rllib.agents.marwil.marwil import MARWILTrainer
from ray.rllib.agents.marwil.marwil_tf_policy import MARWILTFPolicy
from ray.rllib.agents.marwil.bc import BC_DEFAULT_CONFIG, validate_config

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import explained_variance
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

import tensorflow_addons as tfa

tf1, tf, tfv = try_import_tf()

# From https://github.com/ray-project/ray/blob/master/rllib/agents/marwil/marwil_tf_policy.py
class BCLoss:
    def __init__(
        self,
        policy,
        value_estimates,
        action_dist,
        actions,
        cumulative_rewards,
        vf_loss_coeff,
        beta,
    ):
        # VF is not fitted on BC
        # Advantage Estimation.
        adv = cumulative_rewards - value_estimates
        adv_squared = tf.reduce_mean(tf.math.square(adv))
        # Value function's loss term (MSE).
        self.v_loss = 0.5 * adv_squared

        # L = - A * log\pi_\theta(a|s)
        logprobs = action_dist.logp(actions)
        # multiply actions by their proportions (e.g. 10:1 for 1s and 0s)
        logprobs *= tf.cast(actions * 49 + 1, tf.float32)

        self.p_loss = -1.0 * tf.reduce_mean(logprobs)

        self.total_loss = self.p_loss + vf_loss_coeff * self.v_loss

        self.explained_variance = tf.reduce_mean(
            explained_variance(cumulative_rewards, value_estimates)
        )


def bc_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)
    value_estimates = model.value_function()

    policy.loss = BCLoss(
        policy,
        value_estimates,
        action_dist,
        train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        policy.config["vf_coeff"],
        policy.config["beta"],
    )

    return policy.loss.total_loss


# for some reason this must be redeclared here
# since errors are thrown if we use the original implementation
def stats(policy, train_batch):
    return {
        "policy_loss": policy.loss.p_loss,
        "vf_loss": policy.loss.v_loss,
        "total_loss": policy.loss.total_loss,
        "vf_explained_var": policy.loss.explained_variance,
        "lr": policy.cur_lr,
    }


@tf.function
def optimizer(
    policy: Policy, config: TrainerConfigDict
) -> "tf.keras.optimizers.Optimizer":
    if config["optimizer"] == "radam":
        print("Using RAdam Optimizer")
        return tfa.optimizers.RectifiedAdam(
            lr=policy.cur_lr, epsilon=config["optim_epsilon"]
        )
    else:
        print("Using Adam Optimizer")
        if policy.config["framework"] in ["tf2", "tfe"]:
            return tf.keras.optimizers.Adam(
                learning_rate=policy.cur_lr, epsilon=config["optim_epsilon"]
            )
        else:
            return tf1.train.AdamOptimizer(
                learning_rate=policy.cur_lr, epsilon=config["optim_epsilon"]
            )


def setup_early_mixins(
    policy: Policy, obs_space, action_space, config: TrainerConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


BC_DEFAULT_CONFIG["lr_schedule"] = None
BC_DEFAULT_CONFIG["optim_epsilon"] = 1e-7

BCTFPolicy = MARWILTFPolicy.with_updates(
    loss_fn=bc_loss,
    stats_fn=stats,
    # optimizer_fn=optimizer,
    before_init=setup_early_mixins,
    mixins=[
        LearningRateSchedule,
    ],
)

CustomBCTrainer = MARWILTrainer.with_updates(
    name="BC",
    default_config=BC_DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=BCTFPolicy,
)
