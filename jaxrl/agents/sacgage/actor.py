from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, log_std_min: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, log_stds, means = actor.apply_fn({'params': actor_params}, batch.observations, log_std_min=log_std_min)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        ent_coef = 0.0
        actor_loss = (log_probs * ent_coef - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            "std_mean": jnp.mean(jnp.exp(log_stds)), "mean_min": jnp.min(means), "mean_max": jnp.max(means)
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def update_gmean(key: PRNGKey, actor: Model,
           batch: Batch, log_std_min: float, gmean_factor: float) -> Tuple[Model, InfoDict]:

    def actor_gmean_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, _, means = actor.apply_fn({'params': actor_params}, batch.observations, log_std_min=log_std_min)
        actor_loss = gmean_factor * jnp.sum(means**2, axis=-1).mean()
        return actor_loss, {
            'actor_loss_gmean': actor_loss,
        }

    new_actor, info = actor.apply_gradient(actor_gmean_fn)

    return new_actor, info
