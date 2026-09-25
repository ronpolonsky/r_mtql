"""MTQL update mixin for separate critic and actor training batches.

The critic batch may contain successful and failed trajectories. The actor
batch is sampled independently from successful trajectories only. All loss
definitions and network modules remain inherited from the existing MTQL
agents.
"""

import jax


class MTQLSeparateBatchesMixin:
    """Route independent batches through the existing critic/actor losses."""

    @jax.jit
    def total_loss(
        self,
        critic_batch,
        actor_batch,
        grad_params,
        rng=None,
        step=None,
    ):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(
            critic_batch,
            grad_params,
            critic_rng,
            step=step,
        )
        for key, value in critic_info.items():
            info[f"critic/{key}"] = value

        actor_loss, actor_info = self.actor_loss(
            actor_batch,
            grad_params,
            actor_rng,
        )
        for key, value in actor_info.items():
            info[f"actor/{key}"] = value

        return critic_loss + actor_loss, info

    @jax.jit
    def update(self, critic_batch, actor_batch, step=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                critic_batch,
                actor_batch,
                grad_params,
                rng=rng,
                step=step,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

