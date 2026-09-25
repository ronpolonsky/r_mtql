import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_flat = jnp.concatenate(
            [jnp.reshape(g, -1) for g in jax.tree_util.tree_leaves(grads)], axis=0
        )
        final_grad_max = jnp.max(grad_flat)
        final_grad_min = jnp.min(grad_flat)
        final_grad_norm = jnp.linalg.norm(grad_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch, metadata: Mapping[str, Any] | None = None):
    """Atomically save the complete agent state and optional training metadata.

    The Flax state dict includes every pytree field, including model and target
    parameters, optimizer state, optimizer steps, and agent RNG state.
    """

    save_dict = {
        'agent': flax.serialization.to_state_dict(agent),
    }
    if metadata is not None:
        save_dict['metadata'] = metadata
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    tmp_path = f'{save_path}.tmp-{os.getpid()}'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, save_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f'Saved to {save_path}')


def restore_agent(
    agent,
    restore_path,
    restore_epoch,
    ignore_unknown: bool = True,
    reset_opt_state: bool = False,
    return_metadata: bool = False,
):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
        reset_opt_state: If True, keep only params/step from the checkpoint and use the
            freshly-initialized optimizer state.  Necessary whenever the optimizer chain
            structure changes between the saved run and the current run (e.g. adding
            grad-norm clipping changes the chain length and makes opt_state shapes
            incompatible).
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    state = load_dict['agent']
    metadata = load_dict.get('metadata', {})
    if ignore_unknown:
        target_state = flax.serialization.to_state_dict(agent)

        def _filter_state_dict(source, reference):
            if not isinstance(source, Mapping) or not isinstance(reference, Mapping):
                return source
            return {
                key: _filter_state_dict(source[key], reference[key])
                for key in reference.keys()
                if key in source
            }

        state = _filter_state_dict(state, target_state)

    if reset_opt_state:
        # Replace the loaded opt_state with the freshly initialized one so that
        # changes to the optimizer chain (e.g. adding clip_by_global_norm) don't
        # cause shape mismatches.
        fresh_state = flax.serialization.to_state_dict(agent)
        state['network']['opt_state'] = fresh_state['network']['opt_state']

    agent = flax.serialization.from_state_dict(agent, state)

    print(f'Restored from {restore_path}')

    if return_metadata:
        return agent, metadata
    return agent


def count_params(params):
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


def print_param_stats(agent):
    params = agent.network.params
    total = count_params(params)

    groups = {}
    for key in params:
        n = count_params(params[key])
        groups[key] = n

    print(f"\nParameter counts:")
    print(f"  {'Total':<40} {total:>12,}")
    print(f"  {'-' * 54}")
    for key, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {key:<40} {n:>12,}")
    print()
