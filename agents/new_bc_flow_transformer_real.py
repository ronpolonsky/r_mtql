"""Real-DROID adapter for the matched pure flow-matching BC baseline.

The implementation is intentionally a thin subclass of
``agents/new_bc_flow_transformer.py``. It uses the same MTQL flow actor and
loss; only the agent name and real-data encoder defaults differ.
"""

from agents.new_bc_flow_transformer import (
    NewBCFlowTransformerAgent,
    get_config as _get_bc_config,
)


class NewBCFlowTransformerRealAgent(NewBCFlowTransformerAgent):
    """Pure behavior cloning on normalized real-DROID trajectories."""


def get_config():
    """Return the flow-BC configuration for the real DROID adapter."""

    config = _get_bc_config()
    config.agent_name = "new_bc_flow_transformer_real"
    config.encoder = "resnet"
    config.actor_lr = 1e-4
    config.actor_grad_clip = 5.0
    return config
