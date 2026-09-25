"""MTQL real-DROID agent with the MLP-critic ablation.

This is the real-data counterpart of ``agents/mtql_mlp.py``. The MLP
variant keeps the MTQL actor, image/proprio encoder, history representation,
action chunking, and optimization interface unchanged; only the Q critic is
replaced by the flattened-token MLP defined in ``mtql_mlp.py``.
"""

from agents.mtql_mlp import MTQLMLPAgent, get_config as _get_mlp_config


class MTQLMLPRealAgent(MTQLMLPAgent):
    """Real DROID interface adapter around the MTQL MLP-critic agent."""


def get_config():
    """Return MLP-critic settings with the real DROID defaults."""

    config = _get_mlp_config()
    config.agent_name = "mtql_mlp_real"
    # Match the transformer real run's visual/proprio input path and critic
    # optimization settings. The MLP critic does not use attention entropy,
    # so the entropy target is intentionally irrelevant for this variant.
    config.encoder = "resnet"
    config.critic_grad_clip = 5.0
    config.alpha = 300
    config.attention_entropy_target = ((3.5, 3.5), (3.0, 3.0))
    return config
