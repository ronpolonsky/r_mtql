"""MTQL transformer agent for the real DROID observation/action interface.

This is intentionally a thin real-interface adapter around the original
``MTQLTransformerAgent``.  The optimization, transformer, history, and flow
logic remain identical to ``agents/mtql_transformer.py``; the real-data entry
point supplies dictionary observations (images + proprioception) and 7-D
Cartesian/gripper actions at runtime.
"""

from agents.mtql_transformer import (
    MTQLTransformerAgent,
    get_config as _get_base_config,
)


class MTQLTransformerRealAgent(MTQLTransformerAgent):
    """Original MTQL transformer with the real DROID interface contract."""


def get_config():
    """Return the base MTQL hyperparameters with real-data defaults."""

    config = _get_base_config()
    config.agent_name = "mtql_transformer_real"
    # DROID observations are {image, wrist_image, proprio}; the MTQL
    # registry's ``resnet`` key is a lightweight IMPALA encoder that consumes
    # both visual views and proprioception. It is intentionally not EXPO's
    # separate ResNetV2 critic encoder.
    # ``proprio_only`` is the image-free diagnostic alternative.
    config.encoder = "resnet"
    # Match the shared counting-scoop MTQL hyperparameters.
    config.critic_grad_clip = 5.0
    config.alpha = 300
    config.attention_entropy_target = ((3.5, 3.5), (3.0, 3.0))
    return config

