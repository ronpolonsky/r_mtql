"""Real MTQL MLP with all-outcome critic and success-only actor."""

from agents.mtql_mlp_real import (
    MTQLMLPRealAgent,
    get_config as _get_base_config,
)
from agents.mtql_separate_batches import MTQLSeparateBatchesMixin


class MTQLMLPSuccessActorRealAgent(
    MTQLSeparateBatchesMixin,
    MTQLMLPRealAgent,
):
    """Change only the source batches used by inherited MTQL losses."""


def get_config():
    config = _get_base_config()
    config.agent_name = "mtql_mlp_success_actor_real"
    config.training_batch_contract = (
        "critic_all_outcomes_actor_success_only"
    )
    return config

