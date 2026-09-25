"""Real MTQL Transformer with all-outcome critic and success-only actor."""

from agents.mtql_separate_batches import MTQLSeparateBatchesMixin
from agents.mtql_transformer_real import (
    MTQLTransformerRealAgent,
    get_config as _get_base_config,
)


class MTQLTransformerSuccessActorRealAgent(
    MTQLSeparateBatchesMixin,
    MTQLTransformerRealAgent,
):
    """Change only the source batches used by inherited MTQL losses."""


def get_config():
    config = _get_base_config()
    config.agent_name = "mtql_transformer_success_actor_real"
    config.training_batch_contract = (
        "critic_all_outcomes_actor_success_only"
    )
    return config

