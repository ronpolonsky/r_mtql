"""Language-conditioned pure flow-BC baseline for real DROID data."""

from agents.language_conditioned import (
    LanguageNewBCFlowTransformerAgent,
    bc_language_config,
)


class NewBCFlowTransformerLanguageRealAgent(LanguageNewBCFlowTransformerAgent):
    """Pure flow matching BC conditioned on the language prompt."""


def get_config():
    return bc_language_config()

