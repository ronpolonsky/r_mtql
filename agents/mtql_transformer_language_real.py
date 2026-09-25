"""Language-conditioned MTQL transformer for real DROID trajectories.

This is a new agent variant.  It uses raw augmented images and one learned
prompt token; it does not draw the target onto the image.
"""

from agents.language_conditioned import (
    LanguageMTQLTransformerAgent,
    transformer_language_config,
)


class MTQLTransformerLanguageRealAgent(LanguageMTQLTransformerAgent):
    """Real-DROID MTQL transformer conditioned on the language prompt."""


def get_config():
    return transformer_language_config()

