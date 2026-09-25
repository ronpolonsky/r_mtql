"""Language-conditioned MTQL MLP-critic ablation for real DROID data."""

from agents.language_conditioned import (
    LanguageMTQLMLPAgent,
    mlp_language_config,
)


class MTQLMLPLanguageRealAgent(LanguageMTQLMLPAgent):
    """Real-DROID MTQL actor with a language-conditioned MLP critic."""


def get_config():
    return mlp_language_config()

