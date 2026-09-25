#!/usr/bin/env python
"""Evaluate checkpoints from the success-only-actor MTQL variants.

The separate-batch classes differ from the existing real agents only in the
training-only ``update`` path. Evaluation therefore maps each new config to
its structurally identical base agent before delegating to the unchanged
robot evaluator.
"""

from absl import app

from eval_mtql_droid import FLAGS, main as _base_main


_EVAL_AGENT_NAMES = {
    "mtql_transformer_success_actor_real": "mtql_transformer_real",
    "mtql_mlp_success_actor_real": "mtql_mlp_real",
}


def main(argv):
    training_agent_name = FLAGS.agent.get("agent_name")
    try:
        eval_agent_name = _EVAL_AGENT_NAMES[training_agent_name]
    except KeyError as error:
        raise ValueError(
            "eval_mtql_droid_success_actor.py requires one of the new "
            "success-only-actor configs; got "
            f"agent_name={training_agent_name!r}."
        ) from error

    # The base evaluator's allowlist predates these training-only subclasses.
    # Their parameter trees and inference methods are otherwise identical.
    FLAGS.agent.agent_name = eval_agent_name
    return _base_main(argv)


if __name__ == "__main__":
    app.run(main)
