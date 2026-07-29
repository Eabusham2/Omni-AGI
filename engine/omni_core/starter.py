"""Project-authored seed corpus and manifest for the bundled Omni Starter.

This is deliberately small enough to materialize on ordinary hardware. It is
an auditable baseline, not a claim of frontier capability. The text is training
data only; it is never inserted as a hidden runtime prompt.
"""

import hashlib
import json
from typing import Any, Dict, List, Tuple


STARTER_CORPUS: List[str] = [
    (
        "Language expresses connected ideas at several scales: sounds and "
        "symbols form words, words form relations, and relations form a whole."
    ),
    (
        "A remembered experience can strengthen distributed neural assemblies "
        "and the synapses between them while temporary context later fades."
    ),
    (
        "Evidence may be incomplete. Competing explanations can remain active "
        "until observation, calculation, or an external tool distinguishes them."
    ),
    (
        "A file action has a typed operation, a path, structured arguments, a "
        "result, and an auditable effect."
    ),
    (
        "A web action can search or fetch public information and return sources "
        "that support later learning."
    ),
    (
        "A code action forms a program, runs a test, observes the result, and "
        "uses the evidence in the next attempt."
    ),
    (
        "Imagination maps an internal idea assembly into image, sound, or moving "
        "visual latents without requiring a remembered passage as a text prompt."
    ),
    (
        "An agent fork explores an isolated line of work and returns evidence, "
        "artifacts, and replayable experience for a reviewed merge."
    ),
    (
        "Learning changes fast spike-timing synapses immediately and changes "
        "slow ternary cortical parameters through continued prediction training."
    ),
    (
        "Improvement is an experiment: identify a limitation, create a candidate, "
        "measure it on held-out tasks, retain evidence, and roll back regressions."
    ),
    (
        "Temporal state links earlier activity with present perception. Rehearsal "
        "can stabilize useful pathways while unused activity gradually decays."
    ),
    (
        "A coherent answer integrates the relevant whole before choosing the next "
        "language token, action, question, or additional internal computation."
    ),
]


STARTER_ACTION_EXAMPLES: List[Tuple[str, str]] = [
    ("hello, tell me what you notice", "talk"),
    ("explain the relationship in ordinary language", "talk"),
    ("read the selected file and inspect its structure", "tool"),
    ("search the web for current primary sources", "tool"),
    ("make an image from this internal scene", "imagine"),
    ("generate a short sound for this idea", "imagine"),
    ("fork agents to investigate these independent parts", "agent"),
    ("let another branch test this hypothesis", "agent"),
    ("stay with this uncertainty and compute further", "ponder"),
    ("consider the unresolved alternatives again", "ponder"),
    ("learn this dataset into the neural substrate", "learn"),
    ("consolidate this experience into long term synapses", "learn"),
    ("create and evaluate an improvement candidate", "evolve"),
    ("test whether the improved learner improves itself", "evolve"),
    ("stop the current activity", "stop"),
    ("cancel and leave the state unchanged", "stop"),
]


def starter_manifest() -> Dict[str, Any]:
    payload = json.dumps(
        {
            "corpus": STARTER_CORPUS,
            "actions": STARTER_ACTION_EXAMPLES,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "id": "omni-starter-bundled-1",
        "format": "omni-starter-training-manifest",
        "architecture": "OmniCortex",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "examples": len(STARTER_CORPUS),
        "actionTrajectories": len(STARTER_ACTION_EXAMPLES),
        "objectives": [
            "next-byte corpus prediction",
            "whole-experience idea reconstruction",
            "temporal prediction",
            "spike homeostasis",
            "structured action trajectory imitation",
        ],
        "preferenceTraining": False,
        "rewardModel": False,
        "rlhf": False,
        "dpo": False,
        "personaPrompt": False,
        "source": "Project-authored synthetic seed corpus",
        "license": "PolyForm-Noncommercial-1.0.0-or-commercial-license",
    }
