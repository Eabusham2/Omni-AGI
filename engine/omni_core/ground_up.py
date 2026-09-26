"""Versioned, auditable local curriculum metadata for ground-up builds.

The current v3 contract admits only immutable, platform-neutral tool/action
records. Synthetic media and imagination-selector fixtures are explicitly
empty: real user-selected or captured media is the first source allowed to
train those parameter groups.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .native_curriculum import (
    NATIVE_ACTION_EXAMPLES,
    NATIVE_TOOL_NEGATIVE_EXAMPLES,
    NATIVE_TOOL_TRAJECTORIES,
)


PROJECT_DATA_LICENSE = (
    "PolyForm-Noncommercial-1.0.0-or-commercial-license"
)
GROUND_UP_V3_CURRICULUM_ID = "omni-local-capability-curriculum-3"
GROUND_UP_V3_CURRICULUM_SHA256 = (
    "4ac61ceb3f207e41e8858b964473052b87c85376627fa5d3880f9ee2481e13a3"
)
GROUND_UP_V3_TRAINING_PROTOCOL_ID = "declared-tool-action-only-v2"

GROUND_UP_CURRICULUM_ID = GROUND_UP_V3_CURRICULUM_ID


class _ImmutableRecord(dict):
    """JSON-serializable dictionary that cannot drift after module import."""

    __slots__ = ()

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("verified curriculum records are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> "_ImmutableRecord":
        return self

    def __deepcopy__(self, _memo: Dict[int, Any]) -> "_ImmutableRecord":
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = dict.__new__(_ImmutableRecord)
        dict.__init__(
            frozen,
            ((str(key), _freeze(item)) for key, item in value.items()),
        )
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_clone(value: Any) -> Any:
    """Return ordinary mutable JSON values without exposing module constants."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


# The action data is copied once and guarded by a fixed digest.  A later edit to
# The source-data digest therefore fails loudly if the curriculum drifts.
GROUND_UP_ACTION_EXAMPLES: Tuple[Tuple[str, str], ...] = _freeze(
    NATIVE_ACTION_EXAMPLES
)

# Ground-up brains learn stable semantic capability IDs and symbolic argument
# examples; a visible desktop adapter resolves the native path/shell dialect at
# execution or held-out evaluation time, never as an optimizer input.
GROUND_UP_TOOL_TRAJECTORIES: Tuple[Mapping[str, Any], ...] = _freeze(
    [
        {
            "toolId": "system.files",
            "action": "list",
            "utterance": (
                "list every entry in the explicitly selected absolute folder, "
                "continuing through every page"
            ),
            "arguments": {"path": "<absolute-path>", "pageSize": 512},
            "outcome": "success",
            "result": {"entries": ["notes.md"], "hasMore": False},
        },
        {
            "toolId": "system.files",
            "action": "read",
            "utterance": "read the explicitly selected absolute file path",
            "arguments": {"path": "<absolute-path>"},
            "outcome": "success",
            "result": {"text": "visible evidence"},
        },
        {
            "toolId": "system.files",
            "action": "write",
            "utterance": (
                "write the explicitly quoted content to the explicitly selected "
                "absolute file path"
            ),
            "arguments": {
                "path": "<absolute-path>",
                "content": "<explicit-content>",
            },
            "outcome": "permission-denied",
            "result": {"wait": True},
        },
        {
            "toolId": "system.shell",
            "action": "run",
            "utterance": (
                "run the explicitly quoted native shell command in the explicitly "
                "selected absolute working directory"
            ),
            "arguments": {
                "command": "<explicit-command>",
                "cwd": "<absolute-path>",
            },
            "outcome": "success",
            "result": {"exitCode": 0},
        },
        {
            "toolId": "code.execute",
            "action": "run",
            "utterance": (
                "run the explicitly selected saved program with its declared "
                "local language toolchain"
            ),
            "arguments": {
                "language": "python",
                "entryPath": "<absolute-path>",
            },
            "outcome": "error",
            "result": {"exitCode": 1, "stderr": "assertion failed"},
        },
        *[
            dict(value)
            for value in NATIVE_TOOL_TRAJECTORIES
        ],
    ]
)
GROUND_UP_TOOL_NEGATIVE_EXAMPLES: Tuple[Mapping[str, Any], ...] = _freeze(
    [dict(value) for value in NATIVE_TOOL_NEGATIVE_EXAMPLES]
)


def _expected_action_kind(tool_id: str) -> str:
    if tool_id == "modality.imagine":
        return "imagine"
    if tool_id == "agent.fork":
        return "agent"
    if tool_id == "source.self-modify":
        return "evolve"
    return "tool"


# These are derived views of already-counted source records.  They may train an
# action head, but they do not add hidden utterances to the 68-record corpus.
GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES: Tuple[
    Mapping[str, Any], ...
] = _freeze(
    [
        {
            "utterance": trajectory["utterance"],
            "expectedKind": _expected_action_kind(str(trajectory["toolId"])),
            "sourceToolId": trajectory["toolId"],
            "sourceAction": trajectory["action"],
        }
        for trajectory in GROUND_UP_TOOL_TRAJECTORIES
    ]
    + [
        {
            "utterance": negative["utterance"],
            "expectedKind": "talk",
            "negativeReason": negative["reason"],
        }
        for negative in GROUND_UP_TOOL_NEGATIVE_EXAMPLES
    ]
)


GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION = (
    "unique-kind-rarest-token-triplet-v1"
)
GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION = (
    "unique-kind-novel-scalar-argument-token-v1"
)
_GROUND_UP_ACTION_KIND_ANCHOR_STOPWORDS = frozenset(
    """
    a an and are as at be before by can every for from has have i in into is it
    its me my of on or so than that the their them then there these they this
    through to up was we what when where which while who why will with you your
    explicitly selected absolute current declared active local visible
    """.split()
)


def _ground_up_anchor_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) >= 3
        and token not in _GROUND_UP_ACTION_KIND_ANCHOR_STOPWORDS
    )


def _ground_up_argument_value_tokens(value: Any) -> frozenset[str]:
    values: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            values.update(_ground_up_argument_value_tokens(item))
    elif isinstance(value, tuple):
        for item in value:
            values.update(_ground_up_argument_value_tokens(item))
    elif isinstance(value, str):
        values.update(_ground_up_anchor_tokens(value))
    return frozenset(values)


def _derive_ground_up_action_kind_anchor_views() -> Tuple[
    Mapping[str, Any], ...
]:
    """Derive auditable semantic anchors from declared records only.

    A token is eligible only when every occurrence across the complete action
    curriculum has the same action-kind label.  Each route then contributes
    its three rarest eligible tokens.  This strengthens generalization across
    concrete paths/commands without introducing a paraphrase, native value, or
    held-out readiness-probe string as an optimizer input.
    """

    kinds_by_token: Dict[str, set[str]] = {}
    frequency_by_token: Dict[str, int] = {}
    labelled_text = [
        (str(record["utterance"]), str(record["expectedKind"]))
        for record in GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES
    ] + [
        (str(text), str(kind)) for text, kind in GROUND_UP_ACTION_EXAMPLES
    ]
    for text, kind in labelled_text:
        for token in _ground_up_anchor_tokens(text):
            kinds_by_token.setdefault(token, set()).add(kind)
            frequency_by_token[token] = frequency_by_token.get(token, 0) + 1

    views = []
    for index, record in enumerate(GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES):
        kind = str(record["expectedKind"])
        eligible = [
            token
            for token in _ground_up_anchor_tokens(str(record["utterance"]))
            if kinds_by_token.get(token) == {kind}
        ]
        selected = sorted(
            eligible,
            key=lambda token: (
                frequency_by_token[token],
                -len(token),
                token,
            ),
        )[:3]
        if not selected:
            raise RuntimeError(
                "a declared action route has no unambiguous derived anchor"
            )
        views.append(
            {
                "utterance": " ".join(selected),
                "expectedKind": kind,
                "sourceRouteIndex": index,
                "sourceRecordSha256": _canonical_sha256(record),
                "derivation": GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION,
            }
        )
    return _freeze(views)


GROUND_UP_ACTION_KIND_ANCHOR_VIEWS = (
    _derive_ground_up_action_kind_anchor_views()
)


def _derive_ground_up_action_argument_anchor_views() -> Tuple[
    Mapping[str, Any], ...
]:
    """Expose novel scalar argument values as separately sealed anchors."""

    declared_text_tokens: set[str] = set()
    kinds_by_token: Dict[str, set[str]] = {}
    for record in GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES:
        kind = str(record["expectedKind"])
        tokens = _ground_up_anchor_tokens(str(record["utterance"]))
        declared_text_tokens.update(tokens)
        for token in tokens:
            kinds_by_token.setdefault(token, set()).add(kind)
    for text, kind_value in GROUND_UP_ACTION_EXAMPLES:
        kind = str(kind_value)
        tokens = _ground_up_anchor_tokens(str(text))
        declared_text_tokens.update(tokens)
        for token in tokens:
            kinds_by_token.setdefault(token, set()).add(kind)

    argument_tokens_by_route = []
    for index, trajectory in enumerate(GROUND_UP_TOOL_TRAJECTORIES):
        kind = str(
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES[index]["expectedKind"]
        )
        tokens = _ground_up_argument_value_tokens(
            trajectory.get("arguments", {})
        )
        argument_tokens_by_route.append(tokens)
        for token in tokens:
            kinds_by_token.setdefault(token, set()).add(kind)

    sources: Dict[Tuple[str, str], list[int]] = {}
    for index, tokens in enumerate(argument_tokens_by_route):
        kind = str(
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES[index]["expectedKind"]
        )
        for token in tokens:
            if (
                token not in declared_text_tokens
                and kinds_by_token.get(token) == {kind}
            ):
                sources.setdefault((token, kind), []).append(index)

    views = []
    for (token, kind), source_indices in sorted(sources.items()):
        if len(source_indices) != 1:
            raise RuntimeError(
                "a novel scalar argument anchor has ambiguous provenance"
            )
        source_index = source_indices[0]
        views.append(
            {
                "utterance": token,
                "expectedKind": kind,
                "sourceRouteIndices": source_indices,
                "sourceRecordSha256": _canonical_sha256(
                    GROUND_UP_TOOL_TRAJECTORIES[source_index]
                ),
                "derivation": GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION,
            }
        )
    return _freeze(views)


GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS = (
    _derive_ground_up_action_argument_anchor_views()
)

GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT = "\n".join(
    "%s %s %s %s"
    % (
        value["toolId"],
        value["action"],
        value["outcome"],
        value["utterance"],
    )
    for value in GROUND_UP_TOOL_TRAJECTORIES
)

# V3 intentionally admits no project-authored synthetic perception or selector
# examples.  The empty constants and their hashes are part of the manifest so
# adding one cannot happen silently in a later runtime edit.
GROUND_UP_MODALITY_TRAINING_FIXTURES: Tuple[Mapping[str, Any], ...] = ()
GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES: Tuple[
    Mapping[str, Any], ...
] = ()

# Symbolic, platform-neutral readiness probes.  Native commands and absolute
# paths may replace placeholders only in a no-gradient host evaluation.
GROUND_UP_READINESS_PROBE_FIXTURES: Tuple[Mapping[str, Any], ...] = _freeze(
    [
        {
            "id": "files",
            "utterance": "read the explicitly selected absolute file path",
            "expectedKind": "tool",
            "toolId": "system.files",
            "action": "read",
            "arguments": {"path": "<absolute-path>"},
        },
        {
            "id": "shell",
            "utterance": (
                "run the explicitly quoted native shell command in the explicitly "
                "selected absolute working directory"
            ),
            "expectedKind": "tool",
            "toolId": "system.shell",
            "action": "run",
            "arguments": {
                "command": "<explicit-command>",
                "cwd": "<absolute-path>",
            },
        },
        {
            "id": "web",
            "utterance": "search the web for current primary sources",
            "expectedKind": "tool",
            "toolId": "web.search",
            "action": "search",
            "arguments": {"query": "<explicit-query>"},
        },
        {
            "id": "imagination",
            "utterance": "make an image from this internal scene",
            "expectedKind": "imagine",
            "toolId": "modality.imagine",
            "action": "generate",
            "arguments": {"modality": "image"},
        },
        {
            "id": "agent",
            "utterance": "fork agents to investigate these independent parts",
            "expectedKind": "agent",
            "toolId": "agent.fork",
            "action": "start",
            "arguments": {"objective": "<explicit-objective>"},
        },
        {
            "id": "learn",
            "utterance": "learn this dataset into the neural substrate",
            "expectedKind": "learn",
            "toolId": None,
            "action": None,
            "arguments": {},
        },
        {
            "id": "evolve",
            "utterance": "create and evaluate an improvement candidate",
            "expectedKind": "evolve",
            "toolId": "source.self-modify",
            "action": "propose",
            "arguments": {
                "objective": "<explicit-objective>",
                "candidateKind": "substrate",
            },
        },
        {
            "id": "settings",
            "utterance": (
                "inspect which tools I can currently use before choosing an action"
            ),
            "expectedKind": "tool",
            "toolId": "studio.settings",
            "action": "inspect-access",
            "arguments": {},
        },
    ]
)


# Fixed digests turn accidental edits to an existing curriculum version into an
# import-time failure instead of silently assigning old provenance to new data.
GROUND_UP_ACTION_EXAMPLES_SHA256 = (
    "3fd2d943ec7d9d48cda4d2a79b9764f8aa941aadc85319ac1a4d8c244c177ec8"
)
GROUND_UP_TOOL_TRAJECTORIES_SHA256 = (
    "eb1eb1b217ea1086f4e8b0e6568518957e15ac8e2883cf2bc768d0b1ff3204c0"
)
GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256 = (
    "eb8ab55450b121c90c070da12a46432bdf319acc2cdf7096cb8cb2f98ba4c6d0"
)
GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256 = (
    "bed7523e4b9ff54f6a272152b6a8725a0b204f9d6a2dc0703968fea732a2c410"
)
GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256 = (
    "e4da8f7d50df8f2cd530fde4ac2b8d37c63aee73fe263a39cc1cd2701f5e9b9a"
)
GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256 = (
    "4ee76ce042f193f33c0830eda890961ae1d27119df7a597976cccdc0517f1252"
)
GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256 = (
    "40df078138f5e328ff94361daef41aaa68938afb8ad099f6422c76d7b1c28cf0"
)
GROUND_UP_EMPTY_FIXTURES_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256 = GROUND_UP_EMPTY_FIXTURES_SHA256
GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256 = (
    GROUND_UP_EMPTY_FIXTURES_SHA256
)


def _assert_frozen_record_hashes() -> None:
    expected = (
        (GROUND_UP_ACTION_EXAMPLES, GROUND_UP_ACTION_EXAMPLES_SHA256),
        (GROUND_UP_TOOL_TRAJECTORIES, GROUND_UP_TOOL_TRAJECTORIES_SHA256),
        (
            GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
            GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256,
        ),
        (
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES,
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256,
        ),
        (
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS,
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
        ),
        (
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS,
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
        ),
        (
            GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT,
            GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256,
        ),
        (
            GROUND_UP_MODALITY_TRAINING_FIXTURES,
            GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256,
        ),
        (
            GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES,
            GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256,
        ),
    )
    for value, digest in expected:
        if _canonical_sha256(value) != digest:
            raise RuntimeError(
                "a verified curriculum record changed without a version bump"
            )

_assert_frozen_record_hashes()


GROUND_UP_V3_DATASET_LEDGER: Tuple[Mapping[str, Any], ...] = _freeze(
    [
        {
            "id": "omni-local-action-examples-3",
            "kind": "structured-action",
            "records": len(GROUND_UP_ACTION_EXAMPLES),
            "sha256": GROUND_UP_ACTION_EXAMPLES_SHA256,
            "admitted": True,
            "license": PROJECT_DATA_LICENSE,
            "upstreamModel": None,
        },
        {
            "id": "omni-local-system-tool-trajectories-3",
            "kind": "structured-tool-trajectory",
            "records": len(GROUND_UP_TOOL_TRAJECTORIES),
            "sha256": GROUND_UP_TOOL_TRAJECTORIES_SHA256,
            "admitted": True,
            "license": PROJECT_DATA_LICENSE,
            "upstreamModel": None,
        },
        {
            "id": "omni-local-no-action-examples-3",
            "kind": "structured-no-action",
            "records": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "sha256": GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256,
            "admitted": True,
            "license": PROJECT_DATA_LICENSE,
            "upstreamModel": None,
        },
        {
            "id": "omni-local-synthetic-modality-fixtures-3",
            "kind": "synthetic-modality",
            "records": 0,
            "sha256": GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256,
            "admitted": False,
            "license": PROJECT_DATA_LICENSE,
            "upstreamModel": None,
        },
        {
            "id": "omni-local-imagination-selector-fixtures-3",
            "kind": "synthetic-imagination-selector",
            "records": 0,
            "sha256": GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256,
            "admitted": False,
            "license": PROJECT_DATA_LICENSE,
            "upstreamModel": None,
        },
    ]
)

GROUND_UP_V3_TRAINING_VIEW_LEDGER: Tuple[Mapping[str, Any], ...] = _freeze(
    [
        {
            "id": "omni-local-action-route-view-3",
            "kind": "derived-action-route-target",
            "records": len(GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES),
            "sha256": GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256,
            "addsSourceRecords": False,
            "platformSpecific": False,
        },
        {
            "id": "omni-local-whole-tool-view-3",
            "kind": "derived-whole-tool-text",
            "records": 1,
            "sha256": GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256,
            "addsSourceRecords": False,
            "platformSpecific": False,
        },
        {
            "id": "omni-local-discriminative-action-anchor-view-3",
            "kind": "derived-discriminative-action-anchor",
            "records": len(GROUND_UP_ACTION_KIND_ANCHOR_VIEWS),
            "sha256": GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
            "addsSourceRecords": False,
            "platformSpecific": False,
            "derivation": GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION,
        },
        {
            "id": "omni-local-scalar-argument-anchor-view-3",
            "kind": "derived-scalar-argument-anchor",
            "records": len(GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS),
            "sha256": GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
            "addsSourceRecords": False,
            "platformSpecific": False,
            "derivation": GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION,
        },
    ]
)

GROUND_UP_V3_DATASET_LEDGER_SHA256 = (
    "0932a4b02d21ecafeb9fbe8e89198f10fd91dc68ead7ab3e5632f306a22e60d4"
)
GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256 = (
    "34dfc1aa25698a18791bb50eb42e279aaf0db6d255bd1e90c846d582a20b064c"
)
GROUND_UP_READINESS_PROBE_FIXTURES_SHA256 = (
    "1ba3604a47d8764ffe5321c4f64fca77ee5f150fa327e383ba13a64e939c60a5"
)

for _ledger_value, _ledger_digest in (
    (GROUND_UP_V3_DATASET_LEDGER, GROUND_UP_V3_DATASET_LEDGER_SHA256),
    (
        GROUND_UP_V3_TRAINING_VIEW_LEDGER,
        GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256,
    ),
    (
        GROUND_UP_READINESS_PROBE_FIXTURES,
        GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
    ),
):
    if _canonical_sha256(_ledger_value) != _ledger_digest:
        raise RuntimeError("a verified OmniCortex v3 ledger changed without a version bump")

GROUND_UP_V3_SOURCE_RECORDS = (
    len(GROUND_UP_ACTION_EXAMPLES)
    + len(GROUND_UP_TOOL_TRAJECTORIES)
    + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
)


def _v3_receipt_contract_body() -> Dict[str, Any]:
    return {
        "format": "omni-ground-up-training-receipt",
        "formatVersion": 2,
        "transaction": "create-before-first-durable-checkpoint",
        "trainingProtocol": GROUND_UP_V3_TRAINING_PROTOCOL_ID,
        "recordsExpected": GROUND_UP_V3_SOURCE_RECORDS,
        "recordGroupsExpected": {
            "actionExamples": len(GROUND_UP_ACTION_EXAMPLES),
            "toolTrajectories": len(GROUND_UP_TOOL_TRAJECTORIES),
            "negativeToolExamples": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "syntheticModalityFixtures": 0,
            "imaginationSelectorFixtures": 0,
        },
        "datasetLedgerSha256": GROUND_UP_V3_DATASET_LEDGER_SHA256,
        "trainingViewLedgerSha256": GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256,
        "readinessProbeSha256": GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
        "readinessExpected": {
            "representativeProbes": len(GROUND_UP_READINESS_PROBE_FIXTURES),
            "toolTrajectoryProbes": len(GROUND_UP_TOOL_TRAJECTORIES),
            "negativeNoActionProbes": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "trainingEligible": False,
        },
        "syntheticModalityTraining": False,
        "imaginationSelectorTraining": False,
        "platformSpecificTrainingExamples": False,
        "substrateMutationRequired": True,
        "originKind": "ground-up",
        "pretrainedWeights": False,
        "externalFoundation": None,
        "externalWeightFiles": [],
        "baseFrozen": False,
        "adapterOnly": False,
        "hiddenPrompt": False,
        "apiTeacher": False,
        "rewardModel": False,
        "rlhf": False,
        "initializationChecksumBinding": "parameterChecksumBefore",
        "requiredChecksumBindings": [
            "randomInitialization.parameterChecksum=trainingReceipt.parameterChecksumBefore",
            "trainingReceipt.parameterChecksumAfter=current.parameterChecksum",
            "trainingReceipt.modalityParameterChecksumBefore=trainingReceipt.modalityParameterChecksumAfter",
            "trainingReceipt.imaginationSelectorChecksumBefore=trainingReceipt.imaginationSelectorChecksumAfter",
            "packedTernary.parameterChecksum=trainingReceipt.parameterChecksumAfter",
            "packedTernary.curriculumSha256=curriculum.sha256",
            "packedTernary.trainingManifestSha256=groundUpTrainingManifest.contentSha256",
        ],
    }


GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256 = (
    "d77c78a9ee47793ad8e5443791f98b76b6ade8742c4f2501d9a2fe50c1c886bc"
)
if (
    _canonical_sha256(_v3_receipt_contract_body())
    != GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256
):
    raise RuntimeError("the verified OmniCortex v3 receipt contract changed")




def _v3_manifest_body() -> Dict[str, Any]:
    return {
        "id": GROUND_UP_V3_CURRICULUM_ID,
        "format": "omni-ground-up-curriculum-manifest",
        "formatVersion": 3,
        "source": "bundled project-authored local tool/action curriculum",
        "sourceKind": "ordinary-training-examples",
        "license": PROJECT_DATA_LICENSE,
        "externalFoundation": None,
        "pretrainedWeights": False,
        "upstreamModel": None,
        "hiddenPrompt": False,
        "apiTeacher": False,
        "languageCorpusRecords": 0,
        "dialogueAnswerRecords": 0,
        "actionExamples": len(GROUND_UP_ACTION_EXAMPLES),
        "toolTrajectories": len(GROUND_UP_TOOL_TRAJECTORIES),
        "negativeToolExamples": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
        "syntheticModalityFixtures": 0,
        "imaginationSelectorFixtures": 0,
        "sourceRecords": GROUND_UP_V3_SOURCE_RECORDS,
        "trainingProtocol": {
            "id": GROUND_UP_V3_TRAINING_PROTOCOL_ID,
            "sourcePhases": [
                "bundled-declared-tool-action-curriculum",
                "explicitly-selected-user-data-after-origin",
            ],
            "derivedViewsAddSourceRecords": False,
            "syntheticModalityTraining": False,
            "imaginationSelectorTraining": False,
            "platformSpecificOptimizerInputs": False,
            "readinessProbesAreOptimizerInputs": False,
            "declaredViewStoppingRule": (
                "declared-only-convergence-no-readiness-selection"
            ),
            "languageHeadFrozenDuringRouteRehearsal": True,
            "textToolOptimizerScope": [
                "decoder-language",
                "memory-bridge",
                "idea-adapter",
                "liquid",
            ],
            "textToolExcludedParameterGroups": [
                "modalities",
                "imagination-selector",
            ],
        },
        "systemCapabilityContract": {
            "files": "system.files",
            "shell": "system.shell",
            "pathArguments": "explicit-host-absolute-path",
            "shellDialect": "resolved-by-visible-host-adapter",
            "platformSpecificTrainingExamples": False,
        },
        "datasetLedgerSha256": GROUND_UP_V3_DATASET_LEDGER_SHA256,
        "datasetLedger": GROUND_UP_V3_DATASET_LEDGER,
        "trainingViewLedgerSha256": GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256,
        "trainingViewLedger": GROUND_UP_V3_TRAINING_VIEW_LEDGER,
        "readinessProbeContract": {
            "records": len(GROUND_UP_READINESS_PROBE_FIXTURES),
            "sha256": GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
            "trainingEligible": False,
            "nativeValuesResolvedByVisibleHostAdapter": True,
        },
        "trainingReceiptContract": {
            "format": "omni-ground-up-training-receipt",
            "formatVersion": 2,
            "sha256": GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256,
        },
    }


def _v3_manifest() -> Dict[str, Any]:
    body = _v3_manifest_body()
    identity = _canonical_sha256(body)
    if identity != GROUND_UP_V3_CURRICULUM_SHA256:
        raise RuntimeError("the verified OmniCortex v3 manifest changed")
    return {**body, "sha256": identity}


_GROUND_UP_V3_MANIFEST = _freeze(_v3_manifest())
GROUND_UP_CURRICULUM_MANIFEST_SHA256S = frozenset(
    {GROUND_UP_V3_CURRICULUM_SHA256}
)


def ground_up_curriculum_manifest() -> Dict[str, Any]:
    """Return the current v3 manifest for every future new Build."""

    return _json_clone(_GROUND_UP_V3_MANIFEST)




def current_ground_up_curriculum_manifest() -> Dict[str, Any]:
    """Return the v3 contract required for every future new Build."""

    return _json_clone(_GROUND_UP_V3_MANIFEST)


def ground_up_v3_curriculum_manifest() -> Dict[str, Any]:
    """Explicit versioned spelling for the current v3 manifest."""

    return current_ground_up_curriculum_manifest()


def resolve_ground_up_curriculum_manifest(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Resolve an exact supported static manifest from a hash or manifest.

    A dynamic training manifest may contain additional receipt fields. Every
    static field of the current curriculum must still be identical.
    """

    manifests = (_GROUND_UP_V3_MANIFEST,)
    if isinstance(value, str):
        for manifest in manifests:
            if value == manifest["sha256"]:
                return _json_clone(manifest)
        return None
    if not isinstance(value, Mapping):
        return None
    for manifest in manifests:
        expected_manifest = _json_clone(manifest)
        if value.get("sha256") != expected_manifest["sha256"]:
            continue
        if all(
            value.get(key) == expected
            for key, expected in expected_manifest.items()
        ):
            return expected_manifest
    return None


def current_ground_up_training_receipt_contract() -> Dict[str, Any]:
    """Return the static v3 fields a runtime receipt must include verbatim."""

    curriculum = current_ground_up_curriculum_manifest()
    return {
        **_v3_receipt_contract_body(),
        "contractSha256": GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256,
        "curriculumId": curriculum["id"],
        "curriculumSha256": curriculum["sha256"],
    }


def seal_ground_up_v3_training_receipt(
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    """Add a canonical content hash without mutating the caller's receipt."""

    body = _json_clone(receipt)
    body.pop("contentSha256", None)
    return {**body, "contentSha256": _canonical_sha256(body)}


def validate_ground_up_v3_training_receipt(
    receipt: Any,
) -> Dict[str, Any]:
    """Validate the self-contained portion of a v3 training receipt.

    The runtime must additionally bind the receipt's checksums and tensor names
    to the live model, origin, and packed shards.  This pure validator makes the
    versioned integration boundary testable without constructing a brain.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("OmniCortex v3 training receipt must be an object")
    expected = current_ground_up_training_receipt_contract()
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("OmniCortex v3 training receipt contract does not match")

    def sha(field: str) -> str:
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("OmniCortex v3 training receipt checksum is invalid")
        return value

    if (
        receipt.get("recordsVisited") != GROUND_UP_V3_SOURCE_RECORDS
        or receipt.get("completeCoverage") is not True
        or receipt.get("parametersChanged") is not True
        or receipt.get("substrateChanged") is not True
        or receipt.get("modalityParametersChanged") is not False
        or receipt.get("imaginationSelectorParametersChanged") is not False
    ):
        raise ValueError("OmniCortex v3 training receipt coverage is invalid")
    before = sha("parameterChecksumBefore")
    after = sha("parameterChecksumAfter")
    modality_before = sha("modalityParameterChecksumBefore")
    modality_after = sha("modalityParameterChecksumAfter")
    selector_before = sha("imaginationSelectorChecksumBefore")
    selector_after = sha("imaginationSelectorChecksumAfter")
    if (
        before == after
        or modality_before != modality_after
        or selector_before != selector_after
    ):
        raise ValueError("OmniCortex v3 training receipt mutation proof is invalid")
    substrate_before = receipt.get("substrateBefore")
    substrate_after = receipt.get("substrateAfter")
    substrate_keys = {"neurons", "assemblies", "synapses"}
    if (
        not isinstance(substrate_before, Mapping)
        or not isinstance(substrate_after, Mapping)
        or set(substrate_before) != substrate_keys
        or set(substrate_after) != substrate_keys
        or any(
            not isinstance(substrate_before[key], int)
            or isinstance(substrate_before[key], bool)
            or int(substrate_before[key]) < 0
            or not isinstance(substrate_after[key], int)
            or isinstance(substrate_after[key], bool)
            or int(substrate_after[key]) < int(substrate_before[key])
            for key in substrate_keys
        )
        or not any(
            int(substrate_after[key]) > int(substrate_before[key])
            for key in substrate_keys
        )
    ):
        raise ValueError("OmniCortex v3 training receipt substrate proof is invalid")
    changed = receipt.get("changedNativeCoreTensorNames")
    if (
        not isinstance(changed, list)
        or not changed
        or any(
            not isinstance(name, str) or name.startswith("modalities.")
            for name in changed
        )
        or len(set(changed)) != len(changed)
        or not isinstance(
            receipt.get("changedNativeCoreParameterTensors"), int
        )
        or isinstance(
            receipt.get("changedNativeCoreParameterTensors"), bool
        )
        or receipt.get("changedNativeCoreParameterTensors") != len(changed)
        or not isinstance(receipt.get("nativeCoreParameterTensors"), int)
        or isinstance(receipt.get("nativeCoreParameterTensors"), bool)
        or int(receipt["nativeCoreParameterTensors"]) < len(changed)
    ):
        raise ValueError("OmniCortex v3 training receipt tensor inventory is invalid")
    content_sha = sha("contentSha256")
    body = _json_clone(receipt)
    body.pop("contentSha256", None)
    if content_sha != _canonical_sha256(body):
        raise ValueError("OmniCortex v3 training receipt content checksum failed")
    return _json_clone(receipt)


def seal_ground_up_v3_training_manifest(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Seal the complete dynamic Build manifest after its receipt is final."""

    body = _json_clone(manifest)
    body.pop("contentSha256", None)
    resolved = resolve_ground_up_curriculum_manifest(body)
    if (
        resolved is None
        or resolved.get("formatVersion") != 3
        or not isinstance(body.get("trainingReceipt"), Mapping)
    ):
        raise ValueError("OmniCortex v3 training manifest contract does not match")
    validate_ground_up_v3_training_receipt(body["trainingReceipt"])
    return {**body, "contentSha256": _canonical_sha256(body)}


def validate_ground_up_v3_training_manifest(
    manifest: Any,
) -> Dict[str, Any]:
    """Validate the static identity, nested receipt, and dynamic content seal."""

    if not isinstance(manifest, Mapping):
        raise ValueError("OmniCortex v3 training manifest must be an object")
    content_sha = manifest.get("contentSha256")
    if (
        not isinstance(content_sha, str)
        or len(content_sha) != 64
        or any(character not in "0123456789abcdef" for character in content_sha)
    ):
        raise ValueError("OmniCortex v3 training manifest checksum is invalid")
    expected = seal_ground_up_v3_training_manifest(manifest)
    if expected["contentSha256"] != content_sha:
        raise ValueError("OmniCortex v3 training manifest content checksum failed")
    return _json_clone(manifest)


__all__ = [
    "GROUND_UP_ACTION_EXAMPLES",
    "GROUND_UP_ACTION_EXAMPLES_SHA256",
    "GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION",
    "GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS",
    "GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256",
    "GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION",
    "GROUND_UP_ACTION_KIND_ANCHOR_VIEWS",
    "GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256",
    "GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES",
    "GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256",
    "GROUND_UP_CURRICULUM_ID",
    "GROUND_UP_CURRICULUM_MANIFEST_SHA256S",
    "GROUND_UP_EMPTY_FIXTURES_SHA256",
    "GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES",
    "GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256",
    "GROUND_UP_MODALITY_TRAINING_FIXTURES",
    "GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256",
    "GROUND_UP_READINESS_PROBE_FIXTURES",
    "GROUND_UP_READINESS_PROBE_FIXTURES_SHA256",
    "GROUND_UP_TOOL_NEGATIVE_EXAMPLES",
    "GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256",
    "GROUND_UP_TOOL_TRAJECTORIES",
    "GROUND_UP_TOOL_TRAJECTORIES_SHA256",
    "GROUND_UP_V3_CURRICULUM_ID",
    "GROUND_UP_V3_CURRICULUM_SHA256",
    "GROUND_UP_V3_DATASET_LEDGER",
    "GROUND_UP_V3_DATASET_LEDGER_SHA256",
    "GROUND_UP_V3_SOURCE_RECORDS",
    "GROUND_UP_V3_TRAINING_PROTOCOL_ID",
    "GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256",
    "GROUND_UP_V3_TRAINING_VIEW_LEDGER",
    "GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256",
    "GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256",
    "GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT",
    "current_ground_up_curriculum_manifest",
    "current_ground_up_training_receipt_contract",
    "ground_up_curriculum_manifest",
    "ground_up_v3_curriculum_manifest",
    "resolve_ground_up_curriculum_manifest",
    "seal_ground_up_v3_training_manifest",
    "seal_ground_up_v3_training_receipt",
    "validate_ground_up_v3_training_manifest",
    "validate_ground_up_v3_training_receipt",
]
