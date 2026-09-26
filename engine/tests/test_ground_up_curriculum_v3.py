import ast
import hashlib
import inspect
import json
import re
import textwrap
import unittest
from unittest.mock import patch

from omni_core import capability_rehearsal
from omni_core.brain import AdaptiveBrain

from omni_core.ground_up import (
    GROUND_UP_ACTION_EXAMPLES,
    GROUND_UP_ACTION_EXAMPLES_SHA256,
    GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION,
    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS,
    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
    GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION,
    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS,
    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
    GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES,
    GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256,
    GROUND_UP_CURRICULUM_MANIFEST_SHA256S,
    GROUND_UP_EMPTY_FIXTURES_SHA256,
    GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES,
    GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256,
    GROUND_UP_MODALITY_TRAINING_FIXTURES,
    GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256,
    GROUND_UP_READINESS_PROBE_FIXTURES,
    GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256,
    GROUND_UP_TOOL_TRAJECTORIES,
    GROUND_UP_TOOL_TRAJECTORIES_SHA256,
    GROUND_UP_V3_CURRICULUM_SHA256,
    GROUND_UP_V3_DATASET_LEDGER_SHA256,
    GROUND_UP_V3_SOURCE_RECORDS,
    GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256,
    GROUND_UP_V3_TRAINING_PROTOCOL_ID,
    GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256,
    GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256,
    GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT,
    current_ground_up_curriculum_manifest,
    current_ground_up_training_receipt_contract,
    ground_up_curriculum_manifest,
    ground_up_v3_curriculum_manifest,
    resolve_ground_up_curriculum_manifest,
    seal_ground_up_v3_training_manifest,
    seal_ground_up_v3_training_receipt,
    validate_ground_up_v3_training_manifest,
    validate_ground_up_v3_training_receipt,
)


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class GroundUpCurriculumV3Tests(unittest.TestCase):
    def test_only_current_v3_curriculum_resolves(self):
        current = current_ground_up_curriculum_manifest()

        self.assertEqual(ground_up_curriculum_manifest(), current)
        self.assertEqual(current["formatVersion"], 3)
        self.assertEqual(ground_up_v3_curriculum_manifest(), current)
        self.assertEqual(current["sha256"], GROUND_UP_V3_CURRICULUM_SHA256)
        self.assertEqual(
            GROUND_UP_CURRICULUM_MANIFEST_SHA256S,
            {GROUND_UP_V3_CURRICULUM_SHA256},
        )
        self.assertEqual(
            resolve_ground_up_curriculum_manifest(current), current
        )
        self.assertEqual(
            resolve_ground_up_curriculum_manifest(current["sha256"]),
            current,
        )

        self.assertIsNone(
            resolve_ground_up_curriculum_manifest({"formatVersion": 2})
        )
        tampered = {**current, "sourceRecords": current["sourceRecords"] - 1}
        self.assertIsNone(resolve_ground_up_curriculum_manifest(tampered))
        self.assertIsNone(resolve_ground_up_curriculum_manifest("0" * 64))

    def test_v3_hashes_the_complete_split_source_and_derived_view_ledgers(self):
        manifest = current_ground_up_curriculum_manifest()
        body = dict(manifest)
        body.pop("sha256")

        self.assertEqual(canonical_sha256(body), manifest["sha256"])
        self.assertEqual(
            canonical_sha256(manifest["datasetLedger"]),
            GROUND_UP_V3_DATASET_LEDGER_SHA256,
        )
        self.assertEqual(
            canonical_sha256(manifest["trainingViewLedger"]),
            GROUND_UP_V3_TRAINING_VIEW_LEDGER_SHA256,
        )
        self.assertEqual(len(GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES), 33)
        self.assertEqual(
            [entry["records"] for entry in manifest["trainingViewLedger"]],
            [33, 1, 33, 1],
        )
        self.assertEqual(
            manifest["trainingProtocol"]["id"],
            GROUND_UP_V3_TRAINING_PROTOCOL_ID,
        )
        self.assertEqual(
            manifest["trainingProtocol"]["declaredViewStoppingRule"],
            "declared-only-convergence-no-readiness-selection",
        )
        self.assertTrue(
            manifest["trainingProtocol"][
                "languageHeadFrozenDuringRouteRehearsal"
            ]
        )
        self.assertEqual(
            manifest["trainingProtocol"]["textToolOptimizerScope"],
            [
                "decoder-language",
                "memory-bridge",
                "idea-adapter",
                "liquid",
            ],
        )
        self.assertEqual(
            manifest["trainingProtocol"][
                "textToolExcludedParameterGroups"
            ],
            ["modalities", "imagination-selector"],
        )
        anchor_ledger = manifest["trainingViewLedger"][-2]
        self.assertEqual(
            anchor_ledger["sha256"],
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
        )
        self.assertEqual(
            anchor_ledger["derivation"],
            GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION,
        )
        self.assertFalse(anchor_ledger["addsSourceRecords"])
        self.assertFalse(anchor_ledger["platformSpecific"])
        argument_ledger = manifest["trainingViewLedger"][-1]
        self.assertEqual(
            argument_ledger["sha256"],
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
        )
        self.assertEqual(
            argument_ledger["derivation"],
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION,
        )
        self.assertFalse(argument_ledger["addsSourceRecords"])
        self.assertFalse(argument_ledger["platformSpecific"])
        self.assertEqual(manifest["sourceRecords"], 68)
        self.assertEqual(GROUND_UP_V3_SOURCE_RECORDS, 68)
        self.assertEqual(
            sum(entry["records"] for entry in manifest["datasetLedger"]),
            68,
        )
        self.assertEqual(
            sum(
                entry["records"]
                for entry in manifest["datasetLedger"]
                if entry["admitted"]
            ),
            68,
        )
        self.assertEqual(
            [entry["kind"] for entry in manifest["datasetLedger"][:3]],
            [
                "structured-action",
                "structured-tool-trajectory",
                "structured-no-action",
            ],
        )
        self.assertEqual(
            [entry["records"] for entry in manifest["datasetLedger"][:3]],
            [35, 27, 6],
        )
        self.assertEqual(
            [entry["sha256"] for entry in manifest["datasetLedger"][:3]],
            [
                GROUND_UP_ACTION_EXAMPLES_SHA256,
                GROUND_UP_TOOL_TRAJECTORIES_SHA256,
                GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256,
            ],
        )

    def test_every_optimizer_input_is_immutable_and_platform_neutral(self):
        with self.assertRaises(TypeError):
            GROUND_UP_TOOL_TRAJECTORIES[0]["action"] = "changed"
        with self.assertRaises(TypeError):
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES[0][
                "expectedKind"
            ] = "changed"
        with self.assertRaises(TypeError):
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS[0]["utterance"] = "changed"

        values = {
            "actions": GROUND_UP_ACTION_EXAMPLES,
            "tools": GROUND_UP_TOOL_TRAJECTORIES,
            "negative": GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
            "routes": GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES,
            "anchors": GROUND_UP_ACTION_KIND_ANCHOR_VIEWS,
            "argumentAnchors": GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS,
            "wholeToolView": GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT,
        }
        encoded = json.dumps(values, sort_keys=True).lower()
        for forbidden in (
            "windows.",
            "powershell",
            "get-childitem",
            "c:\\",
            "/tmp",
            "/workspace",
        ):
            self.assertNotIn(forbidden, encoded)

        self.assertEqual(
            canonical_sha256(GROUND_UP_ACTION_EXAMPLES),
            GROUND_UP_ACTION_EXAMPLES_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_TOOL_TRAJECTORIES),
            GROUND_UP_TOOL_TRAJECTORIES_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            GROUND_UP_TOOL_NEGATIVE_EXAMPLES_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES),
            GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_ACTION_KIND_ANCHOR_VIEWS),
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS),
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_WHOLE_TOOL_CURRICULUM_TEXT),
            GROUND_UP_WHOLE_TOOL_CURRICULUM_SHA256,
        )

    def test_action_kind_anchors_are_derived_only_from_declared_records(self):
        self.assertEqual(
            len(GROUND_UP_ACTION_KIND_ANCHOR_VIEWS),
            len(GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES),
        )
        readiness_text = {
            str(record["utterance"]).casefold()
            for record in GROUND_UP_READINESS_PROBE_FIXTURES
        }
        for expected_index, anchor in enumerate(
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS
        ):
            with self.subTest(index=expected_index):
                source = GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES[
                    expected_index
                ]
                self.assertEqual(anchor["sourceRouteIndex"], expected_index)
                self.assertEqual(
                    anchor["sourceRecordSha256"],
                    canonical_sha256(source),
                )
                self.assertEqual(
                    anchor["expectedKind"], source["expectedKind"]
                )
                self.assertEqual(
                    anchor["derivation"],
                    GROUND_UP_ACTION_KIND_ANCHOR_DERIVATION,
                )
                source_tokens = set(
                    re.findall(
                        r"[a-z0-9]+",
                        str(source["utterance"]).casefold(),
                    )
                )
                anchor_tokens = str(anchor["utterance"]).split()
                self.assertGreater(len(anchor_tokens), 0)
                self.assertLessEqual(len(anchor_tokens), 3)
                self.assertTrue(set(anchor_tokens).issubset(source_tokens))
                self.assertNotIn(
                    str(anchor["utterance"]).casefold(), readiness_text
                )

    def test_argument_anchor_is_a_novel_declared_scalar_not_probe_text(self):
        self.assertEqual(len(GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS), 1)
        anchor = GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS[0]
        self.assertEqual(anchor["utterance"], "python")
        self.assertEqual(anchor["expectedKind"], "tool")
        self.assertEqual(anchor["sourceRouteIndices"], (4,))
        self.assertEqual(
            anchor["sourceRecordSha256"],
            canonical_sha256(GROUND_UP_TOOL_TRAJECTORIES[4]),
        )
        self.assertEqual(
            anchor["derivation"],
            GROUND_UP_ACTION_ARGUMENT_ANCHOR_DERIVATION,
        )
        declared_utterances = " ".join(
            str(record["utterance"])
            for record in GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES
        ).casefold()
        self.assertNotIn("python", declared_utterances)
        self.assertIn(
            "python",
            json.dumps(
                GROUND_UP_TOOL_TRAJECTORIES[4]["arguments"],
                sort_keys=True,
            ).casefold(),
        )
        self.assertTrue(
            all(
                "python" not in str(record["utterance"]).casefold()
                for record in GROUND_UP_READINESS_PROBE_FIXTURES
            )
        )

    def test_v3_explicitly_admits_no_synthetic_media_or_selector_fixture(self):
        manifest = current_ground_up_curriculum_manifest()

        self.assertEqual(GROUND_UP_MODALITY_TRAINING_FIXTURES, ())
        self.assertEqual(
            GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES, ()
        )
        self.assertEqual(
            canonical_sha256(GROUND_UP_MODALITY_TRAINING_FIXTURES),
            GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256,
        )
        self.assertEqual(
            canonical_sha256(
                GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES
            ),
            GROUND_UP_IMAGINATION_SELECTOR_TRAINING_FIXTURES_SHA256,
        )
        self.assertEqual(
            GROUND_UP_MODALITY_TRAINING_FIXTURES_SHA256,
            GROUND_UP_EMPTY_FIXTURES_SHA256,
        )
        self.assertEqual(manifest["syntheticModalityFixtures"], 0)
        self.assertEqual(manifest["imaginationSelectorFixtures"], 0)
        self.assertFalse(
            manifest["trainingProtocol"]["syntheticModalityTraining"]
        )
        self.assertFalse(
            manifest["trainingProtocol"]["imaginationSelectorTraining"]
        )
        excluded = manifest["datasetLedger"][3:]
        self.assertEqual([entry["records"] for entry in excluded], [0, 0])
        self.assertTrue(all(entry["admitted"] is False for entry in excluded))

    def test_readiness_probes_are_hashed_but_never_training_records(self):
        manifest = current_ground_up_curriculum_manifest()
        probes = manifest["readinessProbeContract"]

        self.assertEqual(len(GROUND_UP_READINESS_PROBE_FIXTURES), 8)
        self.assertEqual(
            canonical_sha256(GROUND_UP_READINESS_PROBE_FIXTURES),
            GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
        )
        self.assertEqual(probes["records"], 8)
        self.assertEqual(
            probes["sha256"], GROUND_UP_READINESS_PROBE_FIXTURES_SHA256
        )
        self.assertFalse(probes["trainingEligible"])
        self.assertFalse(
            manifest["trainingProtocol"]["readinessProbesAreOptimizerInputs"]
        )

    def test_v3_receipt_contract_is_hash_bound_and_rejects_hidden_mutation(self):
        contract = current_ground_up_training_receipt_contract()
        manifest = current_ground_up_curriculum_manifest()
        self.assertEqual(contract["formatVersion"], 2)
        self.assertEqual(contract["recordsExpected"], 68)
        self.assertEqual(contract["curriculumSha256"], manifest["sha256"])
        self.assertEqual(
            contract["contractSha256"],
            GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256,
        )
        self.assertEqual(
            manifest["trainingReceiptContract"]["sha256"],
            GROUND_UP_V3_TRAINING_RECEIPT_CONTRACT_SHA256,
        )

        receipt = seal_ground_up_v3_training_receipt(
            {
                **contract,
                "recordsVisited": 68,
                "completeCoverage": True,
                "parametersChanged": True,
                "substrateChanged": True,
                "substrateBefore": {
                    "neurons": 0,
                    "assemblies": 0,
                    "synapses": 0,
                },
                "substrateAfter": {
                    "neurons": 3,
                    "assemblies": 2,
                    "synapses": 1,
                },
                "modalityParametersChanged": False,
                "imaginationSelectorParametersChanged": False,
                "parameterChecksumBefore": "1" * 64,
                "parameterChecksumAfter": "2" * 64,
                "modalityParameterChecksumBefore": "3" * 64,
                "modalityParameterChecksumAfter": "3" * 64,
                "imaginationSelectorChecksumBefore": "4" * 64,
                "imaginationSelectorChecksumAfter": "4" * 64,
                "nativeCoreParameterTensors": 2,
                "changedNativeCoreParameterTensors": 1,
                "changedNativeCoreTensorNames": ["decoder.output.weight"],
            }
        )
        self.assertEqual(validate_ground_up_v3_training_receipt(receipt), receipt)
        dynamic_manifest = seal_ground_up_v3_training_manifest(
            {
                **manifest,
                "trainedAt": "2026-09-07T00:00:00Z",
                "trainingReceipt": receipt,
            }
        )
        self.assertEqual(
            validate_ground_up_v3_training_manifest(dynamic_manifest),
            dynamic_manifest,
        )
        tampered_manifest = {
            **dynamic_manifest,
            "trainedAt": "2099-01-01T00:00:00Z",
        }
        with self.assertRaisesRegex(ValueError, "content checksum"):
            validate_ground_up_v3_training_manifest(tampered_manifest)

        for field, value in (
            ("recordsVisited", 67),
            ("syntheticModalityTraining", True),
            ("modalityParametersChanged", True),
            ("modalityParameterChecksumAfter", "5" * 64),
            ("changedNativeCoreTensorNames", ["modalities.image.weight"]),
            ("changedNativeCoreTensorNames", [{}]),
        ):
            with self.subTest(field=field):
                tampered = seal_ground_up_v3_training_receipt(
                    {**receipt, field: value}
                )
                with self.assertRaises(ValueError):
                    validate_ground_up_v3_training_receipt(tampered)

        checksum_tamper = {**receipt, "parameterChecksumAfter": "6" * 64}
        with self.assertRaisesRegex(ValueError, "content checksum"):
            validate_ground_up_v3_training_receipt(checksum_tamper)

    def test_v3_public_rehearsal_selects_declared_training_only(self):
        brain = object()
        report = {
            "probe": {"passed": True},
            "steps": 1,
        }
        with patch.object(
            capability_rehearsal,
            "_rehearse_probe_action_routes",
            return_value=report,
        ) as rehearse:
            v3 = capability_rehearsal.rehearse_public_capability_routes(
                brain, curriculum_version=3
            )
            rehearse.assert_called_once_with(
                brain,
                maximum_steps=192,
                minimum_steps=1,
                declared_training_only=True,
            )

        self.assertFalse(v3["readinessProbesAreOptimizerInputs"])
        self.assertEqual(
            v3["symbolicReadinessProbeSha256"],
            GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
        )


    def test_future_builder_uses_v3_and_no_synthetic_modality_trainer(self):
        tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(AdaptiveBrain._train_ground_up_curriculum)
            )
        )
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("_train_starter_modalities", calls)
        self.assertIn("_clear_ground_up_transient_state", calls)
        self.assertIn("current_ground_up_curriculum_manifest", names)
        rehearsal = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rehearse_public_capability_routes"
        )
        keywords = {item.arg: item.value for item in rehearsal.keywords}
        self.assertEqual(ast.literal_eval(keywords["curriculum_version"]), 3)

    def test_text_learning_never_applies_global_stability_to_modalities(self):
        for method in (
            AdaptiveBrain._optimize_experience,
            AdaptiveBrain._optimize_dialogue_pair,
            AdaptiveBrain._latent_rehearsal_step,
        ):
            with self.subTest(method=method.__name__):
                tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
                scoped_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {
                        "_stability_penalty",
                        "_accumulate_slow_importance",
                        "_commit_slow_anchors",
                    }
                ]
                self.assertTrue(scoped_calls)
                for call in scoped_calls:
                    self.assertTrue(
                        call.args
                        or any(
                            keyword.arg == "parameters"
                            for keyword in call.keywords
                        ),
                        "%s has an unscoped %s call"
                        % (method.__name__, call.func.attr),
                    )


if __name__ == "__main__":
    unittest.main()
