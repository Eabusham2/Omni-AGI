"""New diagnostics/publication contracts; no model construction or execution."""
import unittest
from pathlib import Path
from types import SimpleNamespace

from omni_core.derived_inference_sidecar import current_sidecar_descriptor, record_sidecar_validation, sidecar_status


def owner():
    return SimpleNamespace(mutable_state_manifest={"activeGeneration": "mutable"},
        memory=SimpleNamespace(persistence_manifest={"activeGeneration": "graph"}), _connected_graph_states=None,
        packed_ternary_manifest={"contentSha256": "packed", "parameterChecksum": "parameters"})


class DerivedInferenceSidecarTests(unittest.TestCase):
    def test_stale_verified_copy_is_withheld_without_writing_or_repacking(self):
        brain = owner()
        status = record_sidecar_validation(brain, manifest_sha256="packed", parameter_checksum="parameters", refresh_required=True, reason="order-changed")
        self.assertTrue(status["existingSidecarFullyVerified"])
        self.assertFalse(status["runtimeConsumesSidecar"])
        self.assertTrue(status["explicitExportMustRepack"])
        self.assertIsNone(current_sidecar_descriptor(brain))
        self.assertIsNone(brain.packed_ternary_manifest)
        self.assertEqual(brain.mutable_state_manifest, {"activeGeneration": "mutable"})

    def test_checked_descriptor_is_generation_bound_and_returned_metadata_is_detached(self):
        brain = owner()
        record_sidecar_validation(brain, manifest_sha256="packed", parameter_checksum="parameters", refresh_required=False, reason="explicit-export")
        current = current_sidecar_descriptor(brain)
        self.assertEqual(current["contentSha256"], "packed")
        current["contentSha256"] = "modified-by-caller"
        self.assertEqual(brain.packed_ternary_manifest["contentSha256"], "packed")
        brain.memory.persistence_manifest = {"activeGeneration": "later-graph"}
        self.assertTrue(sidecar_status(brain)["refreshRequired"])
        self.assertIsNone(current_sidecar_descriptor(brain))

    def test_unknown_copy_is_not_advertised_current(self):
        brain = owner()
        self.assertTrue(sidecar_status(brain)["refreshRequired"])
        self.assertIsNone(current_sidecar_descriptor(brain))

    def test_actual_load_defers_only_stale_derived_copy_after_full_existing_verification(self):
        source = (Path(__file__).parents[1] / "omni_core" / "brain.py").read_text()
        begin = source.index("packed_refresh_required = False")
        end = source.index("for recovered in recovered_candidates:", begin)
        self.assertIn("verify_ternary_shards(packed_path, retain_names=())", source[begin:end])
        stale = source[source.index("elif packed_refresh_required:", begin):source.index("for recovered in recovered_candidates:", begin)]
        self.assertNotIn("brain.export_packed_ternary()", stale)
        self.assertNotIn("atomic_write_json(", stale)
        self.assertIn("refresh_required=True", stale)
        checkpoint = source[source.index("    def checkpoint("):source.index("    def snapshot(")]
        self.assertLess(checkpoint.index("self.save()"), checkpoint.index("self.export_packed_ternary()"))
        self.assertIn("require_disk", source[source.index("    def export_packed_ternary("):source.index("    def _ternary_audit(")])


if __name__ == "__main__":
    unittest.main()
