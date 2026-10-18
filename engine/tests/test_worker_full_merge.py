"""Transport/lifecycle contract; real native assembly has separate fixtures."""
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import Worker, RpcFault
from omni_core.connected_runtime import action_result_continuation
from omni_core.brain import AdaptiveBrain


class WorkerFullMergeTests(unittest.TestCase):
    def test_assembly_is_a_cancellable_exclusive_destination_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Worker()
            try:
                def assemble(root, identity, parents, settings, **options):
                    self.assertEqual(identity, "new-identity")
                    self.assertEqual(root, Path(directory).resolve())
                    self.assertEqual(len(worker._brain_leases), 1)
                    self.assertEqual(worker.brains, {})
                    self.assertTrue(worker.request_cooperative_cancel())
                    self.assertTrue(options["cancelled"]())
                    raise InterruptedError("cancelled at exact copied state boundary")
                with patch("omni_core.full_merge.assemble_full_merge", assemble):
                    with self.assertRaises(RpcFault) as caught:
                        worker.dispatch({"jsonrpc": "2.0", "id": "merge-1", "method": "full_merge_assemble",
                            "params": {"brainId": "new-identity", "storagePath": directory,
                                       "parents": [], "config": {}}})
                self.assertEqual(caught.exception.code, -32800)
                self.assertTrue(caught.exception.data["safeBoundary"])
                self.assertFalse(worker._brain_leases)
                self.assertIsNone(worker._active_request)
                self.assertFalse((Path(directory) / "engine/brain.json").exists())
            finally:
                worker.shutdown({}, None)

    def test_verification_closes_lease_without_installing_another_live_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Worker()
            try:
                with patch("omni_core.full_merge.verify_full_merge", return_value={"brainId": "new-identity", "verified": True}) as verify:
                    result = worker.dispatch({"jsonrpc": "2.0", "id": 1, "method": "full_merge_verify",
                        "params": {"brainId": "new-identity", "storagePath": directory}})
                self.assertTrue(result["result"]["verified"])
                self.assertEqual(verify.call_args.args[:2], (Path(directory).resolve(), "new-identity"))
                self.assertEqual(worker.brains, {})
                self.assertEqual(worker._brain_leases, {})
            finally:
                worker.shutdown({}, None)

    def test_inspection_worker_does_not_expose_merge_or_learning(self):
        with patch.dict("os.environ", {"OMNI_WORKER_ROLE": "inspection"}):
            worker = Worker()
        try:
            for method in ("full_merge_assemble", "full_merge_verify", "full_merge_activate_ingestion"):
                self.assertNotIn(method, worker.methods)
        finally:
            worker.shutdown({}, None)


    def test_action_evidence_rpc_forwards_closed_continuation_and_real_three_argument_progress(self):
        worker = Worker()
        selected = {"pendingId": "b" * 64, "originalEvidenceId": "c" * 64, "parentId": "parent"}
        def learned(**arguments):
            self.assertEqual(arguments["merge_continuation"], selected)
            arguments["progress"](0.5, "Committed one record", {"committedRecords": 1})
            return {"committed": True}
        brain = SimpleNamespace(brain_id="merged", learn_action_result=learned)
        try:
            with patch.object(worker, "_get", return_value=brain), patch.object(worker, "notify") as notify:
                result = worker.learn_action_result({"brainId": "merged", "evidenceId": "a" * 64,
                    "evidenceSha256": "d" * 64, "executionId": "00000000-0000-4000-8000-000000000000",
                    "evidencePath": "fixture", "provenance": {}, "mergeContinuation": selected}, "request")
                self.assertTrue(result["committed"])
                self.assertEqual(notify.call_args.kwargs["progress"], 0.5)
                self.assertEqual(notify.call_args.kwargs["data"], {"committedRecords": 1})
        finally:
            worker.shutdown({}, None)

    def test_failed_action_learning_reloads_committed_state_without_an_external_action(self):
        worker = Worker()
        brain = SimpleNamespace(brain_id="merged", learn_action_result=Mock(side_effect=RuntimeError("mid-record failure")))
        try:
            with patch.object(worker, "_get", return_value=brain), patch.object(worker, "_restore_committed_brain_after_failed_chat") as recover:
                with self.assertRaisesRegex(RuntimeError, "mid-record"):
                    worker.learn_action_result({"brainId": "merged", "evidenceId": "a" * 64,
                        "evidenceSha256": "d" * 64, "executionId": "00000000-0000-4000-8000-000000000000"}, "request")
                recover.assert_called_once_with(brain, prior_ledger_head=None, turn_id="", input_sha256="")
        finally:
            worker.shutdown({}, None)

    def test_action_continuation_cannot_select_a_different_parent_source_or_policy(self):
        evidence = "a" * 64
        selection = {"pendingId": "b" * 64, "originalEvidenceId": evidence, "parentId": "parent"}
        checkpoint = {"transactionId": evidence, "contentHash": "c" * 64,
            "sourceNameHash": hashlib.sha256(("action-result-" + evidence).encode()).hexdigest(),
            "policy": "pretrain", "resolvedKind": "jsonl"}
        queue = [{"id": selection["pendingId"], "parentId": "parent", "status": "queued", "checkpoint": checkpoint}]
        brain = SimpleNamespace(_connected_regions=object(), connected_pending_ingestions=queue,
            _sha256_identifier=AdaptiveBrain._sha256_identifier)
        # Isolate selection after strict queue parsing. The real source
        # fixture validates and resumes a complete native v3 cursor.
        with patch("omni_core.connected_runtime.validate_pending_ingestions", return_value=queue):
            self.assertEqual(action_result_continuation(brain, selection, "c" * 64)[1], "action-result-" + evidence)
            for bad in ({**selection, "parentId": "other"}, {**selection, "originalEvidenceId": "d" * 64},
                        {**selection, "extra": "forged"}):
                with self.assertRaises(ValueError):
                    action_result_continuation(brain, bad, "c" * 64)
            with self.assertRaises(ValueError):
                action_result_continuation(brain, selection, "e" * 64)
            checkpoint["policy"] = "archive"
            with self.assertRaises(ValueError):
                action_result_continuation(brain, selection, "c" * 64)


if __name__ == "__main__":
    unittest.main()
