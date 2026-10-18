"""Bounded complete-state adoption; no model training or user data."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
from omni_core.brain import AdaptiveBrain
from omni_core.connected_runtime import adopt_connected_runtime, validate_runtime_adoption
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.full_merge import ParentRegionSnapshot
from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.offload import DurableReplayBuffer, PagedWorkingMemory
from omni_core.persistence import atomic_write_json
from omni_core.tokenizer import ByteTokenizer
from omni_core.recent_token_activity import RecentTokenActivity


class TinyPolicy:
    def require_disk(self, count, label):
        if count > 4 * 1024 * 1024:
            raise AssertionError("tiny fixture attempted a large allocation")


def source(root, identifier, *, text="A real ordinary experience", tokens=(1, 71), region_id=None):
    relative = "merge/parents/" + identifier + "/engine"
    path = root / relative
    ledger = NeuralConversationLedger(path / "conversation.sqlite3", identifier)
    message = {"id": "same-original-id", "role": "human", "content": text,
               "created_at": "2026-10-01T00:00:00Z", "attention_epoch": 0}
    ledger.append("message", [message])
    head = ledger.summary()
    ledger.close()
    replay = DurableReplayBuffer(path / "state/replay.sqlite3", TinyPolicy())
    replay.append(torch.tensor([1.0, -1.0]))
    pages = PagedWorkingMemory(path / "state/working-memory.sqlite3", TinyPolicy())
    pages.append(torch.tensor([0.5, -0.5]), {"assemblyId": "a-1", "salience": 0.8})
    atomic_save_tensors_bounded(path / "plasticity.safetensors", {
        "state.working_memory": torch.tensor([[0.25, 0.75]]),
        "memory_lifecycle.afterimage_vectors": torch.tensor([[1.0, -1.0]]),
    })
    metadata = {
        "brain_id": identifier, "conversation": head,
        "recent_token_context": list(tokens),
        "recent_token_activity": RecentTokenActivity([{"tokenCount": len(tokens), "afterimageId": "afterimage"}]).metadata(tokens),
        "updated_at": "2026-10-01T00:00:00Z",
        "workspace_items": [{"assemblyId": "a-1", "id": "workspace", "salience": 0.9}],
        "memory_lifecycle": {"afterimageItems": [{"id": "afterimage", "assemblyId": "a-1", "strength": 0.7, "lastActiveCycle": 1}],
            "activeFocus": [{"assemblyId": "a-1", "salience": 0.9}], "cycle": 2},
        "pending_chat_slow_learning": [{"format": "omni-chat-slow-learning-job", "formatVersion": 1,
            "jobId": "a" * 64, "turnId": "parent-turn", "inputSha256": hashlib.sha256(text.encode()).hexdigest(),
            "humanMessageId": "same-original-id", "evidenceRole": "human", "afterimageId": "afterimage",
            "replayCursor": {"phase": "text", "character": 2, "byte": 0}, "completedReplayPasses": 3}],
        "counters": {"training_steps": 7}, "modality_training": {"image": 2},
        "training_sources": [{"name": "real source", "sha256": "c" * 64}],
    }
    atomic_write_json(path / "brain.json", metadata)
    return ParentRegionSnapshot(identifier, region_id or identifier, relative, "a" * 64, identifier * 64,
        {"idea_dim": 2, "vsa_dim": 4}, "b" * 64, "c" * 64, "d" * 64)


def target(root, parents, *, maximum=16, slots=1):
    unique = {p.region_id: p for p in parents}
    policy = TinyPolicy()
    return SimpleNamespace(
        brain_id="merged", storage_path=root, engine_path=root / "engine",
        _connected_regions=SimpleNamespace(descriptor={"regions": [{"regionId": p.region_id, "config": p.config} for p in unique.values()]}),
        config=SimpleNamespace(idea_dim=2 * len(unique), vsa_dim=4 * len(unique), vocab_size=261, max_seq_len=maximum, working_memory_slots=slots),
        resource_policy=policy,
        conversation=NeuralConversationLedger(root / "engine/conversation.sqlite3", "merged"),
        replay=DurableReplayBuffer(root / "engine/state/replay.sqlite3", policy),
        paged_working_memory=PagedWorkingMemory(root / "engine/state/working-memory.sqlite3", policy),
        _validated_ingestion_checkpoints=AdaptiveBrain._validated_ingestion_checkpoints,
        working_memory=[], workspace_items=[], memory_lifecycle=OrganicMemoryLifecycle(),
        counters={"training_steps": 0}, modality_training={"image": 0}, tokenizer=ByteTokenizer(), current_context={},
    )


class ConnectedRuntimeAdoptionTests(unittest.TestCase):
    def test_complete_history_pending_learning_and_activity_are_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = [source(root, "a"), source(root, "b", text="A conflicting independent experience", tokens=(1, 81))]
            brain = target(root, parents)
            try:
                before = {str(path.relative_to(root / "merge")): path.read_bytes() for path in (root / "merge").rglob("*") if path.is_file()}
                receipt = adopt_connected_runtime(brain, parents)
                self.assertEqual(receipt["counts"]["historyEntries"], 2)
                self.assertEqual(brain.recent_token_context, [1, 71, 1, 81])
                self.assertFalse(receipt["historyInjectedAsPrompt"])
                self.assertEqual(len(brain.replay), 2)
                self.assertTrue(torch.equal(brain.replay[0], torch.tensor([1., -1., 0., 0.])))
                self.assertEqual(len(brain.working_memory), 1)
                self.assertEqual(len(brain.paged_working_memory.page_ids()), 3)
                self.assertEqual(len(brain.memory_lifecycle.afterimage_vectors), 2)
                self.assertEqual(brain.memory_lifecycle.afterimage_vectors[1].numel(), 4)
                self.assertEqual(brain.recent_token_activity.summary()["trackedTokens"], 4)
                self.assertEqual([row["afterimageId"] for row in brain.recent_token_activity.spans],
                    [row["id"] for row in brain.memory_lifecycle.afterimage_items])
                self.assertEqual(brain.counters["training_steps"], 14)
                for job in brain.pending_chat_slow_learning:
                    message = brain.conversation.payload_by_id("message", job["humanMessageId"])
                    self.assertEqual(hashlib.sha256(message["content"].encode()).hexdigest(), job["inputSha256"])
                    self.assertEqual(job["replayCursor"], {"phase": "text", "character": 2, "byte": 0})
                    self.assertEqual(job["completedReplayPasses"], 3)
                self.assertEqual(receipt, adopt_connected_runtime(brain, parents))
                self.assertEqual(brain.conversation.integrity()["messageCount"], 2)
                self.assertEqual(len(brain.replay), 2)
                self.assertEqual(validate_runtime_adoption(receipt), receipt)
                for relative, data in before.items():
                    self.assertEqual((root / "merge" / relative).read_bytes(), data)
            finally:
                brain.conversation.close()

    def test_identical_history_deduplicates_but_retains_all_parent_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = [source(root, "a"), source(root, "b", region_id="a")]
            brain = target(root, parents)
            try:
                receipt = adopt_connected_runtime(brain, parents)
                self.assertEqual(receipt["counts"]["historyEntries"], 1)
                self.assertEqual(len(receipt["parents"]), 2)
                self.assertEqual(len(brain.pending_chat_slow_learning), 1)
                self.assertEqual(len(brain.replay), 1)
            finally:
                brain.conversation.close()

    def test_undersized_context_rejected_before_any_history_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = [source(root, "a"), source(root, "b")]
            brain = target(root, parents, maximum=3)
            try:
                with self.assertRaisesRegex(ValueError, "larger working window"):
                    adopt_connected_runtime(brain, parents)
                self.assertEqual(brain.conversation.counts()["totalEntries"], 0)
                self.assertEqual(len(brain.replay), 0)
            finally:
                brain.conversation.close()

    def test_corrupt_committed_history_and_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = [source(root, "a"), source(root, "b")]
            path = root / parents[0].engine_relative_path / "brain.json"
            metadata = json.loads(path.read_text())
            metadata["conversation"]["headSha256"] = "f" * 64
            atomic_write_json(path, metadata)
            brain = target(root, parents)
            try:
                with self.assertRaisesRegex(ValueError, "bound to its committed"):
                    adopt_connected_runtime(brain, parents)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    validate_runtime_adoption({"format": "omni-connected-runtime-adoption", "formatVersion": 1, "sha256": "f" * 64})
            finally:
                brain.conversation.close()


if __name__ == "__main__":
    unittest.main()
