"""Tiny native merge fixtures; no saved user brain or corpus training."""
import copy
import sys
import tempfile
import unittest
import shutil
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded, BoundedTensorFile
from omni_core.config import OmniConfig
from omni_core.connected_regions import ConnectedModalityHub, ConnectedOmniDecoder, NativeRegion, RegionIdentity
from omni_core.full_merge import (connected_descriptor, connected_state_inventory, inspect_parent_snapshot,
    load_connected_decoder_bounded, load_connected_regions, merged_config, assemble_full_merge, verify_full_merge)
from omni_core.persistence import atomic_write_json, tensor_checksum
from omni_core.vsa import NeuralSubstrate
from omni_core.brain import AdaptiveBrain
from omni_core.offload import ResourcePolicy, ResourceReading, DurableReplayBuffer, PagedWorkingMemory
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.shared_resource_ledger import SharedResourceLedger
from omni_core.connected_transaction import ConnectedExperienceTransaction


def config(width=8):
    return OmniConfig.micro(d_model=width, idea_dim=width, n_heads=1, n_layers=1,
        d_ff=16, vsa_dim=16, router_neurons=4, max_seq_len=16,
        working_memory_slots=4, memory_resident_items=4, modality_channels=4,
        image_size=8, audio_samples=16, video_frames=2)


def region(name, width=8):
    return NativeRegion(RegionIdentity(name, name, "a" * 64), config(width))


def connected():
    parts = [region("a"), region("b", 16)]
    root = replace(config(24), vsa_dim=32, router_neurons=8)
    return ConnectedOmniDecoder(root, parts)


def snapshot(root, parent):
    relative = "snapshots/" + parent.identity.parent_id + "/engine"
    directory = root / relative
    tensors = {}
    for field in ("decoder", "memory_bridge", "idea_adapter", "liquid", "modalities"):
        tensors.update({field + "." + key: value for key, value in getattr(parent, field).state_dict().items()})
    atomic_save_tensors_bounded(directory / "core.safetensors", tensors)
    plastic = {"router." + key: value for key, value in parent.router.state_dict().items()}
    plastic["state.liquid"] = parent.liquid_state
    atomic_save_tensors_bounded(directory / "plasticity.safetensors", plastic)
    memory = NeuralSubstrate(parent.config.vsa_dim, seed=parent.config.seed)
    memory.learn("a private experience from " + parent.identity.parent_id)
    memory.save_sharded(directory / "substrate")
    metadata = {"brain_id": parent.identity.parent_id, "config": parent.config.to_dict(),
                "substrate": memory.metadata(), "expert_count": 0}
    atomic_write_json(directory / "brain.json", metadata)
    return inspect_parent_snapshot(root, parent_id=parent.identity.parent_id,
        engine_relative_path=relative, generation_sha256="a" * 64)


class FullMergeRegionsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)

    def test_all_parent_owners_preserved_and_no_float_learned_copy(self):
        first, second = region("a"), region("b", 16)
        before = [tensor_checksum(part.state_dict().values()) for part in (first, second)]
        graph = ConnectedOmniDecoder(config(24), [first, second]).eval()
        hub = ConnectedModalityHub(graph)
        report = connected_state_inventory(graph, hub)
        self.assertEqual(report["regions"], 2)
        self.assertEqual(report["floatingLearnedParameters"], 0)
        self.assertEqual(before, [tensor_checksum(part.state_dict().values()) for part in (first, second)])
        state = graph.state_dict()
        for prefix in ("decoder", "memory_bridge", "idea_adapter", "liquid", "modalities", "router"):
            self.assertTrue(any(key.startswith("regions.1." + prefix + ".") for key in state), prefix)

    def test_joined_action_learning_snapshots_include_nested_packed_owners(self):
        inner = region("inner", 16)
        inner.decoder = ConnectedOmniDecoder(config(16), [region("a"), region("b")])
        graph = ConnectedOmniDecoder(config(24), [inner, region("c")])
        head = graph.action_policy
        saved = {name: value.clone() for name, value in head.state_dict().items()}
        self.assertTrue(any(name.startswith("borrowed_regions.0.borrowed_regions.") for name in saved))
        self.assertTrue(any("connection" in name or "strength" in name for name in saved))
        self.assertEqual(len(list(head.modules())), len({id(module) for module in head.modules()}))
        self.assertFalse(any(name.startswith("action_policy.borrowed_regions") for name in graph.state_dict()))
        leaf = inner.decoder.regions[0].decoder.action_policy
        for value in leaf.buffers():
            if value.dtype == torch.uint8 and value.numel():
                value.flatten()[0].bitwise_xor_(1)
                break
        self.assertNotEqual(tensor_checksum(saved.values()), tensor_checksum(head.state_dict().values()))
        head.load_state_dict(saved)
        self.assertEqual(tensor_checksum(saved.values()), tensor_checksum(head.state_dict().values()))

    def test_causal_connected_forward_and_regions_have_real_influence(self):
        graph = connected().eval()
        ids = torch.tensor([[1, 70, 71, 72]])
        with torch.no_grad():
            before = graph(ids)
            changed_future = graph(torch.tensor([[1, 70, 99, 73]]))
        self.assertEqual(tuple(before["hidden"].shape), (1, 4, 24))
        self.assertTrue(torch.equal(before["logits"][:, :2], changed_future["logits"][:, :2]))
        graph.cortical_exchange.bridges["r0_to_r1"].fill_ternary_(1)
        with torch.no_grad():
            after = graph(ids)
        self.assertFalse(torch.equal(before["hidden"][..., 8:], after["hidden"][..., 8:]))
        self.assertFalse(torch.equal(before["logits"], after["logits"]))

    def test_neural_recall_router_and_liquid_are_active_without_persistent_prefill_mutation(self):
        graph = connected().eval()
        calls = []
        for index, part in enumerate(graph.regions):
            part.attach_recall(lambda cue, i=index: (calls.append((i, tuple(cue.shape))) or torch.ones(16)))
        prior = [tensor_checksum([part.liquid_state, part.router.population.membrane]) for part in graph.regions]
        with torch.no_grad():
            graph(torch.tensor([[1, 80, 81]]))
        self.assertEqual([i for i, _ in calls], [0, 0, 0, 1, 1, 1])
        self.assertEqual(prior, [tensor_checksum([part.liquid_state, part.router.population.membrane]) for part in graph.regions])

    def test_integration_updates_packed_bridges_and_not_parent_snapshots(self):
        graph = connected().train()
        before = [tensor_checksum(part.state_dict().values()) for part in graph.regions]
        graph.last_region_activity = (torch.ones(1, 8), torch.ones(1, 16))
        bridges = tensor_checksum(graph.workspace_exchange.state_dict().values())
        for module in graph.workspace_exchange.bridges.values():
            module.online_learning_rate = 10.0
        for module in graph.cortical_exchange.bridges.values():
            module.online_learning_rate = 10.0
        graph.integration_loss().backward()
        self.assertNotEqual(bridges, tensor_checksum(graph.workspace_exchange.state_dict().values()))
        self.assertEqual(before, [tensor_checksum(part.state_dict().values()) for part in graph.regions])

    def test_saved_connected_core_reloads_identically(self):
        graph = connected().eval()
        graph.cortical_exchange.bridges["r0_to_r1"].fill_ternary_(1)
        ids = torch.tensor([[1, 65, 66]])
        with torch.no_grad():
            expected = graph(ids)["logits"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connected.safetensors"
            atomic_save_tensors_bounded(path, {"decoder." + key: value for key, value in graph.state_dict().items()})
            restored = connected().eval()
            load_connected_decoder_bounded(restored, BoundedTensorFile(path))
            with torch.no_grad():
                actual = restored(ids)["logits"]
        self.assertTrue(torch.equal(expected, actual))

    def test_action_and_media_share_full_connected_state(self):
        graph = connected().eval()
        hub = ConnectedModalityHub(graph).eval()
        state = torch.randn(1, 24)
        features = torch.randn(1, 1024)
        prefixes = torch.tensor([[257, 123]])
        with torch.no_grad():
            arguments = graph.action_argument_head(state, features, prefixes)
            embedding = hub.perception_embedding("image", torch.rand(1, 3, 8, 8))
            audio = hub.generate("audio", state, seed=9)
        self.assertEqual(tuple(arguments.shape), (1, 2, 257))
        self.assertEqual(tuple(embedding.shape), (1, 24))
        self.assertEqual(tuple(audio.shape), (1, 16))
        self.assertTrue(bool(torch.isfinite(audio).all()))

    def test_connected_generation_uses_one_append_only_frontier_and_invalidates_shift(self):
        graph = connected().eval()
        graph.workspace_exchange.bridges["r0_to_r1"].fill_ternary_(1)
        ids = torch.tensor([[1, 80, 81]])
        with torch.no_grad():
            expected = graph(ids)["logits"][:, -1]
            logits, cache = graph._generation_step_bounded(ids, None, None)
            self.assertEqual(cache.length, 3)
            self.assertEqual(len(cache.attention), 2)
            self.assertEqual(tuple(logits.shape), (1, 261))
            self.assertTrue(torch.allclose(expected, logits, atol=1e-5, rtol=1e-5))
            continued, resumed = graph._generation_step_bounded(torch.tensor([[1, 80, 81, 82]]), None, cache)
            self.assertIs(cache, resumed)
            self.assertEqual(resumed.length, 4)
            _, reset = graph._generation_step_bounded(torch.tensor([[81, 82]]), None, resumed)
            self.assertIsNot(reset, resumed)
            self.assertEqual(reset.length, 2)
            reset.close()

    def test_nested_parent_remains_connected_in_forward_cached_decode_and_reload(self):
        def nested():
            child = connected().eval()
            child_config = child.config
            parent = NativeRegion(RegionIdentity("merged", "merged", "c" * 64), child_config,
                connected=SimpleNamespace(decoder=child, modalities=ConnectedModalityHub(child)))
            outer = ConnectedOmniDecoder(replace(config(32), vsa_dim=48, router_neurons=12), [parent, region("c")]).eval()
            return outer
        graph = nested()
        ids = torch.tensor([[1, 76, 77]])
        with torch.no_grad():
            before = graph(ids)["logits"]
            graph.regions[0].decoder.cortical_exchange.bridges["r0_to_r1"].fill_ternary_(1)
            after = graph(ids)["logits"]
            self.assertFalse(torch.equal(before, after))
            cached, frontier = graph._generation_step_bounded(ids, None, None)
            self.assertTrue(torch.allclose(after[:, -1], cached, atol=1e-5, rtol=1e-5))
            self.assertIsNotNone(frontier.nested[0])
            frontier.close()
            routes = graph.tool_route_head(torch.randn(1, 1024))
            self.assertTrue(bool(torch.isfinite(routes).all()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested.safetensors"
            atomic_save_tensors_bounded(path, {"decoder." + k: v for k, v in graph.state_dict().items()})
            restored = nested()
            restored.restore_layout(graph.layout())
            load_connected_decoder_bounded(restored, BoundedTensorFile(path))
            with torch.no_grad():
                self.assertTrue(torch.equal(after, restored(ids)["logits"]))

    def test_real_native_snapshot_loading_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = [snapshot(root, region("a")), snapshot(root, region("b", 16))]
            original_bytes = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            descriptor = connected_descriptor(snapshots)
            merged = load_connected_regions(root, descriptor, merged_config(descriptor, settings=config()))
            merged.decoder.eval()
            with torch.no_grad():
                value = merged.decoder(torch.tensor([[1, 80, 81]]))
            self.assertEqual(value["hidden"].shape[-1], 24)
            self.assertEqual(set(merged.memories), {s.region_id for s in snapshots})
            for path, content in original_bytes.items():
                self.assertEqual((root / path).read_bytes(), content)
            merged.close()

    def test_only_exact_neural_identity_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = snapshot(root, region("a"))
            two = replace(one, parent_id="b", generation_sha256="b" * 64)
            deduped = connected_descriptor([one, two])
            self.assertEqual(len(deduped["regions"]), 1)
            self.assertEqual(len(deduped["parents"]), 2)
            changed = replace(two, neural_sha256="c" * 64, region_id="different")
            self.assertEqual(len(connected_descriptor([one, changed])["regions"]), 2)

    def test_production_brain_uses_connected_factory_saves_and_reloads(self):
        # Physical resource behavior has separate coverage. This tiny (<10MB)
        # test uses a private, explicit quota fixture; it changes no app RAM or
        # disk setting and makes no claim about fitting a production model.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = [snapshot(root, region("a")), snapshot(root, region("b", 16))]
            descriptor = connected_descriptor(snapshots)
            settings = merged_config(descriptor, settings=config())
            quota = SharedResourceLedger(root / "test-quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(
                        total_memory_bytes=8 * 1024**3, available_memory_bytes=6 * 1024**3,
                        process_memory_bytes=256 * 1024**2, disk_total_bytes=128 * 1024**3,
                        disk_free_bytes=100 * 1024**3, managed_process_memory_bytes=256 * 1024**2,
                        managed_memory_verified=True,
                    ))
            with patch("omni_core.brain.ResourcePolicy", policy):
                brain = AdaptiveBrain("merged", root, settings, _connected_descriptor=descriptor)
                try:
                    self.assertIsInstance(brain.decoder, ConnectedOmniDecoder)
                    brain.decoder.eval()
                    with torch.no_grad():
                        expected = brain.decoder(torch.tensor([[1, 81, 82]]))["logits"]
                    brain.save()
                    self.assertEqual(len(brain._metadata()["connected_regions"]["regions"]), 2)
                finally:
                    brain.close()
                restored = AdaptiveBrain.load(root, expected_brain_id="merged")
                try:
                    restored.decoder.eval()
                    with torch.no_grad():
                        actual = restored.decoder(torch.tensor([[1, 81, 82]]))["logits"]
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertEqual(set(restored._connected_regions.memories), {p.region_id for p in snapshots})
                finally:
                    restored.close()

    def test_production_assembler_adopts_runtime_and_has_truthful_merge_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "first"
            root.mkdir()
            quota = SharedResourceLedger(root / "test-quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
                        available_memory_bytes=6 * 1024**3, process_memory_bytes=256 * 1024**2,
                        disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
                        managed_process_memory_bytes=256 * 1024**2, managed_memory_verified=True))
            snapshots = []
            for identifier, width in (("a", 8), ("b", 16)):
                initial = snapshot(root, region(identifier, width))
                directory = root / initial.engine_relative_path
                conversation = NeuralConversationLedger(directory / "conversation.sqlite3", identifier)
                conversation.append("message", [{"id": "msg-" + identifier, "role": "human",
                    "content": "An experience from " + identifier, "created_at": "2026-10-01T00:00:00Z"}])
                metadata = json.loads((directory / "brain.json").read_text())
                metadata["conversation"] = conversation.summary()
                metadata["recent_token_context"] = [1, 70 + width]
                conversation.close()
                replay = DurableReplayBuffer(directory / "state/replay.sqlite3", policy(directory))
                replay.append(torch.ones(width))
                PagedWorkingMemory(directory / "state/working-memory.sqlite3", policy(directory))
                atomic_write_json(directory / "brain.json", metadata)
                snapshots.append(inspect_parent_snapshot(root, parent_id=identifier,
                    engine_relative_path=initial.engine_relative_path, generation_sha256="a" * 64))
            declarations = [{"brainId": p.parent_id, "enginePath": p.engine_relative_path,
                             "generationSha256": p.generation_sha256} for p in snapshots]
            original = {str(path.relative_to(root)): path.read_bytes()
                        for path in (root / "snapshots").rglob("*") if path.is_file()}
            with patch("omni_core.brain.ResourcePolicy", policy):
                receipt = assemble_full_merge(root, "merged", declarations, config())
                self.assertEqual(receipt["runtimeAdoption"]["counts"]["historyEntries"], 2)
                self.assertFalse(receipt["origin"]["initialCurriculumRerun"])
                origin_bytes = (root / "engine/origin/connected-origin.json").read_bytes()
                retry = assemble_full_merge(root, "merged", declarations, config())
                self.assertEqual(retry["runtimeAdoption"], receipt["runtimeAdoption"])
                self.assertEqual(retry["regionalState"], receipt["regionalState"])
                self.assertEqual((root / "engine/origin/connected-origin.json").read_bytes(), origin_bytes)
                verified = verify_full_merge(root, "merged")
                self.assertTrue(verified["verified"])
                self.assertEqual(verified["replay"]["count"], 2)
                self.assertEqual(verified["regionalState"]["regions"], 2)
                # Merge a completed merge again. The preserved parent's inner
                # exchanges and graph generations must remain usable, not be
                # loaded as an incompatible plain decoder or archived only.
                second = Path(temporary) / "second"
                source_parent = second / "merge/parents/merged"
                shutil.copytree(root, source_parent)
                third = snapshot(second, region("c"))
                third_engine = second / third.engine_relative_path
                ledger = NeuralConversationLedger(third_engine / "conversation.sqlite3", "c")
                ledger.append("message", [{"id": "msg-c", "role": "human", "content": "Third independent experience"}])
                third_metadata = json.loads((third_engine / "brain.json").read_text())
                third_metadata["conversation"] = ledger.summary()
                third_metadata["recent_token_context"] = [1, 90]
                ledger.close()
                DurableReplayBuffer(third_engine / "state/replay.sqlite3", policy(third_engine)).append(torch.ones(8))
                PagedWorkingMemory(third_engine / "state/working-memory.sqlite3", policy(third_engine))
                atomic_write_json(third_engine / "brain.json", third_metadata)
                second_declarations = [
                    {"brainId": "merged", "enginePath": "merge/parents/merged/engine", "generationSha256": "b" * 64},
                    {"brainId": "c", "enginePath": third.engine_relative_path, "generationSha256": "c" * 64},
                ]
                twice = assemble_full_merge(second, "twice", second_declarations, config())
                self.assertEqual(twice["runtimeAdoption"]["counts"]["historyEntries"], 3)
                repeated = verify_full_merge(second, "twice")
                self.assertTrue(repeated["verified"])
                reloaded = AdaptiveBrain.load(second, expected_brain_id="twice")
                try:
                    self.assertIsInstance(reloaded.decoder.regions[0].decoder, ConnectedOmniDecoder)
                    self.assertEqual(len(list(reloaded._connected_regions.iter_memories())), 4)
                    self.assertTrue(reloaded._ground_up_origin_is_complete(second / "engine/origin"))
                    with torch.no_grad():
                        out = reloaded.decoder(torch.tensor([[1, 75]]))["logits"]
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    from tests.test_connected_transactions import graph_state
                    before_graphs = graph_state(reloaded._connected_regions)
                    before_core = tensor_checksum(reloaded.decoder.state_dict().values())
                    nested_router = reloaded.decoder.regions[0].decoder.regions[1].router
                    with patch.object(nested_router, "route", side_effect=RuntimeError("nested-admission fault")):
                        with self.assertRaisesRegex(RuntimeError, "nested-admission fault"):
                            with ConnectedExperienceTransaction(reloaded):
                                reloaded.decoder.observe_experience(torch.ones(1, reloaded.config.idea_dim), event_key="nested-failed-event")
                    self.assertEqual(graph_state(reloaded._connected_regions), before_graphs)
                    self.assertEqual(tensor_checksum(reloaded.decoder.state_dict().values()), before_core)
                finally:
                    reloaded.close()
            for path, data in original.items():
                self.assertEqual((root / path).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
