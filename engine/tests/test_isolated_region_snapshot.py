"""Synthetic module skeleton/control/storage fixtures; no model constructors or decoding."""
import inspect
import sys
import tempfile
import threading
import unittest
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.isolated_module_snapshot import IsolatedModuleSnapshot
from omni_core.slow_state_snapshot import SlowStateSnapshot, slow_snapshot_lifetime
from omni_core.offload import NeuralStateResourcePause
from omni_core.brain import AdaptiveBrain
from omni_core.parameter_diagnostics import packed_diagnostic_write
from omni_core.native_core_paging import release_native_tensor_chunk
from worker import Worker
from codec_gateway import CodecOwner


class SyntheticRegion(nn.Module):
    def __init__(self): raise AssertionError("snapshot must not construct/initialize a neural model")
    def __deepcopy__(self, memo): raise AssertionError("original-device module deepcopy")
    def authoritative_packed_tensors(self): return (self._buffers["codes"],)

def region():
    value = object.__new__(SyntheticRegion)
    # Only the nn.Module registry protocol, populated with tiny primitive
    # tensors. No constructor, forward, backward or optimizer runs.
    value.__dict__.update({"training": True, "_parameters": {}, "_buffers": {
        "codes": torch.tensor([[0x55, 0xAA], [0x55, 0x55]], dtype=torch.uint8), "scale": torch.tensor([0.25])},
        "_modules": {}, "_non_persistent_buffers_set": set(), "_backward_hooks": OrderedDict(),
        "_backward_pre_hooks": OrderedDict(), "_forward_hooks": OrderedDict(),
        "_forward_hooks_with_kwargs": OrderedDict(), "_forward_hooks_always_called": OrderedDict(),
        "_forward_pre_hooks": OrderedDict(), "_forward_pre_hooks_with_kwargs": OrderedDict(),
        "_state_dict_hooks": OrderedDict(), "_state_dict_pre_hooks": OrderedDict(),
        "_load_state_dict_pre_hooks": OrderedDict(), "_load_state_dict_post_hooks": OrderedDict(),
        "_is_full_backward_hook": None, "nested": {"activity": [torch.tensor([2.0, 3.0])]},
        "_native_core_pager": None, "_native_compute_device": torch.device("cpu"), "_online_transaction": None})
    return value

class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="region-state-fixture-")
        self.root = Path(self.temp.name)
        self.readings, self.disk = [], []
        self.policy = SimpleNamespace(status=lambda **kw: (self.readings.append(kw) or {"memoryPressure": False}),
            require_disk=lambda size, stage: self.disk.append((size, stage)))
    def tearDown(self): self.temp.cleanup()

    def test_existing_architecture_snapshot_uses_cpu_mapped_codes_not_deepcopy_or_constructor(self):
        source = region()
        with patch.object(torch.Tensor, "__deepcopy__", side_effect=AssertionError("tensor deepcopy")):
            snapshot = IsolatedModuleSnapshot({"image": source}, directory=self.root / "private", policy=self.policy, chunk_bytes=2)
        copied = snapshot.roots["image"]
        self.assertIs(type(copied), type(source))
        self.assertTrue(torch.equal(copied.codes, source.codes))
        self.assertEqual(copied.codes.device.type, "cpu")
        self.assertEqual(snapshot.pager.status()["cpuHeapBytes"], source.scale.numel() * source.scale.element_size() + source.codes.numel())
        self.assertEqual(snapshot.admitted_packed_ram_bytes, snapshot.packed_bytes)
        self.assertGreater(snapshot.packed_bytes, 0)
        copied.codes.fill_(0)
        copied.nested["activity"][0].fill_(0)
        self.assertFalse(torch.equal(copied.codes, source.codes))
        self.assertTrue(torch.equal(source.nested["activity"][0], torch.tensor([2., 3.])))
        self.assertTrue(self.readings)
        self.assertEqual(self.disk, []) # fitting small region does not write packed state to SSD
        snapshot.close()
        self.assertTrue(snapshot.closed)
        self.assertFalse((self.root / "private").exists())

    def test_nonweight_minimum_pause_is_explicit_before_copy(self):
        source = region()
        policy = SimpleNamespace(status=lambda **_kw: {"memoryPressure": True}, require_disk=self.policy.require_disk)
        with self.assertRaises(NeuralStateResourcePause) as raised:
            IsolatedModuleSnapshot({"audio": source}, directory=self.root / "private", policy=policy)
        self.assertTrue(raised.exception.status["paused"])
        self.assertFalse((self.root / "private").exists())

    def test_cancelled_copy_does_not_leave_a_model_or_owned_pages(self):
        with self.assertRaises(InterruptedError):
            IsolatedModuleSnapshot({"video": region()}, directory=self.root / "private", policy=self.policy, cancelled=lambda: True)
        self.assertFalse((self.root / "private").exists())

    def test_slow_integrity_snapshot_writes_refs_bounded_and_restores_registered_targets(self):
        source = region()
        metadata = {"expert_count": 1, "checksum": "fixture", "recent_token_context": [1, 2]}
        optimizer = {"state": {0: {"moment": torch.tensor([0.5])}}, "param_groups": [{"lr": .01}]}
        with patch.object(torch.Tensor, "clone", side_effect=AssertionError("full tensor clone")):
            snapshot = SlowStateSnapshot({"decoder": source}, {"optimizer": optimizer}, metadata,
                directory=self.root / "slow", policy=self.policy)
        self.assertEqual(snapshot["optimizer"]["state"][0]["moment"].item(), .5)
        self.assertTrue(all(spec.dtype != torch.uint8 for spec in snapshot.reader.specs.values()))
        self.assertIsNone(snapshot.journal.database) # no/few mutations do not copy a whole packed core
        self.assertTrue(self.disk)
        replacement = torch.zeros_like(source.codes)
        with packed_diagnostic_write(source, source.codes, replacement): source.codes.copy_(replacement)
        source._buffers["codes"] = source.codes.clone()
        source.scale.fill_(0)
        snapshot.restore_modules({"decoder": source})
        self.assertTrue(torch.equal(source.codes, torch.tensor([[0x55, 0xAA], [0x55, 0x55]], dtype=torch.uint8)))
        self.assertEqual(source.scale.item(), .25)
        directory = snapshot.directory
        snapshot.close()
        self.assertFalse(directory.exists())

    def test_ram_first_snapshot_spills_only_when_shared_remaining_budget_cannot_fit_owner(self):
        source = region()
        # Synthetic admission admits metadata/nonweight scratch but not even
        # this tiny complete packed owner; no extra private budget is invented.
        limit = [None]
        def status(**kw):
            size = kw["estimated_ram_bytes"]
            if limit[0] is None and size > 2_000_000: limit[0] = size
            return {"memoryPressure": limit[0] is not None and size > limit[0]}
        policy = SimpleNamespace(status=status, require_disk=self.policy.require_disk)
        with patch("omni_core.native_core_paging.release_native_tensor_chunk", wraps=release_native_tensor_chunk) as released:
            snapshot = IsolatedModuleSnapshot({"image": source}, directory=self.root / "cold", policy=policy)
        self.assertLess(snapshot.admitted_packed_ram_bytes, snapshot.packed_bytes)
        self.assertTrue(self.disk)
        self.assertTrue(torch.equal(snapshot.roots["image"].codes, source.codes))
        self.assertTrue(any(call.args[0] is snapshot.roots["image"].codes for call in released.call_args_list))
        snapshot.close()

    def test_failure_rollback_restores_learned_codes_and_uint8_control_baseline(self):
        source = region()
        source._buffers["_row_stability"] = torch.zeros(2, dtype=torch.uint8)
        snapshot = SlowStateSnapshot({"decoder": source}, {}, {}, directory=self.root / "slow", policy=self.policy)
        self.assertIsNone(snapshot.journal.database)
        self.assertEqual(snapshot._module_keys["decoder"]["codes"], "@journal")
        self.assertNotEqual(snapshot._module_keys["decoder"]["_row_stability"], "@journal")
        try:
            changed_codes = torch.zeros_like(source.codes)
            with packed_diagnostic_write(source, source.codes, changed_codes): source.codes.copy_(changed_codes)
            source._row_stability.copy_(torch.tensor([3, 7], dtype=torch.uint8)) # unhooked control still has an exact baseline
            raise RuntimeError("failure after both state mutations")
        except RuntimeError:
            snapshot.restore_modules({"decoder": source})
        self.assertTrue(torch.equal(source.codes, torch.tensor([[0x55, 0xAA], [0x55, 0x55]], dtype=torch.uint8)))
        self.assertTrue(bool(source._row_stability.eq(0).all()))
        snapshot.close()

    def test_slow_metadata_is_admitted_before_deepcopy_or_file_creation(self):
        policy = SimpleNamespace(status=lambda **_kw: {"memoryPressure": True}, require_disk=self.policy.require_disk)
        with patch("omni_core.slow_state_snapshot.copy.deepcopy", side_effect=AssertionError("copy before admission")):
            with self.assertRaises(NeuralStateResourcePause):
                SlowStateSnapshot({"decoder": region()}, {}, {"recent_token_context": [1, 2, 3]}, directory=self.root / "slow", policy=policy)
        self.assertFalse((self.root / "slow").exists())

    def test_nested_lifetimes_keep_parent_snapshot_until_parent_restore_finishes(self):
        created = []
        @slow_snapshot_lifetime
        def child():
            item = SlowStateSnapshot({"decoder": region()}, {}, {}, directory=self.root / "slow", policy=self.policy)
            created.append(item)
            raise RuntimeError("child failure")
        @slow_snapshot_lifetime
        def parent():
            source = region()
            item = SlowStateSnapshot({"decoder": source}, {}, {}, directory=self.root / "slow", policy=self.policy)
            created.append(item)
            with self.assertRaises(RuntimeError): child()
            self.assertFalse(item.closed)
            item.restore_modules({"decoder": source})
            self.assertFalse(item.closed)
        parent()
        self.assertTrue(all(item.closed for item in created))

    def test_inline_preparation_failure_keeps_owned_failed_future_not_silent_none(self):
        worker = Worker.__new__(Worker)
        worker._inline_executor_closed = False
        worker._inline_lock = threading.RLock()
        worker._inline_generations = {}
        worker._codec_gateway = None
        worker._cooperative_cancel = threading.Event()
        worker.notify = lambda *_args, **_kw: None
        brain = SimpleNamespace(brain_id="brain", engine_path=self.root / "engine", device="cpu",
            config=SimpleNamespace(vsa_dim=2), liquid_state=torch.tensor([0., 0.]),
            resource_policy=SimpleNamespace(status=lambda **_kw: {"memoryPressure": True}, include_accelerator_memory=False))
        record = worker._start_inline_generation(brain, "a" * 32, "turn", {
            "kind": "imagine", "toolId": "modality.imagine", "action": "generate", "arguments": {"modality": "image"}}, lambda *_args: None)
        self.assertIsNotNone(record)
        self.assertTrue(record.finished.is_set())
        self.assertIsInstance(record.future.exception(), NeuralStateResourcePause)
        self.assertFalse(record.staging_root.exists())
        self.assertFalse(worker._cooperative_cancel.is_set())

    def inline_fixture(self, submit=None):
        worker = Worker.__new__(Worker)
        worker._inline_executor_closed = False
        worker._inline_lock = threading.RLock()
        worker._inline_generations = {}
        worker._cooperative_cancel = threading.Event()
        notifications = []
        worker.notify = lambda kind, **data: notifications.append((kind, data))
        worker._codec_gateway = SimpleNamespace(current_owner=lambda: CodecOwner("rpc", "brain", "", "turn", ""),
            scope=lambda *_args: nullcontext())
        def immediate(function):
            future = Future()
            try: future.set_result(function())
            except BaseException as error: future.set_exception(error)
            return future
        worker._inline_executor = SimpleNamespace(submit=submit or immediate)
        class FixtureBrain(SimpleNamespace):
            def _modality_idea(self, *_args): return torch.tensor([[.25, .5]])
            def _modality_idea_evidence(self, *_args): return {"sameBrain": True}
            def generate_modality(self, **_kwargs): return self.fixture_decode(self)
        brain = FixtureBrain(brain_id="brain", engine_path=self.root / "engine", device="cpu",
            config=SimpleNamespace(vsa_dim=2), liquid_state=torch.tensor([0., 0.]),
            modalities=SimpleNamespace(image=region()), resource_policy=self.policy,
            counters={"inference_count": 1}, modality_training={}, installed_modality_packs=[])
        brain.fixture_decode = lambda _snapshot: {}
        return worker, brain, notifications

    def start_fixture_inline(self, worker, brain):
        return worker._start_inline_generation(brain, "b" * 32, "turn", {
            "kind": "imagine", "toolId": "modality.imagine", "action": "generate", "arguments": {"modality": "image"}}, lambda *_args: None)

    def test_inline_state_stays_live_through_stub_job_and_closes_only_after_finish(self):
        worker, brain, notices = self.inline_fixture()
        captured = []
        def decode(snapshot):
            record = worker._inline_generations[("brain", "b" * 32)]
            state = record.snapshot_state
            self.assertFalse(state.closed)
            self.assertFalse(record.finished.is_set())
            self.assertTrue(torch.equal(snapshot.modalities.image.codes, brain.modalities.image.codes))
            self.assertIsNot(snapshot.liquid_state, brain.liquid_state)
            captured.append(state)
            return {"fixture": True}
        brain.fixture_decode = decode
        record = self.start_fixture_inline(worker, brain)
        self.assertTrue(record.future.result()["fixture"])
        self.assertTrue(captured[0].closed)
        self.assertIsNone(record.snapshot_state)
        self.assertTrue(record.finished.is_set())
        self.assertEqual([kind for kind, _data in notices], ["inline-imagination-finished"])
        self.assertEqual(notices[0][1]["data"]["requestId"], "rpc")
        worker._remove_inline_root(record)

    def test_inline_submit_failure_retains_exact_failed_artifact_and_closes_snapshot(self):
        def rejected(_function): raise RuntimeError("executor rejected owned artifact")
        worker, brain, notices = self.inline_fixture(submit=rejected)
        record = self.start_fixture_inline(worker, brain)
        self.assertIs(record, worker._inline_generations[("brain", "b" * 32)])
        self.assertTrue(record.finished.is_set())
        self.assertIsInstance(record.future.exception(), RuntimeError)
        self.assertIsNone(record.snapshot_state)
        self.assertFalse(record.staging_root.exists())
        self.assertEqual([kind for kind, _data in notices], ["inline-imagination-finished"])

    def test_inline_late_control_copy_failure_cleans_region_and_reports_original_error(self):
        worker, brain, notices = self.inline_fixture()
        class FailedConfig(SimpleNamespace):
            def __deepcopy__(self, _memo): raise RuntimeError("fixture metadata copy failure")
        brain.config = FailedConfig(vsa_dim=2)
        record = self.start_fixture_inline(worker, brain)
        self.assertIn("metadata copy failure", str(record.future.exception()))
        self.assertIsNone(record.snapshot_state)
        self.assertTrue(record.finished.is_set())
        self.assertFalse(record.staging_root.exists())
        self.assertEqual([kind for kind, _data in notices], ["inline-imagination-finished"])

    def test_three_production_callers_have_real_snapshot_lifetime_guards(self):
        for method in (AdaptiveBrain.chat, AdaptiveBrain.consolidate_pending_chat_learning, AdaptiveBrain._apply_supervised_dialogue):
            self.assertIn("@slow_snapshot_lifetime", inspect.getsource(method))
        self.assertNotIn("value.detach().cpu().clone()", inspect.getsource(AdaptiveBrain._snapshot_slow_transaction_state))
        source = inspect.getsource(Worker._start_inline_generation)
        self.assertIn("IsolatedModuleSnapshot", source)
        self.assertNotIn("copy.deepcopy(getattr(brain.modalities", source)

if __name__ == "__main__": unittest.main()
