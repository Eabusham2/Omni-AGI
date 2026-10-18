"""NEW actual media-body/effects/ownership seams, constructor-free fixtures.

Tiny raw CPU tensors, inert decoder callbacks and temporary files only. No
brain/learned module constructor, packed kernel, Load, training, FFmpeg process,
provider, hardware benchmark or quality qualification is executed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
import weakref
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from test_operation_auto_boundaries import PACKAGE, ROOT, MODEL, MODAL, CONNECTED, empty_module, methods
from test_media_canvas_streaming import CANVAS, CODEC, RANDOM, PLAN, Policy, hub, plan

sys.path.insert(0, str(ROOT))
NATIVE = importlib.import_module(PACKAGE + ".media_whole_operation_native")
BOUNDARY = importlib.import_module(PACKAGE + ".media_artifact_operation")
CODEC_GATEWAY = importlib.import_module("codec_gateway")


class Owner:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


def source_body(root, *, modality="image", file=True):
    """Read the real AdaptiveBrain body without importing/constructing it."""
    class Cancelled(RuntimeError): pass
    namespace = {"__package__": PACKAGE, "torch": torch, "F": F, "os": os, "time": time, "hashlib": hashlib,
        "json": json, "uuid": uuid, "Path": Path, "Mapping": Mapping, "MediaCanvasView": CANVAS.MediaCanvasView, "subprocess": subprocess,
        "EncodedMediaFile": CODEC.EncodedMediaFile, "create_media_canvas": CANVAS.create_media_canvas,
        "write_media_payload": CODEC.write_media_payload, "payload_sha256": CODEC.payload_sha256,
        "encode_png_file": CODEC.encode_png_file, "encode_wav_file": CODEC.encode_wav_file, "encode_mp4_file": CODEC.encode_mp4_file,
        "inline_media_data_url": PLAN.inline_media_data_url, "MediaOutputRequest": PLAN.MediaOutputRequest,
        "ModalityGenerationCancelled": Cancelled, "_finite_number": lambda value, default: default if value is None else float(value),
        "MODALITY_DECODER_STEPS": {"vision": 1, "image": 4, "audio": 3, "video": 3},
        **{name: getattr(BOUNDARY, name) for name in ("media_artifact_directory", "observe_media_output", "observe_media_preview", "media_resource_pause")}}
    names = {"generate_modality", "_media_codec_directory", "_media_png_payload", "_media_wav_payload", "_media_mp4_payload"}
    methods("omni_core/brain.py", "AdaptiveBrain", names, namespace)
    policy, engine, cache = Policy(), Path(root)/"engine", Path(root)/"cache"
    policy.status = lambda **_kw: {"memoryPressure": False, "diskPressure": False}
    engine.mkdir(); cache.mkdir()
    config = types.SimpleNamespace(vision_enabled=False, image_enabled=True, audio_enabled=True, video_enabled=True,
        image_size=4, audio_samples=8, video_frames=2, modality_channels=2, disk_state_offload=True, hardware_tier="personal")
    value, events, prepared_calls, captures = hub(), [], [], []
    value.generate_scaled = types.MethodType(MODAL.ModalityHub.generate_scaled, value)
    brain = Owner(brain_id="owned", config=config, device=torch.device("cpu"), resource_policy=policy, engine_path=engine,
        _live_paging_cache_directory=cache, modalities=value, modality_training={"image": 1, "audio": 1, "video": 1},
        installed_modality_packs=[], counters={"inference_count": 0}, events=types.SimpleNamespace(append=lambda *args: events.append(args)),
        resource_pause={"original": True}, _modality_preview_budget=lambda _m: 4,
        _modality_idea_evidence=lambda *_args: {"source": "inert-cue", "activeAssemblyCount": 0, "promptProvided": False},
        _modality_idea=lambda *_args: torch.tensor([[.125, -.25]]))
    for name in names - {"generate_modality"}: setattr(brain, name, types.MethodType(namespace[name], brain))
    brain._png_bytes = lambda tensor: CODEC.encode_png_file(tensor, directory=cache, policy=policy)
    brain._apng_bytes = lambda tensor, fps=8: CODEC.encode_png_file(tensor, directory=cache, policy=policy, fps=fps)
    brain._wav_bytes = lambda tensor, sample_rate=16000: CODEC.encode_wav_file(tensor, directory=cache, policy=policy, sample_rate=sample_rate)
    brain._mp4_bytes = lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("inert fixture has no codec process"))
    @contextmanager
    def unavailable(_purpose):
        raise ImportError("inert fixture has no codec process")
        yield
    brain._video_runtime_executable = unavailable
    def prepare(kind, *_args, **_kw):
        prepared_calls.append(kind)
        return plan(kind, samples=6 if modality=="video" else 17, file=file), 1.0
    brain._prepare_media_output_plan = prepare
    native = object.__new__(NATIVE.NativeWholeMediaOperationOps); native._owner = weakref.ref(brain)
    native.host = types.SimpleNamespace(parent_cancelled=lambda: False)
    def boundary(owner, prepared, producer, preview, directory, cancelled, started):
        operation = {"prepared": prepared, "producer": producer, "previews": preview is not None}
        produced, baseline = native.execute_genuine(operation, preview, directory, cancelled, started)
        captures.append((operation, baseline)); return produced
    namespace["measured_media_artifact"] = boundary
    method = CANVAS.media_storage_lifetime(namespace["generate_modality"])
    return brain, method, native, events, prepared_calls, captures, Cancelled


class WholeMediaBoundaryCases(unittest.TestCase):
    def test_actual_capture_keeps_only_admitted_cue_and_new_seed_reuses_layout_not_previous_result(self):
        policy, epoch = Policy(), [1]
        policy.require_device = lambda *_args: None
        config = types.SimpleNamespace(image_size=4, audio_samples=8, video_frames=2)
        brain = Owner(brain_id="owned", config=config, core_pager=types.SimpleNamespace(_owners={}), resource_policy=policy)
        native = object.__new__(NATIVE.NativeWholeMediaOperationOps); native._owner = weakref.ref(brain)
        closure = {"modules": {}, "names": {}, "identities": ["inert-source"], "packed": []}
        native._closure = lambda _modalities: closure; native._scope = lambda _closure: "inert-epoch-scope"; native._codec = lambda: None
        native.host = types.SimpleNamespace(_state=lambda _closure, _scope, session: {"kind": "live-owner-epochs", "complete": True,
            "ownerSessionId": session, "bindingSha256": str(epoch[0])*64, "ownerVersions": {"inert-owner": epoch[0]}},
            _hash_tensor=lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest())
        review = {"binding": {"policySha256": "1"*64, "topologySha256": "2"*64, "budgetSha256": "3"*64}}
        cue = torch.tensor([[.125, -.25]])
        prepared = {"modality": "image", "idea": cue, "seed": 1, "settings": {"outputMode": "exact"}, "primaryPlan": plan("image"),
            "audioPlan": None, "companionAudio": False}
        first = native.capture_operation(prepared, lambda *_args: None, True, "session", review)
        second = native.capture_operation({**prepared, "seed": 2}, lambda *_args: None, True, "session", review)
        third = native.capture_operation({**prepared, "settings": {"outputMode": "exact", "fps": 9}}, lambda *_args: None, True, "session", review)
        self.assertEqual(first["stratumKey"], second["stratumKey"]); self.assertNotEqual(first["identity"]["workloadSha256"], second["identity"]["workloadSha256"])
        self.assertNotEqual(first["stratumKey"], third["stratumKey"])
        self.assertIsNot(first["prepared"]["idea"], cue); cue.add_(1)
        self.assertTrue(native.unchanged(first, "session")); epoch[0] += 1; self.assertFalse(native.unchanged(first, "session"))
        for operation in (first, second, third): native.release_operation(operation)
        self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_actual_complete_private_consumer_restores_scope_and_finishes_all_stages_with_real_files(self):
        with tempfile.TemporaryDirectory() as root:
            brain, generate, native, events, _preparation, captures, _cancelled = source_body(root)
            generated = generate(brain, "image", settings={"outputMode": "exact", "width": 6, "height": 5}, seed=11, preview_callback=lambda *_args: None)
            operation, baseline = captures[0]
            brain.modalities.modules = lambda: ()
            observations = {"real": "original"}
            brain.core_pager = types.SimpleNamespace(_observations=observations, _regional_limits=lambda: None)
            brain.working_attention_pager = types.SimpleNamespace(activate=lambda **_kwargs: nullcontext())
            native.admit_candidate = lambda _candidate: {}
            operation.update(closure={"modules": {}, "packed": []}, residency=[], ownerObjects={}, codec=None)
            stages = []
            trial = native.execute_private({"executionDevice": "cpu", "asyncDevices": []}, operation,
                types.SimpleNamespace(stage_completed=lambda name: stages.append(name)), threading.Event())
            self.assertEqual(trial.contract, baseline.contract); self.assertEqual(stages, NATIVE.MEDIA_STAGES)
            self.assertIs(brain.core_pager._observations, observations)
            self.assertEqual(list((brain._live_paging_cache_directory/"media-auto-private").iterdir()), [])
            self.assertEqual(len(events), 1); self.assertTrue(Path(generated["path"]).is_file())
            self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_actual_prepared_body_file_image_audio_video_keep_real_events_once_and_private_replay_is_equal(self):
        for modality in ("image", "audio", "video"):
            with self.subTest(modality=modality), tempfile.TemporaryDirectory() as root:
                brain, generate, native, events, preparation, captures, _cancelled = source_body(root, modality=modality)
                published = []
                settings = {"outputMode": "exact", "width": 6, "height": 5} if modality=="image" else (
                    {"outputMode": "exact", "durationMs": 1062.5, "sampleRate": 16} if modality=="audio" else
                    {"outputMode": "exact", "width": 6, "height": 5, "durationMs": 375, "fps": 8, "sampleRate": 16})
                result = generate(brain, modality, settings=settings, seed=8, preview_callback=lambda *args: published.append(args))
                self.assertEqual(preparation, [modality] + (["audio"] if modality=="video" else []))
                if modality == "video":
                    expected_steps = result["mediaOutputPlan"]["workUnits"]*3 + 3 + 3 + result["synchronizedAudio"]["mediaOutputPlan"]["workUnits"]*3
                    self.assertEqual(result["generationPerformance"]["decoderSteps"], expected_steps)
                self.assertEqual(len(events), 1); self.assertTrue(published)
                artifact, baseline = Path(result["path"]), captures[0][1]
                original = artifact.read_bytes(); self.assertEqual(hashlib.sha256(original).hexdigest(), result["artifactSha256"])
                self.assertIsNotNone(baseline.contract)
                sinks = []
                def replay():
                    with tempfile.TemporaryDirectory(dir=root) as private:
                        produced, actual = native._run(captures[0][0], None, Path(private), lambda: False, time.perf_counter(),
                            private=True, effects_out=sinks)
                        self.assertNotEqual(produced["result"]["path"], result["path"])
                        self.assertEqual(actual.contract, baseline.contract)
                    for sink in sinks: sink.close_files()
                CANVAS.media_storage_lifetime(replay)()
                self.assertEqual(artifact.read_bytes(), original); self.assertEqual(len(events), 1)
                self.assertEqual(brain.resource_pause, {"original": True})

    def test_private_cancel_cannot_delete_the_previously_completed_original_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            brain, generate, native, events, _preparation, captures, Cancelled = source_body(root)
            result = generate(brain, "image", settings={"outputMode": "exact", "width": 6, "height": 5}, seed=9)
            path = Path(result["path"]); expected = path.read_bytes()
            (Path(root)/"private").mkdir()
            with self.assertRaises((Cancelled, InterruptedError, MODAL.ModalityGenerationCancelled)):
                CANVAS.media_storage_lifetime(lambda: native._run(captures[0][0], None, Path(root)/"private", lambda: True, time.perf_counter(), private=True))()
            self.assertEqual(path.read_bytes(), expected); self.assertEqual(len(events), 1)

    def test_optional_evidence_budget_failure_does_not_cancel_original_generation(self):
        with tempfile.TemporaryDirectory() as root:
            brain, generate, _native, events, _preparation, captures, _cancelled = source_body(root)
            reserve = brain.resource_policy.reserve_ram
            def admitted(size, purpose):
                if purpose.startswith("bounded exact media"): raise RuntimeError("optional proof does not fit")
                return reserve(size, purpose)
            brain.resource_policy.reserve_ram = admitted
            result = generate(brain, "image", settings={"outputMode": "exact", "width": 6, "height": 5}, seed=10)
            self.assertTrue(Path(result["path"]).is_file()); self.assertIsNone(captures[0][1].contract); self.assertEqual(len(events), 1)

    def test_tensor_and_file_canvas_exact_neural_proofs_agree_without_flat_full_copy(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy()
            def run():
                canvas = CANVAS.MediaCanvas((1, 3, 2, 3), modality="image", dtype=torch.float32, directory=root, policy=policy)
                value = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 18
                canvas.add((slice(0, 1), slice(0, 3), slice(0, 2), slice(0, 3)), value, torch.ones((2, 3)))
                self.assertEqual(canvas.view().neural_sha256(), CODEC.TensorMediaView(value, "image").neural_sha256())
                with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                    CODEC.TensorMediaView(torch.tensor([[float("nan")]]), "audio").neural_sha256()
                with self.assertRaises(ValueError): canvas.output.read((slice(0, 1), slice(0, 3), slice(0, 2), slice(0, 1000000000)))
            CANVAS.media_storage_lifetime(run)()

    def test_pinned_codec_scope_reverifies_existing_binary_without_setup_or_fallback_and_rejects_changes(self):
        events, verified = [], []
        configuration = {"executablePath": "/fixture/unused", "artifactSha256": "a"*64, "binarySha256": "b"*64,
            "binarySizeBytes": 1, "target": "inert"}
        gateway = CODEC_GATEWAY.CodecRuntimeGateway(lambda *args: events.append(args), environment={},
            resolve_external=lambda _e: self.fail("private codec tried external discovery"), verify=lambda value, **_kw: verified.append(dict(value)))
        gateway.selected_path, gateway.selected_configuration, gateway.selected_identity = configuration["executablePath"], dict(configuration), ("unused", "b"*64)
        owner = CODEC_GATEWAY.CodecOwner("private-fixture", "owned")
        with gateway.pinned_scope(owner, lambda: False, configuration):
            with gateway.lease("owned", "encode-video") as selected: self.assertEqual(selected, "/fixture/unused")
        self.assertEqual(verified, [configuration]); self.assertEqual(events, []); self.assertEqual(gateway.leases, {})
        with gateway.pinned_scope(owner, lambda: False, configuration):
            gateway.selected_configuration = None
            with self.assertRaises(CODEC_GATEWAY.CodecRuntimeUnavailable):
                with gateway.lease("owned", "encode-video"): self.fail("changed runtime was executed")
        self.assertEqual(events, [])

    def test_complete_media_closure_visits_borrowed_connected_parents_and_rejects_foreign_and_overridden_owners(self):
        pager = types.SimpleNamespace(_owners={})
        config = types.SimpleNamespace(image_enabled=True, audio_enabled=False, video_enabled=False, to_dict=lambda: {"image": True})
        def packed():
            value = empty_module(MODEL.PackedAdaptiveBitLinear, attrs={"_native_core_pager": pager})
            pager._owners[value] = types.SimpleNamespace(loaded=True, pins=0, device=None)
            return value
        def leaf():
            return empty_module(MODAL.ModalityHub, children={"image": empty_module(MODAL.TinyImageImagination, children={"projection": packed()})}, attrs={"config": config})
        regions = [types.SimpleNamespace(modalities=leaf(), identity=types.SimpleNamespace(region_id=str(index), parent_id="parent", generation_sha256=str(index)*64)) for index in range(2)]
        exchange = empty_module(CONNECTED.RegionExchange, children={"bridge": packed()})
        decoder = types.SimpleNamespace(regions=regions, workspace_exchange=exchange)
        connected = empty_module(CONNECTED.ConnectedModalityHub, children={"selector": packed()}, attrs={"connected_decoder": decoder, "config": config, "widths": [2, 2]})
        brain = Owner(core_pager=pager, modalities=connected)
        native = object.__new__(NATIVE.NativeWholeMediaOperationOps); native._owner = weakref.ref(brain)
        closure = native._closure(("image",)); self.assertEqual(len(closure["packed"]), 4); self.assertEqual(len(closure["hubs"]), 3)
        parent_projection = regions[0].modalities.image.projection
        parent_projection._native_core_pager = object()
        with self.assertRaisesRegex(ValueError, "foreign"): native._closure(("image",))
        parent_projection._native_core_pager = pager; regions[1].modalities.image.generate = lambda *_args: None
        with self.assertRaisesRegex(ValueError, "callback"): native._closure(("image",))

    def test_scoped_placement_restores_controls_modes_and_observation_identity_even_after_failure(self):
        control = empty_module(torch.nn.Tanh, buffers={"value": torch.tensor([.25])})
        unrelated = empty_module(torch.nn.Tanh)
        packed = empty_module(MODEL.PackedAdaptiveBitLinear, buffers={"signs": torch.ones(1, dtype=torch.uint8)},
            attrs={"_native_compute_device": torch.device("cpu")})
        root = empty_module(MODAL.ModalityHub, children={"control": control, "unrelated": unrelated, "packed": packed})
        owner, observations = types.SimpleNamespace(loaded=True, device=None), {"original": "viewport"}
        pager = types.SimpleNamespace(_owners={packed: owner}, _lock=threading.RLock(), _observations=observations,
            _observe_mutations=lambda *_args: None, _regional_limits=lambda: None)
        brain = Owner(core_pager=pager, resource_policy=Policy(), modalities=root, device=torch.device("cpu"))
        native = object.__new__(NATIVE.NativeWholeMediaOperationOps); native._owner = weakref.ref(brain)
        native.admit_candidate = lambda _candidate: {}; original = control.value
        operation = {"closure": {"modules": {id(control): control, id(packed): packed}, "packed": [packed]},
            "ownerObjects": {packed: owner}, "residency": [(packed, None)]}
        with self.assertRaisesRegex(RuntimeError, "inert body failure"):
            with native._placement({"executionDevice": "cpu"}, operation, threading.Event(), private=True):
                self.assertIsNot(pager._observations, observations)
                root.eval(); control._buffers["value"] = torch.tensor([.75]); pager._observations["private"] = "not-published"
                raise RuntimeError("inert body failure")
        self.assertIs(control.value, original); self.assertIs(pager._observations, observations)
        self.assertTrue(root.training and control.training and unrelated.training and packed.training)
        self.assertEqual(brain.device, torch.device("cpu")); self.assertEqual(packed._native_compute_device, torch.device("cpu"))
