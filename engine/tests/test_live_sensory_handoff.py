"""Bounded stdlib/extracted contracts. No Worker/brain/model is constructed."""
import ast
import base64
import binascii
import contextvars
import hashlib
import importlib.util
import sys
import threading
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ENGINE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


queue = load("live_sensory_queue")
admission = load("stdio_request_admission")
worker = load("worker_live_sensory")


def extracted(path, name, namespace):
    tree = ast.parse((ENGINE / path).read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    code = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace[name]


class Lease:
    def __init__(self): self.releases = 0
    def mark_allocated(self, *_args): pass
    def release(self): self.releases += 1


class Fault(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.data = code, data


class Number:
    """One inert scalar; not a tensor library or model."""
    shape = (1, 1)
    def __init__(self, value): self.value = float(value.value if isinstance(value, Number) else value)
    def detach(self): return self
    def clone(self): return Number(self.value)
    def copy_(self, other): self.value = Number(other).value
    def float(self): return self
    def mean(self): return self
    def item(self): return self.value
    def __float__(self): return self.value
    def square(self): return Number(self.value ** 2)
    def sqrt(self): return Number(self.value ** .5)
    def to(self, _other): return self
    def reshape(self, *_): return self
    def clamp(self, minimum, maximum): return Number(min(maximum, max(minimum, self.value)))
    def __add__(self, other): return Number(self.value + Number(other).value)
    __radd__ = __add__
    def __sub__(self, other): return Number(self.value - Number(other).value)
    def __rsub__(self, other): return Number(Number(other).value - self.value)
    def __mul__(self, other): return Number(self.value * Number(other).value)
    __rmul__ = __mul__


def scalar_torch():
    return types.SimpleNamespace(Tensor=Number, float32="inert",
        no_grad=nullcontext, isfinite=lambda _value: types.SimpleNamespace(all=lambda: True),
        Generator=lambda **_: types.SimpleNamespace(manual_seed=lambda _seed: None),
        randn=lambda *_args, **_kwargs: Number(0))


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.current = True
        self.inbox = queue.LiveSensoryInbox("brain", "turn", "rpc", self.responses.append, lambda: self.current)
        self.permission = {"source": "camera", "granted": True, "scope": "session", "grantedAt": "2026-10-20T00:00:00Z"}
        self.binding = {"brainId": "brain", "captureRevision": 0,
            "permissionSha256": queue.capture_permission_hash(self.permission), "maxPacketBytes": 16, "maxInFlight": 2}

    def request(self, **change):
        payload = {"brainId": "brain", "storagePath": "/fixture/brain", "sessionId": "capture",
            "captureRevision": 0, "permissionSha256": self.binding["permissionSha256"], "modality": "image",
            "sequence": 1, "timestampMs": 1.0, "mimeType": "image/png", "dataBase64": "YWJj",
            "packetSha256": hashlib.sha256(b"abc").hexdigest(), "settings": {}}
        payload.update(change)
        return {"jsonrpc": "2.0", "id": "packet", "method": queue.LIVE_SENSORY_CONTROL,
            "params": {"brainId": "brain", "streamId": "turn", "operation": "observe_packet", "payload": payload}}

    def test_admission_is_immutable_and_not_a_consumed_ack(self):
        request, lease = self.request(), Lease()
        item = self.inbox.offer(request, lease, self.binding)
        request["params"]["payload"]["settings"]["width"] = 999
        self.assertEqual(item.payload()["settings"], {})
        self.assertEqual(self.responses, [])
        self.assertEqual(lease.releases, 0)
        self.inbox.finish(self.inbox.take(), result={"integrated": True})
        self.assertTrue(self.responses[0]["result"]["integrated"])
        self.assertEqual(lease.releases, 1)
        self.assertEqual(self.inbox.session_bytes, {})

    def test_regular_rpc_allocation_remains_charged_until_consumption(self):
        lease = Lease()
        receipt = admission.RequestAllocationReceipt([lease])
        retained = receipt.retain()
        item = self.inbox.offer(self.request(), retained, self.binding)
        receipt.release()  # the reader finished, not the consumer
        self.assertEqual(lease.releases, 0)
        self.inbox.finish(self.inbox.take(), result={"integrated": True})
        self.inbox.finish(item, result={"integrated": True})
        self.assertEqual(lease.releases, 1)
        self.assertEqual(len(self.responses), 1)

    def test_unadmitted_or_cross_owner_requests_cannot_enter(self):
        with self.assertRaisesRegex(ValueError, "ordinary admitted RAM"):
            self.inbox.offer(self.request(), None, self.binding)
        wrong = self.request()
        wrong["params"]["streamId"] = "other-turn"
        with self.assertRaisesRegex(ValueError, "another brain/turn"):
            self.inbox.offer(wrong, Lease(), self.binding)
        with self.assertRaisesRegex(ValueError, "revision changed"):
            self.inbox.offer(self.request(captureRevision=1), Lease(), self.binding)
        with self.assertRaisesRegex(ValueError, "revision changed"):
            self.inbox.offer(self.request(permissionSha256="f" * 64), Lease(), self.binding)
        self.assertEqual(len(self.inbox.pending), 0)

    def test_packet_bytes_and_live_budget_are_not_bypassed(self):
        with self.assertRaisesRegex(ValueError, "byte envelope"):
            self.inbox.offer(self.request(dataBase64="A" * 25), Lease(), self.binding)
        self.inbox.session_bytes["capture"] = 10 ** 8
        with self.assertRaisesRegex(ValueError, "transport RAM budget"):
            self.inbox.offer(self.request(), Lease(), self.binding)
        self.assertEqual(self.responses, [])

    def test_session_and_turn_retirement_reject_release_without_neural_use(self):
        lease = Lease()
        self.inbox.offer(self.request(), lease, self.binding)
        self.inbox.retire_session("capture")
        self.assertEqual(self.responses[0]["error"]["code"], -32800)
        self.assertEqual(lease.releases, 1)
        self.assertIsNone(self.inbox.take())
        with self.assertRaisesRegex(ValueError, "session has retired"):
            self.inbox.offer(self.request(), Lease(), self.binding)
        self.current = False
        self.inbox.close()
        with self.assertRaisesRegex(ValueError, "turn has retired"):
            self.inbox.offer(self.request(), Lease(), self.binding)

    def host(self):
        case = self
        class Host(worker.WorkerLiveSensory):
            def __init__(self):
                self._active_request_lock = threading.RLock(); self._steering_lock = threading.RLock()
                self._observation_lock = threading.RLock(); self._cooperative_cancel = threading.Event()
                self._active_request = ("chat", "rpc")
                self.session = types.SimpleNamespace(request_id="rpc", claimed=True, generation_open=True,
                    requested=threading.Event(), suspension_requested=threading.Event(), suspension_freezing=False)
                self._chat_steering = {("brain", "turn"): self.session}
                self.capture = types.SimpleNamespace(session_id="capture", brain_id="brain", capture_revision=0,
                    max_packet_bytes=16, max_in_flight=2, permission=case.permission, retired=threading.Event(),
                    modalities=("image",), retention="neural", last_sequence=-1, last_timestamp_ms=-1.,
                    packets_accepted=0, bytes_accepted=0)
                self._observation_sessions = {"capture": self.capture}
                self.sent = []; self._send = self.sent.append
                self.initialize_live_sensory()
            def _observation_session_payload(self, capture):
                return {"id": capture.session_id, "packetsAccepted": capture.packets_accepted}
        Host._decode_observation_packet = extracted("worker.py", "_decode_observation_packet", {
            "base64": base64, "binascii": binascii, "RpcFault": Fault, "_number": float,
            "capture_permission_hash": queue.capture_permission_hash})
        host = Host()
        return host, host.open_live_sensory(("brain", "turn"), "rpc")

    def barrier(self):
        self.private = contextvars.ContextVar("synthetic-private-operation", default=False)
        namespace = {"contextmanager": contextmanager, "PRIVATE_OPERATION": self.private,
            "_WORLD_BARRIER": contextvars.ContextVar("synthetic-world-depth", default=0)}
        actual = contextmanager(extracted("omni_core/whole_path_scheduler.py", "world_effect_barrier", namespace))
        module = types.ModuleType("omni_core.whole_path_scheduler")
        module.world_effect_barrier = actual
        parent = types.ModuleType("omni_core"); parent.__path__ = []
        return patch.dict(sys.modules, {"omni_core": parent, "omni_core.whole_path_scheduler": module})

    def test_actual_control_and_conditioning_callback_defer_ack_until_owner_integration(self):
        host, inbox = self.host()
        lease = Lease(); receipt = admission.RequestAllocationReceipt([lease])
        request = admission.AdmittedRequest(self.request(), receipt)
        self.assertIsNone(host.live_sensory_control(request))
        receipt.release()
        self.assertEqual(host.sent, [])
        self.assertEqual(lease.releases, 0)
        order = []
        class Brain:
            storage_path = Path("/fixture/brain")
            resource_policy = types.SimpleNamespace(reserve_ram=lambda *_: Lease())
            def condition_live_packet(self, current, **packet):
                order.append("actual-inert-conditioner")
                self.assert_episode = packet["episode_assembly_id"]
                return {"assemblyId": packet["episode_assembly_id"], "packetSha256": hashlib.sha256(packet["payload"]).hexdigest()}, current + 7
        brain = Brain()
        class Provider:
            def __call__(self): return []
            def live_sensory_conditioning(self, current, step, episode):
                return host.drain_live_sensory(inbox, brain, brain, current, step, episode)
        callback_factory = extracted("omni_core/brain.py", "_generation_observation_callback", {})
        with self.barrier():
            result = callback_factory(object(), Provider(), "real-current-episode")(3, 9)
        self.assertEqual(result["memoryBias"], 10)
        self.assertEqual(brain.assert_episode, "real-current-episode")
        self.assertEqual(order, ["actual-inert-conditioner"])
        self.assertEqual(host.sent[0]["result"]["observation"]["assemblyId"], "real-current-episode")
        self.assertEqual(result["observationBindings"][0]["observationKind"], "live-sensory")
        self.assertNotIn("toolId", result["observationBindings"][0])
        self.assertEqual(lease.releases, 1)
        model = (ENGINE / "omni_core/model.py").read_text()
        self.assertLess(model.index("memory_bias = refined.detach()"), model.index("def ordinary_cached", model.index("memory_bias = refined.detach()")))
        self.assertIn("inference_cache.close()", model[model.index("memory_bias = refined.detach()"):model.index("def ordinary_cached", model.index("memory_bias = refined.detach()"))])

    def test_private_world_cannot_take_or_ack_a_capture(self):
        host, inbox = self.host(); lease = Lease()
        inbox.offer(self.request(), lease, self.binding)
        with self.barrier():
            token = self.private.set(True)
            try:
                with self.assertRaisesRegex(RuntimeError, "private operation"):
                    host.drain_live_sensory(inbox, object(), object(), 0, 0, "episode")
            finally: self.private.reset(token)
        self.assertEqual(len(inbox.pending), 1)
        self.assertEqual(host.sent, [])
        inbox.close()
        self.assertEqual(lease.releases, 1)

    def test_failed_sensory_admission_releases_already_dequeued_tool_observation_leases(self):
        released = []
        observation = types.SimpleNamespace(release=lambda: released.append(True))
        class Provider:
            def __call__(self): return [observation]
            def live_sensory_conditioning(self, *_):
                raise worker.LiveSensoryIntegrationFailed("inert actual-owner admission failure")
        factory = extracted("omni_core/brain.py", "_generation_observation_callback", {})
        with self.assertRaises(worker.LiveSensoryIntegrationFailed):
            factory(object(), Provider(), "episode")(3, 0)
        self.assertEqual(released, [True])

    def test_reply_controls_retire_once_when_actual_capture_cleanup_delivery_raises(self):
        host, inbox = self.host(); lease = Lease()
        inbox.offer(self.request(), lease, self.binding)
        def broken_delivery(_): raise BrokenPipeError("inert disconnected cleanup writer")
        inbox.send = broken_delivery
        controls = types.SimpleNamespace(active=True)
        brain = types.SimpleNamespace(_native_reply_private_flags=controls)
        actual_retire = extracted("omni_core/reply_execution_planner.py", "retire_reply_controls", {})
        retirements = []
        def retire(owner, value):
            retirements.append((owner, value))
            actual_retire(owner, value)
        tree = ast.parse((ENGINE / "worker.py").read_text())
        chat = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_chat")
        actual_finally = next(node.finalbody for node in ast.walk(chat) if isinstance(node, ast.Try) and
            any(isinstance(child, ast.Call) and ast.unparse(child.func) == "self.close_live_sensory"
                for statement in node.finalbody for child in ast.walk(statement)))
        function = ast.parse("def cleanup(self, control_key, live_sensory, brain, reply_controls):\n    pass").body[0]
        function.body = actual_finally
        namespace = {"retire_reply_controls": retire}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "extracted-worker-cleanup", "exec"), namespace)
        with self.assertRaisesRegex(BrokenPipeError, "disconnected cleanup"):
            namespace["cleanup"](host, ("brain", "turn"), inbox, brain, controls)
        self.assertTrue(inbox.closed)
        self.assertNotIn(("brain", "turn"), host._live_sensory)
        self.assertEqual(lease.releases, 1)
        self.assertEqual(retirements, [(brain, controls)])
        self.assertFalse(controls.active)
        self.assertFalse(hasattr(brain, "_native_reply_private_flags"))

    def test_changed_revision_or_hash_is_rejected_before_conditioning(self):
        for change in ("revision", "hash"):
            host, inbox = self.host(); lease = Lease()
            inbox.offer(self.request(**({"packetSha256": "f" * 64} if change == "hash" else {})), lease, self.binding)
            if change == "revision": host.capture.capture_revision = 1
            brain = types.SimpleNamespace(storage_path=Path("/fixture/brain"))
            with self.barrier(): host.drain_live_sensory(inbox, brain, brain, 0, 0, "episode")
            self.assertIn("error", host.sent[0])
            self.assertFalse(host.sent[0]["error"]["data"]["packetConsumed"])
            self.assertEqual(host.capture.packets_accepted, 0)
            self.assertEqual(lease.releases, 1)

    def test_episode_policy_reuses_ordinary_paths_without_per_frame_optimizer(self):
        tree = ast.parse((ENGINE / "omni_core/brain.py").read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_integrate_live_packet")
        source = ast.unparse(method)
        self.assertIn("retention == 'neural' and (not episode_assembly_id) and seed_episode", source)
        for name in ("_admit_sensory_embedding", "_append_working_memory", "_append_replay", "_organic_replay_priority", "_settle_memory_automatically"):
            self.assertIn(name, source)
        self.assertNotIn("selector_optimizer", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("'working-only'", source)
        worker_source = (ENGINE / "worker.py").read_text()
        self.assertIn("episode_assembly_id=session.episode_assembly_id", worker_source)
        self.assertIn("session.episode_assembly_id = str(observation.get(\"assemblyId\") or \"\")", worker_source)
        self.assertIn("context[\"experience\"].get(\"assembly_id\"", (ENGINE / "omni_core/brain.py").read_text())

    def test_effects_unknown_after_actual_replay_mutation_aborts_instead_of_false_ack(self):
        host, inbox = self.host(); lease = Lease()
        inbox.offer(self.request(), lease, self.binding)
        effects, metadata = [], []
        population = types.SimpleNamespace(membrane=Number(4), spike_count=Number(6))
        class Brain:
            storage_path = Path("/fixture/brain")
            config = types.SimpleNamespace(liquid_dynamics=False)
            memory = types.SimpleNamespace(assembly_by_id={"episode": object()})
            liquid_state = Number(2)
            router = types.SimpleNamespace(population=population,
                route=lambda *_args, **_kwargs: (Number(5), {"spike_rate": .3}))
            resource_policy = types.SimpleNamespace(reserve_ram=lambda *_: metadata.append(Lease()) or metadata[-1])
            def _live_visual_embedding(self, *_): return Number(5), {}
            def _sensory_substrate_vector(self, *_): return Number(5)
            def _idea_model_vector(self, value): return value
            def liquid(self, *_args, **_kwargs):
                effects.append("liquid")
                return Number(7), {"ponder_scale": Number(2), "threshold_offset": Number(0)}
            def _organic_replay_priority(self, **_): return .5
            def _append_replay(self, *_args, **_kwargs): effects.append("replay")
            def _append_working_memory(self, *_args, **_kwargs): effects.append("working")
            def _settle_memory_automatically(self, *_args, **_kwargs): effects.append("settling")
            def idea_adapter(self, _): raise ValueError("inert adapter failure AFTER real extracted settling call")
        actual = extracted("omni_core/brain.py", "_integrate_live_packet", {
            "torch": scalar_torch(), "hashlib": hashlib, "IMAGINATION_MODALITIES": {"image", "audio", "video"},
            "F": types.SimpleNamespace(cosine_similarity=lambda *_: Number(0), normalize=lambda value, **_: value)})
        brain = Brain()
        brain.condition_live_packet = lambda current, **packet: actual(brain, current, seed_episode=False, **packet)
        with self.barrier(), self.assertRaises(worker.LiveSensoryIntegrationFailed):
            host.drain_live_sensory(inbox, brain, brain, Number(3), 0, "episode")
        self.assertEqual(effects, ["liquid", "replay", "working", "settling"])
        receipt = host.sent[0]["error"]["data"]
        self.assertIsNone(receipt["packetConsumed"])
        self.assertEqual(receipt["effectDisposition"], "unknown")
        self.assertFalse(receipt["automaticRetryAllowed"])
        self.assertTrue(inbox.closed)
        self.assertEqual(host.capture.packets_accepted, 0)
        self.assertEqual(lease.releases, 1)
        self.assertEqual(metadata[0].releases, 1)
        source = (ENGINE / "worker.py").read_text()
        self.assertIn("elif execution_session is None:\n                self._restore_committed_brain_after_failed_chat(", source)

    def test_unknown_inherited_effects_fence_real_root_instead_of_claiming_view_close_rolls_back(self):
        tree = ast.parse((ENGINE / "worker.py").read_text())
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and
            any(isinstance(item, ast.Attribute) and item.attr == "_connected_recovery_required" for item in ast.walk(node)))
        function = ast.parse("def failed_turn(self, brain, authority_brain, execution_session, error):\n    pass").body[0]
        function.body = [ast.Try(body=[ast.Raise(exc=ast.Name(id="error", ctx=ast.Load()))],
            handlers=[handler], orelse=[], finalbody=[])]
        namespace = {"LiveSensoryIntegrationFailed": worker.LiveSensoryIntegrationFailed,
            "prior_ledger_head": None, "turn_id": "turn", "input_sha256": "fixture-input"}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "extracted-failed-owner", "exec"), namespace)
        restored = []
        host = types.SimpleNamespace(_restore_committed_brain_after_failed_chat=lambda *_args, **_kwargs: restored.append(True))
        root, view = types.SimpleNamespace(_connected_recovery_required=False), object()
        with self.assertRaises(worker.LiveSensoryIntegrationFailed):
            namespace["failed_turn"](host, view, root, object(), worker.LiveSensoryIntegrationFailed("partial regional effects"))
        self.assertTrue(root._connected_recovery_required)
        self.assertEqual(restored, [])
        with self.assertRaises(worker.LiveSensoryIntegrationFailed):
            namespace["failed_turn"](host, root, root, None, worker.LiveSensoryIntegrationFailed("partial normal effects"))
        self.assertEqual(restored, [True])
        source = (ENGINE / "worker.py").read_text()
        self.assertIn('if getattr(existing, "_connected_recovery_required", False):', source)
        ordinary_observe = ast.unparse(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "observe_packet"))
        self.assertIn("self._get(", ordinary_observe)
        self.assertNotIn("self.brains.get(", ordinary_observe)

    def test_unknown_effects_still_fence_borrowed_owner_and_release_all_leases_if_error_delivery_fails(self):
        host, inbox = self.host(); leases = [Lease(), Lease()]; metadata = []
        inbox.offer(self.request(), leases[0], self.binding)
        inbox.offer(self.request(sequence=2, timestampMs=2), leases[1], self.binding)
        def broken_delivery(_): raise BrokenPipeError("inert disconnected response writer")
        inbox.send = broken_delivery
        def mutate_then_fail(*_args, **_kwargs): raise ValueError("inert partial-effect failure")
        brain = types.SimpleNamespace(storage_path=Path("/fixture/brain"), condition_live_packet=mutate_then_fail,
            resource_policy=types.SimpleNamespace(reserve_ram=lambda *_: metadata.append(Lease()) or metadata[-1]))
        root = types.SimpleNamespace(storage_path=Path("/fixture/brain"), _connected_recovery_required=False)
        with self.barrier(), self.assertRaises(worker.LiveSensoryIntegrationFailed):
            host.drain_live_sensory(inbox, brain, root, Number(3), 0, "episode")
        self.assertTrue(root._connected_recovery_required)
        self.assertTrue(inbox.closed)
        self.assertEqual([lease.releases for lease in leases], [1, 1])
        self.assertEqual(metadata[0].releases, 1)

    def test_pending_and_used_sensory_provenance_survives_actual_cursor_capture_without_reconsume(self):
        boundary_module = load("omni_core/live_sensory_conditioning")
        consumed = []
        binding = {"observationKind": "live-sensory", "observationId": "a" * 64,
            "episodeAssemblyId": "episode", "generationStep": 0, "packetSha256": "b" * 64}
        class Provider:
            def __call__(self): return []
            def live_sensory_pending(self): return not consumed
            def live_sensory_conditioning(self, current, step, episode):
                consumed.append(episode)
                return {"memoryBias": current + 7, "observationBindings": [binding]}
        provider = Provider()
        boundary = boundary_module.SensoryTurnBoundary(provider, "episode")
        current = boundary.poll(Number(3), 0)["memoryBias"]
        callback = extracted("omni_core/brain.py", "_generation_observation_callback", {})(object(), provider,
            "episode", sensory_boundary=boundary)
        retained = callback.suspension_conditioning(current)
        self.assertEqual(consumed, ["episode"])
        self.assertEqual(retained["observationBindings"], [binding])
        self.assertEqual(boundary.bindings, [])
        class Cursor:
            def __init__(self, *values): self.metadata, self.resumed_conditioning = values[11:13]
        decoder = types.SimpleNamespace(last_generation_sensory_observations=[binding],
            last_generation_observations=[binding], last_generation_tool_observations=[])
        capture = extracted("omni_core/generation_cursor.py", "capture", {})
        cursor = capture(Cursor, generated=Number(1), input_length=1, memory_bias=current,
            cache=None, entropies=[], generator=types.SimpleNamespace(get_state=lambda: Number(2)),
            text_boundaries=None, last_text_boundary=1, emission_start=1, token_budget=2,
            options={}, decoder=decoder, resumed_conditioning=retained)
        self.assertEqual(cursor.metadata["last_generation_observations"], [binding])
        self.assertEqual(cursor.metadata["last_generation_sensory_observations"], [binding])
        self.assertEqual(cursor.resumed_conditioning["memoryBias"].value, 10)
        self.assertEqual(cursor.resumed_conditioning["observationBindings"], [binding])
        self.assertEqual(consumed, ["episode"])
        source = (ENGINE / "omni_core/model.py").read_text()
        self.assertIn('retained_conditioning = take_pending(memory_bias) or None', source)
        self.assertIn('resumed_conditioning=retained_conditioning', source)
        self.assertIn('conditioning = resumed_conditioning or conditioning_callback(memory_bias, step)', source)

    def test_world_search_boundary_integrates_before_private_search_finishes_and_replay_binds_once(self):
        native = load("omni_core/best_work")
        boundary_module = load("omni_core/live_sensory_conditioning")
        host, inbox = self.host(); order = []; metadata = []
        class Brain:
            storage_path = Path("/fixture/brain")
            resource_policy = types.SimpleNamespace(reserve_ram=lambda *_: metadata.append(Lease()) or metadata[-1])
            def condition_live_packet(self, current, **packet):
                self.assert_world = not case.private.get()
                order.append("actual-owner-integrated")
                return {"assemblyId": "episode", "packetSha256": hashlib.sha256(packet["payload"]).hexdigest()}, current + 7
        case, brain = self, Brain()
        provider = types.SimpleNamespace(live_sensory_pending=inbox.has_pending,
            live_sensory_conditioning=lambda current, step, episode: host.drain_live_sensory(inbox, brain, brain, current, step, episode))
        boundary = boundary_module.SensoryTurnBoundary(provider, "episode")
        current = [3]
        def world():
            self.assertFalse(self.private.get())
            received = boundary.poll(current[0], 0)
            current[0] = received.get("memoryBias", current[0])
        class Session:
            done = False; mean_log_probability = -.2
            count = 0
            def advance(self):
                token = case.private.set(True)
                try:
                    self.count += 1
                    if self.count == 1:
                        inbox.offer(case.request(), Lease(), case.binding)
                        order.append("private-candidate-first-step")
                    else:
                        case.assertEqual(current[0], 10)
                        case.assertTrue(host.sent)
                        order.append("private-candidate-final-step")
                        self.done = True
                    return {"step": self.count, "entropy": .1}
                finally: case.private.reset(token)
            def close(self): pass
        with self.barrier():
            result = native.run_best_work(seed=3, ordinary_temperature=.65, normalized_entropy=.1,
                exploration_budget=1, start=lambda _: (Session(), {}, lambda: None),
                finish=lambda *_: {"text": "inert candidate"}, cancelled=lambda: False, steered=lambda: False,
                world_boundary=world)
            self.assertEqual(order, ["private-candidate-first-step", "actual-owner-integrated", "private-candidate-final-step"])
            self.assertTrue(brain.assert_world)
            self.assertEqual(result["candidateHostEffects"], 0)
            visible = boundary.condition(current[0], 0)
            self.assertEqual(visible["memoryBias"], 10)
            self.assertEqual(len(visible["observationBindings"]), 1)
            self.assertEqual(boundary.condition(current[0], 1), {})
        self.assertEqual(metadata[0].releases, 0)
        inbox.close()
        self.assertEqual(metadata[0].releases, 1)

    def test_canonical_ponder_admits_live_frame_and_preserves_new_world_router_baseline(self):
        boundary_module = load("omni_core/live_sensory_conditioning")
        host, inbox = self.host()
        case, order = self, []
        population = types.SimpleNamespace(membrane=Number(4), spike_count=Number(6))
        class Scores:
            def __init__(self, selected): self.selected = selected
            def detach(self): return self
            def float(self): return self
            def reshape(self, *_): return self
            def mean(self, **_): return self
            def argmax(self): return Number(self.selected)
            def __getitem__(self, index): return Number(.9 if index == self.selected else .1)
        class Decoder:
            count = 0
            def internal_action_policy(self, current):
                self.count += 1
                if self.count == 1:
                    inbox.offer(case.request(), Lease(), case.binding)
                else:
                    order.append("post-sensory-ponder-step")
                return Scores(0 if self.count == 1 else 1)
        class Brain:
            storage_path = Path("/fixture/brain")
            config = types.SimpleNamespace(liquid_dynamics=False, spiking_dynamics=True)
            liquid_state = Number(2)
            decoder = Decoder()
            resource_policy = types.SimpleNamespace(status=lambda: {}, reserve_ram=lambda *_: Lease())
            idea_adapter = staticmethod(lambda value: value + 1)
            def condition_live_packet(self, current, **packet):
                case.assertEqual(population.membrane.value, 4)
                case.assertEqual(population.spike_count.value, 6)
                population.membrane.copy_(Number(20)); population.spike_count.copy_(Number(40))
                self.liquid_state = Number(9)
                order.append("actual-owner-during-ponder")
                return {"assemblyId": "episode", "packetSha256": hashlib.sha256(packet["payload"]).hexdigest()}, current + 7
        def route(driven, **_):
            population.membrane.copy_(population.membrane + 1)
            population.spike_count.copy_(population.spike_count + 1)
            return driven, {}
        brain = Brain(); brain.router = types.SimpleNamespace(population=population, route=route)
        provider = types.SimpleNamespace(live_sensory_pending=inbox.has_pending,
            live_sensory_conditioning=lambda current, step, episode: host.drain_live_sensory(inbox, brain, brain, current, step, episode))
        boundary = boundary_module.SensoryTurnBoundary(provider, "episode")
        method = extracted("omni_core/brain.py", "_native_pre_speech_ponder", {
            "torch": scalar_torch(), "time": types.SimpleNamespace(perf_counter=lambda: 0., sleep=lambda _: None),
            "ACTION_KINDS": ["ponder", "talk"], "F": types.SimpleNamespace(softmax=lambda value, **_: value),
            "ChatGenerationCancelled": RuntimeError})
        with self.barrier():
            result, trace = method(brain, Number(3), Number(2), seed=1, confidence=.7, noise=0,
                sensory_boundary=boundary)
        self.assertEqual(trace["passes"], 2)
        self.assertEqual(order, ["actual-owner-during-ponder", "post-sensory-ponder-step"])
        self.assertEqual(population.membrane.value, 20)
        self.assertEqual(population.spike_count.value, 40)
        self.assertTrue(host.sent)
        self.assertEqual(boundary.condition(result, 0)["memoryBias"].value, result.value)
        inbox.close()


if __name__ == "__main__": unittest.main()
