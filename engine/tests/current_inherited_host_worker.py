"""Opt-in host transport fixture; no new model, curriculum or disk override."""
import gc
import os
from pathlib import Path
import sys
import threading
import sqlite3
from contextlib import closing, ExitStack
import weakref
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omni_core.brain import AdaptiveBrain
from omni_core.persistence import atomic_write_json, read_json
from capture_connected_execution_fixture import capture_current_seal_fixture
from capture_connected_frontier_reference import reference_current_frontier
import worker

ROOT = Path(os.environ['OMNI_CURRENT_INHERITED_HOST_ROOT']).resolve()
SOURCE = Path(os.environ['OMNI_CURRENT_INHERITED_SOURCE']).resolve()
if not ROOT.name.startswith('omni-connected-current-seal-') or not SOURCE.name.startswith('omni-connected-execution-proof-'):
    raise ValueError('current inherited proof requires its owned root and retained private native source')


class CurrentInheritedHostWorker(worker.Worker):
    def create(self, *_args, **_kwargs):
        raise AssertionError('current inherited proof forbids Worker.create/new neural initialization')

    def __init__(self):
        super().__init__()
        self._fixture_pause = None
        self._fixture_pause_waiting = None
        self._fixture_owner = None
        self._fixture_drop_ack = False
        self.methods.update(fixture_prepare_current=self.fixture_prepare,
            fixture_capture_current=self.fixture_capture, fixture_reference_current=self.fixture_reference,
            fixture_arm_pause=self.fixture_arm_pause, fixture_arm_lost_ack=self.fixture_arm_lost_ack,
            fixture_invariants=self.fixture_invariants)
        original = self.methods['resume_inherited_chat']
        def resume(params, request_id):
            result = original(params, request_id)
            if self._fixture_drop_ack and result.get('turnCommitted') is True:
                self._fixture_drop_ack = False
                atomic_write_json(ROOT / 'lost-final-ack.json', {'brainId': params['brainId'],
                    'executionId': params['executionId'], 'turnId': result['turnId'], 'turnCommitted': True})
                raise worker.RpcFault(-32099, 'One-use fixture lost final acknowledgement AFTER the real inherited commit')
            return result
        self.methods['resume_inherited_chat'] = resume

    def fixture_prepare(self, params, request_id):
        if self.brains or params:
            raise ValueError('metadata-only preparation precedes any model Load')
        return capture_current_seal_fixture(SOURCE, destination_folder=ROOT, root_name='joined', prepare_only=True)

    def _owner(self, params):
        if params != {'brainId': 'joined', 'storagePath': str(ROOT / 'joined')} or set(self.brains) != {'joined'}:
            raise ValueError('fixture computation needs the actual single Worker-owned resident root')
        self._brain_leases['joined'].assert_exclusive()
        return self.brains['joined']

    def _run_owned(self, request_id, operation):
        self._cooperative_cancel.clear()
        with self._active_request_lock:
            self._active_request = ('chat', str(request_id))
        try:
            return operation()
        finally:
            with self._active_request_lock:
                self._active_request = None

    def fixture_capture(self, params, request_id):
        brain = self._owner(params); self._fixture_owner = weakref.ref(brain)
        return self._run_owned(request_id, lambda: capture_current_seal_fixture(SOURCE,
            prepared_fixture=ROOT, root_name='joined', resident_brain=brain,
            cancelled=self._cooperative_cancel.is_set,
            progress=lambda stage, data: self.notify('fixture-progress', brain_id='joined', data={'stage': stage, **data})))

    def fixture_reference(self, params, request_id):
        brain = self._owner(params)
        return self._run_owned(request_id, lambda: reference_current_frontier(brain,
            cancelled=self._cooperative_cancel.is_set))

    def fixture_arm_pause(self, params, request_id):
        brain = self._owner(params)
        capture = read_json(ROOT / 'capture-receipt.json')
        self._fixture_pause = (brain.brain_id, capture['rootTurnId'], threading.Event())
        return {'armed': True}

    def request_chat_suspension(self, params):
        result = super().request_chat_suspension(params)
        pause = self._fixture_pause_waiting or self._fixture_pause
        if pause and result.get('requested') and (params.get('brainId'), params.get('streamId')) == pause[:2]:
            pause[2].set()
        return result

    def notify(self, kind, **fields):
        super().notify(kind, **fields)
        pause = self._fixture_pause
        if kind == 'chat-token' and pause and (fields.get('brain_id'), fields.get('stream_id')) == pause[:2]:
            self._fixture_pause = None
            self._fixture_pause_waiting = pause
            # Only transport handoff is held. A genuine host suspend request
            # releases it; no synthetic token, frontier or automatic Stop.
            try:
                if not pause[2].wait(10):
                    raise TimeoutError('host did not request suspension at the delivered token boundary')
            finally:
                self._fixture_pause_waiting = None

    def fixture_arm_lost_ack(self, params, request_id):
        if params or (ROOT / 'lost-final-ack.json').exists():
            raise ValueError('lost-ACK injection is one-use and has no neural operands')
        self._fixture_drop_ack = True
        return {'armed': True}

    def fixture_invariants(self, params, request_id):
        if params != {'brainId': 'joined', 'storagePath': str(ROOT / 'joined')}:
            raise ValueError('invariant read has another owner')
        gc.collect()
        metadata = read_json(ROOT / 'joined/engine/brain.json')
        brain = self.brains.get('joined')
        result = {'brainId': 'joined', 'pid': os.getpid(), 'resident': brain is not None,
            'ownerReleased': self._fixture_owner is None or self._fixture_owner() is None,
            'leaseReleased': 'joined' not in self._brain_leases,
            'counters': metadata['counters'], 'acceptedInputCount': len(metadata.get('accepted_chat_inputs', [])),
            'activityClock': metadata.get('activity_clock'), 'queue': metadata['connected_executions']}
        with closing(sqlite3.connect((ROOT / 'joined/engine/conversation.sqlite3').as_uri() + '?mode=ro', uri=True)) as ledger:
            result['humanCount'] = ledger.execute("SELECT COUNT(*) FROM entries WHERE kind='message' AND json_extract(payload_json,'$.role')='human'").fetchone()[0]
        if brain is not None:
            result['parameterChecksum'] = brain.parameter_checksum()
            result['resourceStatus'] = brain.resource_policy.status()
        return result


torch.set_num_threads(1)
worker.Worker = CurrentInheritedHostWorker
with ExitStack() as guards:
    for method in ('_train_ground_up_curriculum', 'train', 'ingest', 'consolidate_pending_chat_learning'):
        guards.enter_context(patch.object(AdaptiveBrain, method,
            side_effect=AssertionError('current inherited proof forbids curriculum/corpus/slow replay: ' + method)))
    guards.enter_context(patch.object(AdaptiveBrain, '_select_structured_actions', return_value=({'talk': 1., 'ponder': 0.}, [])))
    # Disclosed structural action choice only; native decoder, packed state,
    # input receipts, clock, finalization, RPC and policy remain real.
    raise SystemExit(worker.main())
