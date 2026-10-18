"""Executable inherited frontiers, not model/corpus initialization.

The actual native proof explicitly requires an already initialized genuine
fixture. Ordinary CI covers only the small closed-protocol admission cases.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.connected_execution import (ConnectedExecutionQueue, ConnectedExecutionSession,
    connected_execution_status, require_joint_mutation_allowed, validated_connected_executions,
    stop_unloaded_connected_execution)
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.full_merge import assemble_full_merge
from omni_core.offload import NeuralStateResourcePause, ResourcePolicy
from worker import Worker


def entry():
    return {'id': 'a' * 64, 'rootTurnId': 'ancestral-' + 'b' * 64, 'parentId': 'parent',
        'regionPath': ['region'], 'sourceEnginePath': 'merge/parents/parent/engine',
        'sourceCheckpointId': 'c' * 32, 'sourceManifestSha256': 'd' * 64, 'sourceBindingId': 'e' * 64,
        'originalTurnId': 'original-turn', 'historyParentId': 'parent', 'historyHumanMessageId': 'human',
        'rootHumanMessageId': 'mapped-human', 'status': 'queued', 'backend': 'cpu', 'lineage': []}


def queue(value):
    return {'format': 'omni-connected-execution-queue', 'formatVersion': 1, 'entries': value}


class ConnectedExecutionProtocolTests(unittest.TestCase):
    def test_new_adoption_requires_supported_frontier_but_archival_inventory_does_not_claim_support(self):
        from omni_core.connected_execution import suspended_execution_identity, execution_checkpoint_snapshot_entries
        historical = {'status': 'suspended', 'brainId': 'parent', 'checkpointId': 'a' * 32,
            'files': {}, 'journal': {}}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(ExecutionCheckpointStore, 'current', return_value=historical), \
                patch.object(ExecutionCheckpointStore, 'require_supported_frontier',
                    side_effect=ValueError('unsupported saved runtime')) as qualify:
            with self.assertRaisesRegex(ValueError, 'unsupported saved runtime'):
                suspended_execution_identity(Path(folder), 'parent')
            qualify.assert_called_once_with(historical)
            with patch('omni_core.connected_execution._verify_files') as authenticate, \
                    patch('omni_core.connected_execution._inside', side_effect=ValueError('inventory boundary')):
                with self.assertRaisesRegex(ValueError, 'inventory boundary'):
                    list(execution_checkpoint_snapshot_entries(Path(folder), brain_id='parent'))
                authenticate.assert_called_once_with(Path(folder).resolve(), historical, require_supported=False)

    def test_resuspension_publishes_only_its_new_paused_clock_without_neural_save(self):
        from omni_core.activity_clock import ActivityClock
        from omni_core.connected_execution import persist_borrowed_generation_clock
        from omni_core.execution_checkpoint import binding_for
        from omni_core.persistence import atomic_write_json
        with tempfile.TemporaryDirectory() as folder:
            engine = Path(folder)
            clock = ActivityClock(4., paused=True)
            brain = SimpleNamespace(brain_id='parent', engine_path=engine,
                mutable_state_manifest={'generation': 'mutable'},
                memory=SimpleNamespace(persistence_manifest={'generation': 'graph'}),
                parameter_checksum=lambda: '1' * 64, _attention_epoch=lambda: 3,
                _owned_activity_clock=lambda: clock)
            original = {'brain_id': 'parent', 'mutable_state': brain.mutable_state_manifest,
                'substrate': {'persistence': brain.memory.persistence_manifest},
                'fresh_attention_boundary': {'epoch': 3}, 'activity_clock': ActivityClock(1., paused=True).snapshot(),
                'untouched': {'source': 'exact'}}
            atomic_write_json(engine / 'brain.json', original)
            persist_borrowed_generation_clock(brain, 'turn', binding_for(brain, 'turn'))
            saved = json.loads((engine / 'brain.json').read_text())
            self.assertEqual(saved, {**original, 'activity_clock': clock.snapshot()})
            clock.resume()
            with self.assertRaisesRegex(ValueError, 'paused activity clock'):
                persist_borrowed_generation_clock(brain, 'turn', binding_for(brain, 'turn'))
            clock.pause()
            with self.assertRaisesRegex(RuntimeError, 'mutated learned state'):
                persist_borrowed_generation_clock(brain, 'turn', 'f' * 64)

    def test_cold_working_beforeimage_recovers_only_uncommitted_root_and_preserves_commit(self):
        from contextlib import nullcontext
        from omni_core.connected_working_publication import ConnectedWorkingPublication, recover_inherited_working_publication, MARKER
        from omni_core.offload import PagedWorkingMemory
        from omni_core.persistence import atomic_write_json
        with tempfile.TemporaryDirectory() as folder:
            engine = Path(folder) / 'engine'; engine.mkdir()
            policy = SimpleNamespace(require_disk=lambda *_args: None, reserve_ram=lambda *_args: nullcontext())
            pages = PagedWorkingMemory(engine / 'state/working-memory.sqlite3', policy)
            pages.append(torch.tensor([1., 2.]), {'assemblyId': 'old'})
            before = pages.checkpoint()
            metadata = {'brain_id': 'root', 'connected_executions': queue([entry()])}
            atomic_write_json(engine / 'brain.json', metadata)
            root = SimpleNamespace(brain_id='root', engine_path=engine, paged_working_memory=pages, resource_policy=policy)
            publication = ConnectedWorkingPublication(root, entry()['id']); publication.stage()
            pages.clear(); pages.append(torch.tensor([9., 8.]), {'assemblyId': 'new'})
            recovered = recover_inherited_working_publication(engine)
            self.assertTrue(recovered['recovered']); self.assertEqual(pages.checkpoint(), before)
            self.assertFalse((engine / MARKER).exists())
            publication = ConnectedWorkingPublication(root, entry()['id']); publication.stage()
            pages.clear(); pages.append(torch.tensor([3., 4.]), {'assemblyId': 'committed'})
            receipt = {'format': 'omni-completed-chat-turn', 'formatVersion': 1, 'turnId': entry()['rootTurnId'],
                'inputSha256': '1' * 64, 'humanMessageId': 'human', 'brainMessageId': 'brain', 'traceId': 'trace',
                'inferenceCount': 1, 'parameterChecksumAfter': '2' * 64, 'committedAt': '2026-10-11T00:00:00Z'}
            result_path = engine / 'connected-executions' / entry()['id'] / 'result.json'
            atomic_write_json(result_path, {'turnReceipt': receipt})
            metadata['connected_executions']['entries'][0].update(status='complete', completion={
                'resultPath': result_path.relative_to(engine.parent).as_posix(),
                'sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(), 'turnId': entry()['rootTurnId']})
            metadata['completed_chat_turns'] = [receipt]
            atomic_write_json(engine / 'brain.json', metadata)
            committed = pages.checkpoint()
            recovered = recover_inherited_working_publication(engine)
            self.assertFalse(recovered['recovered']); self.assertTrue(recovered['rootCommitted'])
            self.assertEqual(pages.checkpoint(), committed)

    def test_mapped_completion_receipt_preserves_stable_native_closed_schema(self):
        from omni_core.connected_execution import mapped_native_turn_receipt
        source = {'format': 'omni-completed-chat-turn', 'formatVersion': 1, 'turnId': 'old',
            'inputSha256': '1' * 64, 'humanMessageId': 'human', 'brainMessageId': 'brain', 'traceId': 'trace',
            'inferenceCount': 1, 'parameterChecksumAfter': '2' * 64, 'committedAt': '2026-10-11T00:00:00Z'}
        root = SimpleNamespace(counters={'inference_count': 9}, _validated_completed_chat_turns=AdaptiveBrain._validated_completed_chat_turns)
        receipt = mapped_native_turn_receipt(root, source, turn_id='new-root', human_id='existing-root-human',
            message_id='new-root-reply', trace_id='new-root-trace', checksum='3' * 64)
        self.assertEqual(set(receipt), set(source)); self.assertEqual(receipt['inferenceCount'], 9)
        self.assertEqual(receipt['turnId'], 'new-root'); self.assertEqual(receipt['humanMessageId'], 'existing-root-human')
        self.assertEqual(source['turnId'], 'old')
        self.assertEqual(AdaptiveBrain._validated_completed_chat_turns([receipt]), [receipt])

    def test_publication_boundary_releases_sqlite_writer_without_disposing_beforeimages(self):
        import sqlite3
        from omni_core.connected_transaction import ConnectedExperienceTransaction, _Participant, _ACTIVE
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regional.sqlite3'
            database = sqlite3.connect(path); database.execute('CREATE TABLE state(value)'); database.commit()
            participant = _Participant(database)
            database.execute('INSERT INTO state VALUES(7)')
            scope = object.__new__(ConnectedExperienceTransaction)
            scope.brain = SimpleNamespace(); scope.databases = {'region': participant}; scope._publication_boundary = False
            beforeimage = object(); scope.journal = beforeimage; scope._token = _ACTIVE.set(scope)
            try:
                scope.prepare_publication()
                self.assertIsNone(_ACTIVE.get()); self.assertIs(scope.journal, beforeimage)
                with sqlite3.connect(path, timeout=.05) as another:
                    another.execute('BEGIN IMMEDIATE')
                    self.assertEqual(another.execute('SELECT value FROM state').fetchone(), (7,))
                    another.rollback()
                scope.prepare_publication()
            finally:
                if scope._token is not None:
                    _ACTIVE.reset(scope._token)
                database.close()

    def test_unloaded_stop_tombstones_history_without_any_neural_load_or_weight_change(self):
        from omni_core.connected_execution import _digest
        from omni_core.conversation_ledger import NeuralConversationLedger
        from omni_core.persistence import atomic_write_json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); engine = root / 'engine'; engine.mkdir()
            ledger = NeuralConversationLedger(engine / 'conversation.sqlite3', 'root')
            human = {'id': 'mapped-human', 'role': 'human', 'content': 'Hello', 'turn_id': 'original-turn', 'attention_epoch': 0}
            ledger.append('message', (human,)); head = ledger.summary(); ledger.close()
            source = root / 'merge/parents/parent/engine'; source.mkdir(parents=True)
            atomic_write_json(source / 'brain.json', {'brain_id': 'parent'})
            saved = {'presentation': {'publishedText': 'ed'}}
            item = entry(); item['sourceManifestSha256'] = _digest(saved)
            atomic_write_json(engine / 'brain.json', {'brain_id': 'root', 'conversation': head,
                'connected_executions': queue([item]), 'mutable_state': {'committed': 'unchanged'}})
            for name in ('core.safetensors', 'plasticity.safetensors'):
                (engine / name).write_bytes(b'unchanged-placeholder-not-a-loadable-model')
            before = {name: (engine / name).read_bytes() for name in ('core.safetensors', 'plasticity.safetensors')}
            with patch.object(ExecutionCheckpointStore, 'current', return_value=saved), \
                    patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('Stop implicitly loaded a neural model')):
                result = stop_unloaded_connected_execution(root, 'root', item['id'])
                retry = stop_unloaded_connected_execution(root, 'root', item['id'])
            self.assertTrue(result['stopped']); self.assertFalse(result['turnCommitted'])
            self.assertEqual(result['text'], 'ed'); self.assertTrue(retry['idempotentCompletion'])
            self.assertEqual({name: (engine / name).read_bytes() for name in before}, before)
            self.assertEqual(json.loads((engine / 'brain.json').read_text())['mutable_state'], {'committed': 'unchanged'})
            ledger = NeuralConversationLedger(engine / 'conversation.sqlite3', 'root')
            try:
                self.assertEqual(ledger.counts()['messageCount'], 2)
                self.assertEqual(ledger.payload_by_id('message', 'mapped-human'), human)
            finally:
                ledger.close()

    def test_pending_frontier_holds_joint_mutation_but_closed_frontier_does_not(self):
        brain = SimpleNamespace(connected_executions=queue([entry()]))
        self.assertTrue(connected_execution_status(brain)['rootMutationHeld'])
        with self.assertRaisesRegex(NeuralStateResourcePause, 'resume inherited'):
            require_joint_mutation_allowed(brain, operation='training')
        brain.connected_executions['entries'][0]['status'] = 'stopped'
        self.assertFalse(require_joint_mutation_allowed(brain, operation='training')['rootMutationHeld'])

    def test_unsafe_source_and_duplicate_turn_are_rejected(self):
        bad = entry(); bad['sourceEnginePath'] = '../outside'
        with self.assertRaises(ValueError):
            validated_connected_executions(queue([bad]))
        other = entry(); other['id'] = 'f' * 64
        with self.assertRaises(ValueError):
            validated_connected_executions(queue([entry(), other]))

    def test_new_effects_get_strict_namespaces_prior_effects_do_not_emit(self):
        session = object.__new__(ConnectedExecutionSession)
        session.root = SimpleNamespace(brain_id='root')
        session.entry = entry(); session.root_turn_id = entry()['rootTurnId']; session.original_turn_id = 'original-turn'
        session.prior_action_ids = {'9' * 32}; session.new_actions = set(); session.source_actions = {}
        previous = {'actionId': '9' * 32, 'action': {'actionId': '9' * 32, 'kind': 'tool', 'toolId': 'system.files', 'action': 'read'}}
        self.assertIsNone(session.outward_event('action', previous))
        action = {'actionId': '8' * 32, 'action': {'actionId': '8' * 32, 'kind': 'tool', 'toolId': 'system.files', 'action': 'read'}}
        outward = session.outward_event('action', action)
        self.assertRegex(outward['actionId'], '^[a-f0-9]{32}$')
        self.assertNotEqual(outward['actionId'], action['actionId'])
        self.assertEqual(outward['actionId'], outward['action']['actionId'])
        self.assertEqual(action['actionId'], '8' * 32)

    def test_user_tool_argument_documents_are_not_rekeyed(self):
        session = object.__new__(ConnectedExecutionSession)
        session.root = SimpleNamespace(brain_id='root')
        session.entry = entry(); session.root_turn_id = entry()['rootTurnId']; session.original_turn_id = 'original-turn'
        session.prior_action_ids = set()
        value = {'kind': 'tool', 'toolId': 'system.files', 'arguments': {'brainId': 'customer-document',
            'json': {'turnId': 'original-turn', 'brainId': 'unrelated-document'}}}
        self.assertEqual(session._outward(value)['arguments'], value['arguments'])

    def test_new_root_observation_alias_preserves_external_effect_and_payload(self):
        from omni_core.chat_tool_observation import FIELDS, FORMAT
        session = object.__new__(ConnectedExecutionSession)
        session.root = SimpleNamespace(brain_id='root'); session.entry = entry()
        session.root_turn_id = entry()['rootTurnId']; session.original_turn_id = 'original-turn'
        source_id = '7' * 32; mapped_id = session.action_id(source_id)
        session.source_actions = {mapped_id: {'kind': 'tool', 'toolId': 'system.files', 'action': 'read', 'actionId': source_id}}
        payload = '{"outputPresent":true,"output":{"ok":true}}'
        value = {'format': FORMAT, 'brainId': 'root', 'turnId': session.root_turn_id, 'neuralActionId': mapped_id,
            'actionEventId': 'root-action-event', 'executionId': 'actual-external-execution', 'toolId': 'system.files',
            'action': 'read', 'completedAt': '2026-10-11T00:00:00Z', 'payloadJson': payload,
            'payloadSha256': hashlib.sha256(payload.encode()).hexdigest()}
        fields = [value[key] for key in ('format','brainId','turnId','neuralActionId','actionEventId',
            'executionId','toolId','action','completedAt','payloadSha256')]
        value['observationId'] = hashlib.sha256('\0'.join(fields).encode()).hexdigest()
        self.assertEqual(set(value), FIELDS)
        inward = session.inward_observation(value)
        self.assertEqual(inward['brainId'], 'parent'); self.assertEqual(inward['turnId'], 'original-turn')
        self.assertEqual(inward['neuralActionId'], source_id)
        for key in ('payloadJson','payloadSha256','executionId','actionEventId','completedAt'):
            self.assertEqual(inward[key], value[key])
        self.assertNotEqual(inward['observationId'], value['observationId'])
        bad = copy.deepcopy(value); bad['executionId'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            session.inward_observation(bad)
        from omni_core.connected_execution import map_inherited_observation
        current = [session.source_actions[mapped_id]]
        self.assertEqual(map_inherited_observation('root', session.entry, current, [], value), inward)
        with self.assertRaisesRegex(ValueError, 'newly emitted'):
            map_inherited_observation('root', session.entry, current, current, value)


@unittest.skipUnless(os.environ.get('OMNI_NATIVE_CONNECTED_EXECUTION_FIXTURE'), 'explicit existing native fixture; never initializes or trains one')
class NativeConnectedExecutionProof(unittest.TestCase):
    def test_actual_parent_frontier_resuspends_cold_finishes_once_inside_merged_owner(self):
        torch.set_num_threads(1)
        source = Path(os.environ['OMNI_NATIVE_CONNECTED_EXECUTION_FIXTURE']).resolve()
        second = source.parent / 'merged/merge/parents/b'
        self.assertTrue((source / 'engine/brain.json').is_file())
        self.assertTrue((second / 'engine/brain.json').is_file())
        original = hashlib.sha256((source / 'engine/brain.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='omni-connected-execution-proof-') as folder, \
                patch.object(ResourcePolicy, '_adaptive_disk_reserve', return_value=1024 * 1024), \
                patch.object(AdaptiveBrain, '_select_structured_actions', return_value=({'talk': 1., 'ponder': 0.}, [])):
            root = Path(folder) / 'merged'
            a, b = root / 'merge/parents/a', root / 'merge/parents/b'
            shutil.copytree(source, a); shutil.copytree(second, b)
            worker = Worker(); events = []; requested = []
            params = {'brainId': 'a', 'storagePath': str(a), 'input': 'Hello', 'streamId': 'inherited-parent-turn',
                'seed': 719, 'maxNewTokens': 7, 'toolSchemas': [], 'toolObservationWaitMs': 0, 'onlineLearning': False}
            def notify(kind, **values):
                events.append((kind, values))
                if kind == 'chat-token' and values['data']['delta'] and not requested:
                    requested.append(worker.request_chat_suspension({'brainId': 'a', 'streamId': params['streamId']}))
            worker.notify = notify
            try:
                worker.load(params, 'load')
                paused = worker.chat(params, 'pause')
                self.assertTrue(paused['suspended'])
                checkpoint = paused['resume']['checkpointId']
                before_inference = worker.brains['a'].counters['inference_count']
                before_experiences = worker.brains['a'].counters['experiences']
                prefix = ''.join(value['data']['delta'] for kind, value in events if kind == 'chat-token')
            finally:
                worker.shutdown({}, None)
            reference = Path(folder) / 'reference'; shutil.copytree(a, reference)
            worker = Worker(); worker.notify = lambda *_args, **_kwargs: None
            try:
                worker.load({'brainId': 'a', 'storagePath': str(reference)}, 'load-reference')
                expected = worker.resume_chat({'brainId': 'a', 'storagePath': str(reference),
                    'checkpointId': checkpoint, 'streamId': params['streamId']}, 'reference')
                self.assertTrue(expected['turnCommitted'])
            finally:
                worker.shutdown({}, None)
            source_seals = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (root / 'merge/parents').rglob('*') if path.is_file()}
            declarations = [{'brainId': name, 'enginePath': 'merge/parents/' + name + '/engine',
                'generationSha256': hashlib.sha256(name.encode()).hexdigest()} for name in ('a', 'b')]
            settings = OmniConfig.micro(max_seq_len=128, working_memory_slots=8)
            receipt = assemble_full_merge(root, 'joined', declarations, settings)
            self.assertEqual(receipt['runtimeAdoption']['counts']['pendingExecutions'], 1)
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            before_inference = brain.counters['inference_count']
            before_experiences = brain.counters['experiences']
            selected = ConnectedExecutionQueue(brain).next_pending()
            self.assertEqual(selected['publishedText'], prefix)
            observed = []
            try:
                rng = torch.get_rng_state().clone()
                with patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('second neural model allocation')):
                    with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                        self.assertIs(session.brain.decoder, session.owner.region.decoder)
                        self.assertEqual(session.brain.config.d_model, 8)
                        self.assertEqual(brain.config.d_model, 16)
                        native_state = copy.deepcopy(session.manifest['worker'])
                        def emit(kind, payload):
                            observed.append((kind, payload))
                            if kind == 'token':
                                native_state['publishedText'] += payload['delta']
                            native_state['lastStreamSequence'] += 1
                        paused_again = session.run(stream_callback=emit,
                            suspension_check=lambda: any(kind == 'token' for kind, _payload in observed),
                            worker_state=lambda: native_state, cancel_check=lambda: False)
                        self.assertTrue(paused_again['suspended'])
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(brain.counters['inference_count'], before_inference)
                self.assertEqual(brain.counters['experiences'], before_experiences)
            finally:
                brain.close()
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    completed = session.run(stream_callback=lambda kind, value: observed.append((kind, value)), cancel_check=lambda: False)
                self.assertTrue(completed['turnCommitted'])
                self.assertEqual(completed['text'], expected['text'])
                self.assertEqual(prefix + ''.join(value['delta'] for kind, value in observed if kind == 'token'), completed['text'])
                self.assertEqual(brain.counters['inference_count'], before_inference + 1)
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                before_events = len(observed)
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    duplicate = session.run(stream_callback=lambda *value: observed.append(value))
                self.assertTrue(duplicate['idempotentCompletion']); self.assertEqual(duplicate['actions'], [])
                self.assertEqual(len(observed), before_events)
            finally:
                brain.close()
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                self.assertEqual(len([row for row in brain.messages if row.get('turn_id') == selected['rootTurnId']]), 1)
                self.assertEqual(brain.conversation.payload_by_id('message', completed['message']['id'])['content'], expected['text'])
            finally:
                brain.close()
            self.assertEqual({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (root / 'merge/parents').rglob('*') if path.is_file()}, source_seals)
        self.assertEqual(hashlib.sha256((source / 'engine/brain.json').read_bytes()).hexdigest(), original)


@unittest.skipUnless(os.environ.get('OMNI_NATIVE_CONNECTED_READY_FIXTURE'), 'explicit previously staged private merge; never assembles or trains one')
class NativeConnectedExecutionExistingMergeProof(unittest.TestCase):
    def test_committed_native_cold_load_after_private_receipt_metadata_repair(self):
        """Targeted recovery of this lane's private failed receipt fixture.

        Not a new end-to-end completion pass: decoding/neural finalization are
        not repeated and the old source-scoped trace is not rewritten.
        """
        from omni_core.persistence import atomic_write_json
        torch.set_num_threads(1)
        folder = Path(os.environ['OMNI_NATIVE_CONNECTED_READY_FIXTURE']).resolve()
        if not folder.name.startswith('omni-connected-execution-proof-'):
            raise ValueError('receipt repair is restricted to this lane private fixture')
        root = folder / 'merged'; engine = root / 'engine'
        metadata = json.loads((engine / 'brain.json').read_text())
        selected = next(row for row in metadata['connected_executions']['entries'] if row['status'] == 'complete')
        result_path = root / selected['completion']['resultPath']
        self.assertEqual(hashlib.sha256(result_path.read_bytes()).hexdigest(), selected['completion']['sha256'])
        result = json.loads(result_path.read_text())
        core = {name: hashlib.sha256((engine / name).read_bytes()).hexdigest() for name in ('core.safetensors', 'plasticity.safetensors')}
        receipt = result['turnReceipt']
        extras = {'inheritedNative', 'inheritedInputAlias'} & set(receipt)
        if extras:
            self.assertEqual(extras, {'inheritedNative', 'inheritedInputAlias'})
            closed = {key: value for key, value in receipt.items() if key not in extras}
            AdaptiveBrain._validated_completed_chat_turns([closed])
            metadata['completed_chat_turns'] = [closed if row['turnId'] == closed['turnId'] else row for row in metadata['completed_chat_turns']]
            result['turnReceipt'] = closed
            result['receiptMetadataRepair'] = {'scope': 'private fixture provenance-key repair only', 'neuralChanges': False}
            atomic_write_json(result_path, result)
            selected['completion']['sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            atomic_write_json(engine / 'brain.json', metadata)
        with patch.object(ResourcePolicy, '_adaptive_disk_reserve', return_value=1024 * 1024):
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                self.assertEqual(brain.counters['inference_count'], 1)
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                self.assertEqual(brain.parameter_checksum(), result['turnReceipt']['parameterChecksumAfter'])
                events = []
                with patch.object(brain, 'metrics', return_value={}), patch.object(brain, 'runtime_card', return_value={}):
                    with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                        retry = session.run(stream_callback=lambda *args: events.append(args))
                self.assertTrue(retry['idempotentCompletion']); self.assertEqual(retry['actions'], [])
                self.assertEqual(retry['text'], result['text']); self.assertEqual(events, [])
                self.assertEqual(brain.conversation.payload_by_id('message', result['message']['id'])['content'], result['text'])
            finally:
                brain.close()
        self.assertEqual({name: hashlib.sha256((engine / name).read_bytes()).hexdigest() for name in core}, core)

    def test_only_existing_final_frontier_after_publication_repair(self):
        import sqlite3
        torch.set_num_threads(1)
        folder = Path(os.environ['OMNI_NATIVE_CONNECTED_READY_FIXTURE']).resolve()
        if not folder.name.startswith('omni-connected-execution-proof-'):
            raise ValueError('final-frontier proof requires its existing owned fixture')
        root = folder / 'merged'
        with sqlite3.connect(folder / 'reference/engine/conversation.sqlite3') as database:
            expected = json.loads(database.execute("SELECT payload_json FROM entries WHERE kind='message' AND json_extract(payload_json,'$.role')='brain' ORDER BY sequence DESC LIMIT 1").fetchone()[0])['content']
        hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / 'merge/parents').rglob('*') if path.is_file()}
        with patch.object(ResourcePolicy, '_adaptive_disk_reserve', return_value=1024 * 1024), \
                patch.object(AdaptiveBrain, '_select_structured_actions', return_value=({'talk': 1., 'ponder': 0.}, [])):
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            selected = ConnectedExecutionQueue(brain).next_pending()
            self.assertIsNotNone(selected); self.assertEqual(selected['status'], 'suspended')
            before = brain.counters['inference_count']; observed = []
            try:
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    result = session.run(stream_callback=lambda kind, value: observed.append((kind, value)), cancel_check=lambda: False)
                self.assertTrue(result['turnCommitted'])
                self.assertEqual(result['text'], expected)
                self.assertEqual(selected['publishedText'] + ''.join(value['delta'] for kind, value in observed if kind == 'token'), expected)
                self.assertEqual(brain.counters['inference_count'], before + 1)
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                count = len(observed)
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    retry = session.run(stream_callback=lambda *args: observed.append(args))
                self.assertTrue(retry['idempotentCompletion']); self.assertEqual(retry['actions'], [])
                self.assertEqual(len(observed), count)
                message_id = result['message']['id']
            finally:
                brain.close()
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                self.assertEqual(brain.conversation.payload_by_id('message', message_id)['content'], expected)
            finally:
                brain.close()
        self.assertEqual({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / 'merge/parents').rglob('*') if path.is_file()}, hashes)

    def test_existing_actual_frontier_resuspends_reloads_finishes_and_deduplicates(self):
        import sqlite3
        torch.set_num_threads(1)
        folder = Path(os.environ['OMNI_NATIVE_CONNECTED_READY_FIXTURE']).resolve()
        if not folder.name.startswith('omni-connected-execution-proof-'):
            raise ValueError('ready native continuation proof must use its own private fixture')
        root = folder / 'merged'
        with sqlite3.connect(folder / 'reference/engine/conversation.sqlite3') as database:
            expected = json.loads(database.execute("SELECT payload_json FROM entries WHERE kind='message' AND json_extract(payload_json,'$.role')='brain' ORDER BY sequence DESC LIMIT 1").fetchone()[0])['content']
        before_sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / 'merge/parents').rglob('*') if path.is_file()}
        with patch.object(ResourcePolicy, '_adaptive_disk_reserve', return_value=1024 * 1024), \
                patch.object(AdaptiveBrain, '_select_structured_actions', return_value=({'talk': 1., 'ponder': 0.}, [])):
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            selected = ConnectedExecutionQueue(brain).next_pending()
            self.assertIsNotNone(selected)
            prefix = selected['publishedText']
            observed = []
            before_inference, before_experiences = brain.counters['inference_count'], brain.counters['experiences']
            try:
                rng = torch.get_rng_state().clone()
                with patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('a second unrelated model was allocated')):
                    with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                        self.assertIs(session.brain.decoder, session.owner.region.decoder)
                        self.assertEqual(session.brain.config.d_model, 8)
                        self.assertEqual(brain.config.d_model, 16)
                        native_state = copy.deepcopy(session.manifest['worker'])
                        def emit(kind, payload):
                            observed.append((kind, payload))
                            if kind == 'token':
                                native_state['publishedText'] += payload['delta']
                            native_state['lastStreamSequence'] += 1
                        paused = session.run(stream_callback=emit,
                            suspension_check=lambda: any(kind == 'token' for kind, _payload in observed),
                            worker_state=lambda: native_state, cancel_check=lambda: False)
                        self.assertTrue(paused['suspended'])
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(brain.counters['inference_count'], before_inference)
                self.assertEqual(brain.counters['experiences'], before_experiences)
            finally:
                brain.close()
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    completed = session.run(stream_callback=lambda kind, value: observed.append((kind, value)), cancel_check=lambda: False)
                self.assertTrue(completed['turnCommitted'])
                self.assertEqual(completed['text'], expected)
                self.assertEqual(prefix + ''.join(value['delta'] for kind, value in observed if kind == 'token'), expected)
                self.assertEqual(brain.counters['inference_count'], before_inference + 1)
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                count = len(observed)
                with ConnectedExecutionQueue(brain).prepare(selected['id']) as session:
                    duplicate = session.run(stream_callback=lambda *value: observed.append(value))
                self.assertTrue(duplicate['idempotentCompletion']); self.assertEqual(duplicate['actions'], [])
                self.assertEqual(len(observed), count)
                message_id = completed['message']['id']
            finally:
                brain.close()
            brain = AdaptiveBrain.load(root, expected_brain_id='joined')
            try:
                self.assertEqual(ConnectedExecutionQueue(brain).status()['pending'], 0)
                self.assertEqual(len([row for row in brain.messages if row.get('turn_id') == selected['rootTurnId']]), 1)
                self.assertEqual(brain.conversation.payload_by_id('message', message_id)['content'], expected)
            finally:
                brain.close()
        self.assertEqual({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / 'merge/parents').rglob('*') if path.is_file()}, before_sources)


if __name__ == '__main__':
    unittest.main()
