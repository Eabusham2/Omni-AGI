"""Metadata-only evidence durability; no neural constructor or inference."""
import copy
from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.chat_tool_observation import ChatToolObservationInbox, FORMAT
from omni_core.connected_execution import (_digest, inherited_action_id, stage_inherited_observations)
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.offload import DurableReplayBuffer, PagedWorkingMemory
from omni_core.persistence import atomic_write_json, read_json
from tests.test_connected_execution import entry, queue


class InheritedLateObservationTests(unittest.TestCase):
    def test_late_new_effect_is_durable_retry_safe_and_does_not_mutate_any_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); engine = root / 'engine'; engine.mkdir()
            policy = SimpleNamespace(require_disk=lambda *_args: None,
                status=lambda **_kwargs: {'memoryPressure': False}, reserve_ram=lambda *_args: nullcontext())
            original_engine = root / 'merge/parents/parent/engine'
            current_engine = engine / 'connected-executions/source/engine'
            for path in (original_engine, current_engine):
                path.mkdir(parents=True)
                atomic_write_json(path / 'brain.json', {'brain_id': 'parent'})
                for name in ('core.safetensors', 'plasticity.safetensors'):
                    (path / name).write_bytes(b'fixture-seal-no-loadable-model')
                ledger = NeuralConversationLedger(path / 'conversation.sqlite3', 'parent'); ledger.close()
                DurableReplayBuffer(path / 'state/replay.sqlite3', policy)
                PagedWorkingMemory(path / 'state/working-memory.sqlite3', policy)
            import gc
            gc.collect()  # Close fixture-created SQLite cycles before freezing source bytes.
            source_id = '7' * 32
            action = {'kind': 'tool', 'actionId': source_id, 'toolId': 'system.files', 'action': 'read', 'arguments': {'path': 'fixture.txt'}}
            inbox = ChatToolObservationInbox('parent', 'original-turn', lambda _size: True, wait_seconds=0)
            inbox.register(source_id, action)
            worker = {'brainId': 'parent', 'turnId': 'original-turn', 'lastStreamSequence': 4,
                'publishedText': 'ed', 'actions': [action], 'inbox': inbox.suspension_state()}
            inbox.close()
            current = {'checkpointId': 'c' * 32, 'status': 'suspended', 'worker': worker, 'frontier': {'turnId': 'original-turn'}}
            original = copy.deepcopy(current); original['worker']['actions'] = []
            selected = entry(); selected.update(sourceManifestSha256=_digest(original),
                resumeEnginePath=current_engine.relative_to(root).as_posix(), resumeCheckpointId=current['checkpointId'],
                resumeManifestSha256=_digest(current), status='suspended')
            state = queue([selected]); atomic_write_json(engine / 'brain.json', {'brain_id': 'root', 'connected_executions': state})
            owner = SimpleNamespace(brain_id='root', storage_path=root, engine_path=engine,
                resource_policy=policy, connected_executions=state)
            payload = '{"outputPresent":true,"output":{"text":"actual captured fixture result"}}'
            observation = {'format': FORMAT, 'brainId': 'root', 'turnId': selected['rootTurnId'],
                'neuralActionId': inherited_action_id('root', selected['id'], source_id), 'actionEventId': 'event-owned-by-root',
                'executionId': 'external-effect-already-completed', 'toolId': 'system.files', 'action': 'read',
                'completedAt': '2026-10-11T00:00:00Z', 'payloadJson': payload, 'payloadSha256': hashlib.sha256(payload.encode()).hexdigest()}
            keys = ('format','brainId','turnId','neuralActionId','actionEventId','executionId','toolId','action','completedAt','payloadSha256')
            observation['observationId'] = hashlib.sha256('\0'.join(observation[key] for key in keys).encode()).hexdigest()
            source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for directory in (original_engine,current_engine)
                for path in directory.rglob('*') if path.is_file()}
            def selected_current(store, expected_id=None):
                if store.engine == original_engine:
                    return copy.deepcopy(original)
                if store.engine == current_engine:
                    return copy.deepcopy(current)
                if not (store.engine / '.execution/active.json').is_file():
                    return copy.deepcopy(current)
                pointer = read_json(store.engine / '.execution/active.json')
                return read_json(store.engine / '.execution' / pointer['checkpointId'] / ('m-' + pointer['manifestSha256'] + '.json'))
            with patch.object(ExecutionCheckpointStore, 'current', selected_current), \
                    patch('omni_core.connected_execution._verify_files'), \
                    patch('omni_core.connected_execution.execution_checkpoint_snapshot_entries', return_value=iter(())), \
                    patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('late evidence allocated a neural model')):
                first = stage_inherited_observations(owner, selected['id'], [observation])
                path = owner.connected_executions['entries'][0]['resumeEnginePath']
                second = stage_inherited_observations(owner, selected['id'], [observation])
            self.assertTrue(first['committed']); self.assertFalse(first['duplicate']); self.assertTrue(second['duplicate'])
            self.assertEqual(owner.connected_executions['entries'][0]['resumeEnginePath'], path)
            self.assertEqual({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for directory in (original_engine,current_engine)
                for path in directory.rglob('*') if path.is_file()}, source_hashes)
            state = read_json(engine / 'brain.json')['connected_executions']['entries'][0]
            saved = selected_current(ExecutionCheckpointStore(root / state['resumeEnginePath'], 'parent'))
            pending = saved['worker']['inbox']['pending']
            self.assertEqual(len(pending), 1); self.assertEqual(pending[0]['payloadJson'], payload)
            self.assertEqual(pending[0]['neuralActionId'], source_id)
            self.assertEqual(pending[0]['executionId'], observation['executionId'])
            self.assertEqual(pending[0]['actionEventId'], observation['actionEventId'])
            self.assertEqual(set(state['runtimeFiles']), {'brain.json','core.safetensors','plasticity.safetensors',
                'conversation.sqlite3','state/replay.sqlite3','state/working-memory.sqlite3'})


if __name__ == '__main__': unittest.main()
