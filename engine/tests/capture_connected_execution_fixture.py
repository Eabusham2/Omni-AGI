"""One short NEW supported frontier, using an existing genuine merged origin.

This is an explicit developer fixture utility, not product migration. It never
restores an old cursor, reruns initial learning, creates a second model, or
resets completed metadata over changed weights. The immutable origin is cloned
before a previously accepted input is recomputed through its registered owner.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.activity_clock import ActivityClock
from omni_core.brain import AdaptiveBrain
from omni_core.chat_tool_observation import ChatToolObservationInbox
from omni_core.connected_execution import (_digest, _runtime_files, persist_borrowed_generation_clock,
    suspended_execution_identity, validated_connected_executions)
from omni_core.connected_execution_owner import BorrowedRegionalOwner, private_runtime_engine
from omni_core.execution_checkpoint import ExecutionCheckpointStore, binding_for
from omni_core.full_merge import verify_connected_origin
from omni_core.persistence import atomic_write_json, read_json


def _inventory(directory):
    result = {}
    for path in Path(directory).rglob('*'):
        if path.is_symlink():
            raise ValueError('native fixture cannot copy a symbolic-link source')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            result[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return result


def capture_current_seal_fixture(previous_fixture, *, progress=None, prepared_fixture=None,
        destination_folder=None, root_name='merged', prepare_only=False, resident_brain=None, cancelled=lambda: False):
    """Return a private root ready for the production Worker resume proof.

    Only the lane's known tiny genuine fixture is accepted. The source's
    existing accepted input is reused without admitting/learning it again;
    queued slow learning must already be disabled in that exact source config.
    The new checkpoint is captured by real current generation, not converted
    from the old incompatible execution payload.
    """
    started = time.perf_counter()
    previous = Path(previous_fixture).resolve()
    if not previous.name.startswith('omni-connected-execution-proof-'):
        raise ValueError('capture requires the previously retained private native fixture')
    old_root = previous / 'merged'
    origin = old_root / 'engine/origin'
    origin_metadata = read_json(origin / 'brain.json')
    root_id = origin_metadata['brain_id']
    if root_id != 'joined':
        raise ValueError('capture requires the known tiny joined fixture identity')
    verify_connected_origin(origin, brain_id=root_id,
        descriptor_sha256=origin_metadata['connected_regions']['sha256'])
    original_queue = validated_connected_executions(origin_metadata.get('connected_executions'))
    pending = [item for item in original_queue['entries'] if item['status'] in {'queued', 'suspended'}]
    if len(pending) != 1 or pending[0]['parentId'] != 'a' or len(pending[0]['regionPath']) != 1:
        raise ValueError('capture expects one exact direct source a frontier')
    original_entry = pending[0]
    original_source = old_root / original_entry['sourceEnginePath']
    saved_source = read_json(original_source / 'brain.json')
    if (saved_source['config']['online_learning'] is not False
            or saved_source['config']['d_model'] != 8
            or saved_source['config']['n_layers'] != 1):
        raise ValueError('fresh fixture capture cannot train or use a different source architecture')
    old_checkpoint = ExecutionCheckpointStore(original_source, 'a').current(original_entry['sourceCheckpointId'])
    if _digest(old_checkpoint) != original_entry['sourceManifestSha256']:
        raise ValueError('historical adopted source frontier lost its seal')
    text = old_checkpoint['presentation']['input']
    if text != 'Hello' or old_checkpoint['frontier']['turnId'] != original_entry['originalTurnId']:
        raise ValueError('fixture capture is limited to its already accepted short input')
    accepted = saved_source['accepted_chat_inputs']
    if len(accepted) != 1 or accepted[0] != old_checkpoint['presentation']['inputReceipt']:
        raise ValueError('fresh computation must reuse the exact accepted human receipt')
    before_source = _inventory(original_source)
    if prepared_fixture is not None and destination_folder is not None:
        raise ValueError('choose a fresh destination or a prepared clone, never both')
    if root_name not in {'merged', root_id}:
        raise ValueError('private fixture root name has another native identity')
    folder = Path(prepared_fixture or destination_folder).resolve() if prepared_fixture is not None or destination_folder is not None else Path(
        tempfile.mkdtemp(prefix='omni-connected-current-seal-')).resolve()
    if not folder.name.startswith('omni-connected-current-seal-') or folder.is_symlink():
        raise ValueError('prepared fixture is not its generated private capture directory')
    root_path = folder / root_name
    capture = root_path / 'engine/connected-execution-fixture/capture/engine'
    if (folder / 'capture-receipt.json').exists() or (capture / '.execution/active.json').exists():
        raise ValueError('do not replace or recompute an already authenticated fresh frontier')
    if progress:
        progress('origin-verified', {'folder': str(folder), 'elapsedSeconds': time.perf_counter() - started})
    if prepared_fixture is None:
        shutil.copytree(origin, root_path / 'engine')
        shutil.copytree(origin, root_path / 'engine/origin')
        for name in ('a', 'b'):
            shutil.copytree(old_root / 'merge/parents' / name, root_path / 'merge/parents' / name)
        receipt_path = old_root / 'engine/connected-assembly.json'
        if receipt_path.is_file():
            shutil.copyfile(receipt_path, root_path / 'engine' / receipt_path.name)
        atomic_write_json(folder / 'capture-preparation.json', {'sourceFixture': str(previous),
            'originMetadataSha256': hashlib.sha256((origin / 'brain.json').read_bytes()).hexdigest(), 'rootName': root_name})
    else:
        expected_preparation = {'sourceFixture': str(previous),
            'originMetadataSha256': hashlib.sha256((origin / 'brain.json').read_bytes()).hexdigest(), 'rootName': root_name}
        preparation_path = folder / 'capture-preparation.json'
        if not preparation_path.exists():
            # The first affected attempt predates this retry marker. Validate
            # its exact copied immutable bytes, not a completed mutable root.
            if hashlib.sha256((root_path / 'engine/brain.json').read_bytes()).hexdigest() != expected_preparation['originMetadataSha256']:
                raise ValueError('old prepared root no longer matches the copied immutable origin')
            for name in ('a', 'b'):
                for filename in ('brain.json', 'core.safetensors', 'plasticity.safetensors'):
                    retained = root_path / 'merge/parents' / name / 'engine' / filename
                    original = old_root / 'merge/parents' / name / 'engine' / filename
                    if hashlib.sha256(retained.read_bytes()).digest() != hashlib.sha256(original.read_bytes()).digest():
                        raise ValueError('old prepared source differs from its original native generation')
            atomic_write_json(preparation_path, expected_preparation)
        preparation = read_json(preparation_path)
        if preparation != expected_preparation:
            raise ValueError('retained prepared fixture has another immutable origin')
        verify_connected_origin(root_path / 'engine/origin', brain_id=root_id,
            descriptor_sha256=origin_metadata['connected_regions']['sha256'])
        if hashlib.sha256((root_path / 'engine/brain.json').read_bytes()).hexdigest() != preparation['originMetadataSha256']:
            if resident_brain is None:
                raise ValueError('retained prepared root metadata was changed before fresh computation')
            # Explicit host Load may persist reviewed resource-only settings.
            # Neither that nor a supplied object can substitute learned state.
            current = read_json(root_path / 'engine/brain.json')
            for key in ('brain_id', 'connected_regions', 'mutable_state', 'substrate', 'counters',
                    'accepted_chat_inputs', 'completed_chat_turns', 'connected_executions', 'recent_token_context', 'workspace_items', 'memory_lifecycle'):
                if current.get(key) != origin_metadata.get(key):
                    raise ValueError('Worker-owned Load changed an immutable learned/activity fixture binding')
            for name in ('core.safetensors', 'plasticity.safetensors'):
                if hashlib.sha256((root_path / 'engine' / name).read_bytes()).digest() != hashlib.sha256((origin / name).read_bytes()).digest():
                    raise ValueError('Worker-owned Load changed saved native tensor bytes')
    if prepare_only:
        if resident_brain is not None:
            raise ValueError('preparation never accepts or loads a neural owner')
        return {'format': 'omni-prepared-connected-native-fixture', 'version': 1, 'folder': str(folder),
            'root': str(root_path), 'brainId': root_id, 'config': origin_metadata['config'], 'modelInitialized': False}
    if resident_brain is not None and (not isinstance(resident_brain, AdaptiveBrain)
            or resident_brain.storage_path != root_path or resident_brain.brain_id != root_id
            or resident_brain.connected_executions != original_queue or not resident_brain._owned_activity_clock().paused):
        raise ValueError('capture requires the actual already-loaded quiescent Worker root, not registry injection')
    brain = None
    try:
        # This is a bounded existing fixture, not a corpus/training download.
        # Actual production RAM/disk admission remains in force.
        with nullcontext():
            brain = resident_brain if resident_brain is not None else AdaptiveBrain.load(root_path, expected_brain_id=root_id)
            if progress:
                progress('registered-owner-loaded', {'elapsedSeconds': time.perf_counter() - started})
            joint_checksum = brain.parameter_checksum()
            source = root_path / original_entry['sourceEnginePath']
            private_runtime_engine(source, capture, brain.resource_policy)
            # Ordinary learned-state migration only. No old execution cache,
            # cursor, RNG seal or compatibility fingerprint is read/restored.
            migrated = read_json(capture / 'brain.json')
            if migrated.get('activity_clock') is None:
                migrated['activity_clock'] = ActivityClock.from_state(None, legacy_markers=tuple(
                    migrated.get(key, 0.) for key in ('last_activity_decay', 'last_idle_cycle_at',
                        'last_idle_visible_action_at'))).snapshot()
                atomic_write_json(capture / 'brain.json', migrated)
            owner = BorrowedRegionalOwner(brain, original_entry, capture, capture)
            with patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('another neural constructor')):
                owner.enter()
                inbox = ChatToolObservationInbox('a', original_entry['originalTurnId'],
                    lambda size: not brain.resource_policy.status(estimated_ram_bytes=size).get('memoryPressure'),
                    wait_seconds=0, reserve_parse=brain.resource_policy.reserve_ram)
                try:
                    view = owner.brain
                    self_binding = binding_for(view, original_entry['originalTurnId'])
                    if self_binding != original_entry['sourceBindingId']:
                        raise ValueError('registered learned owner differs from immutable source generation')
                    view.save = lambda **_options: persist_borrowed_generation_clock(view,
                        original_entry['originalTurnId'], self_binding)
                    before_counters = copy.deepcopy(view.counters)
                    before_receipts = copy.deepcopy(view.accepted_chat_inputs)
                    state = {'brainId': 'a', 'turnId': original_entry['originalTurnId'],
                        'lastStreamSequence': 0, 'publishedText': '', 'actions': [],
                        'inbox': inbox.suspension_state(), 'priorLedgerHead': view.conversation.summary(),
                        'imaginationGrant': 'off'}
                    streamed = []
                    def emit(kind, payload):
                        state['lastStreamSequence'] += 1
                        if kind == 'token':
                            state['publishedText'] += payload['delta']
                            streamed.append(copy.deepcopy(payload))
                        elif kind == 'action':
                            state['actions'].append(copy.deepcopy(payload.get('action', payload)))
                    checkpoints = []
                    def suspend(context, cursor, continuation):
                        if view.counters != before_counters or view.accepted_chat_inputs != before_receipts:
                            raise AssertionError('fresh frontier re-admitted or learned the accepted input')
                        state['inbox'] = inbox.suspension_state()
                        value = ExecutionCheckpointStore(capture, 'a', policy=brain.resource_policy).save(view,
                            context, cursor, phase=continuation['phase'], worker_state=copy.deepcopy(state),
                            journal=context['before_parameters'], continuation=continuation)
                        checkpoints.append(value)
                        return {'suspended': True, 'turnCommitted': False, 'checkpointId': value['checkpointId']}
                    with patch.object(AdaptiveBrain, '_select_structured_actions',
                            return_value=({'talk': 1., 'ponder': 0.}, [])):
                        result = view.chat(text, seed=719, max_new_tokens=7, tool_schemas=[],
                            turn_id=original_entry['originalTurnId'], defer_slow_learning=True,
                            cancel_check=cancelled, stream_callback=emit,
                            tool_observation_provider=inbox,
                            suspension_check=lambda: bool(streamed), suspend_callback=suspend)
                    if not result.get('suspended') or len(checkpoints) != 1 or not streamed:
                        raise AssertionError('current generator did not capture a new actual token frontier')
                    fresh = checkpoints[0]
                    if fresh['checkpointId'] == old_checkpoint['checkpointId'] or fresh.get('runtimeCompatibility') is None:
                        raise AssertionError('fresh checkpoint is not independently current-sealed')
                    if brain.parameter_checksum() != joint_checksum:
                        raise AssertionError('read-only fresh capture changed the registered learned brain')
                finally:
                    inbox.close()
                    owner.close()
            identity = suspended_execution_identity(capture, 'a', policy=brain.resource_policy)
            region = next(row for row in origin_metadata['connected_regions']['regions']
                if row['regionId'] == original_entry['regionPath'][0])
            identifier = _digest(['ancestral-native-execution-1', region['regionId'], 'a', region['generationSha256'], identity])
            new_entry = {key: copy.deepcopy(value) for key, value in original_entry.items()
                if key not in {'resumeEnginePath', 'resumeCheckpointId', 'resumeManifestSha256', 'completion'}}
            new_entry.update(id=identifier, rootTurnId='ancestral-' + _digest([root_id, identifier]),
                sourceEnginePath=capture.relative_to(root_path).as_posix(),
                sourceCheckpointId=identity['checkpointId'], sourceManifestSha256=identity['manifestSha256'],
                sourceBindingId=identity['bindingId'], workerStateSha256=identity['workerStateSha256'],
                minimumContextTokens=fresh['minimumContextTokens'], publishedText=fresh['presentation']['publishedText'],
                runtimeFiles=_runtime_files(capture), status='queued', lineage=[*original_entry['lineage'], {
                    'freshComputation': True, 'historicalExecutionId': original_entry['id'],
                    'historicalCheckpointId': old_checkpoint['checkpointId'], 'historicalSourceManifestSha256': _digest(old_checkpoint),
                    'historicalCursorRestored': False, 'initialCurriculumRerun': False}])
            brain.connected_executions = validated_connected_executions({**original_queue, 'entries': [new_entry]})
            # Learned state did not change. Publish only this new private
            # fixture's queue/clock, never overwrite its sealed origin.
            metadata = read_json(root_path / 'engine/brain.json')
            metadata['connected_executions'] = brain.connected_executions
            metadata['activity_clock'] = brain._owned_activity_clock().pause()
            atomic_write_json(root_path / 'engine/brain.json', metadata)
            final = {'format': 'omni-fresh-connected-execution-fixture', 'version': 1,
                'folder': str(folder), 'root': str(root_path), 'brainId': root_id,
                'executionId': identifier, 'rootTurnId': new_entry['rootTurnId'],
                'originalTurnId': new_entry['originalTurnId'], 'checkpointId': fresh['checkpointId'],
                'publishedText': new_entry['publishedText'], 'parameterChecksum': joint_checksum,
                'elapsedSeconds': time.perf_counter() - started, 'initialCurriculumRerun': False,
                'oldCursorRestored': False, 'secondModelConstructed': False,
                'sourceUnchanged': _inventory(original_source) == before_source}
            if final['sourceUnchanged'] is not True:
                raise AssertionError('original historical fixture source changed')
            atomic_write_json(folder / 'capture-receipt.json', final)
            if progress:
                progress('ready-for-worker-resume', final)
            return final
    except BaseException:
        if progress:
            progress('failed-fixture-retained', {'folder': str(folder), 'elapsedSeconds': time.perf_counter() - started})
        raise
    finally:
        if brain is not None and resident_brain is None:
            brain.close()


if __name__ == '__main__':
    if len(sys.argv) not in (2, 4) or len(sys.argv) == 4 and sys.argv[2] != '--prepared':
        raise SystemExit('usage: capture_connected_execution_fixture.py <existing-private-native-fixture> [--prepared <retained-private-clone>]')
    torch.set_num_threads(1)
    capture_current_seal_fixture(sys.argv[1], prepared_fixture=sys.argv[3] if len(sys.argv) == 4 else None,
        progress=lambda kind, data: print(json.dumps({'phase': kind, **data}, sort_keys=True), flush=True))
