"""Root-owned executable frontiers for suspended parents of a full merge.

An ancestral cursor is never passed to the new connected sampler. Its exact
registered parent region supplies the original geometry until that frontier
finishes. New joint neural mutations wait; prior external effects remain
receipts, not instructions to execute them again.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import uuid
from typing import Mapping

from .execution_checkpoint import ExecutionCheckpointStore, canonical
from .execution_state_codec import file_digest
from .persistence import read_json


FORMAT = 'omni-connected-execution-queue'
VERSION = 1
_HASH = re.compile(r'[a-f0-9]{64}')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
_CHECKPOINT = re.compile(r'[a-f0-9]{32}')
_STATUSES = {'queued', 'suspended', 'complete', 'stopped'}


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest_text_parts(parts):
    if any(not isinstance(value, str) or not value or '\x00' in value for value in parts):
        raise ValueError('inherited receipt fields must be exact nonempty strings')
    return hashlib.sha256('\x00'.join(parts).encode('utf-8')).hexdigest()


def _copy_activity_metadata(value):
    """Private lists/control metadata; do not clone unchanged latent payloads."""
    import torch
    memo, seen = {}, set()
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            memo[id(item)] = item
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, '__dict__'):
            visit(vars(item))
    visit(value)
    return copy.deepcopy(value, memo)


def mapped_native_turn_receipt(brain, original, *, turn_id, human_id, message_id, trace_id, checksum):
    """Stable closed receipt; provenance is carried beside it, not inside it."""
    brain._validated_completed_chat_turns([original])
    receipt = copy.deepcopy(original)
    receipt.update(turnId=turn_id, humanMessageId=human_id, brainMessageId=message_id,
        traceId=trace_id, inferenceCount=int(brain.counters['inference_count']), parameterChecksumAfter=checksum)
    brain._validated_completed_chat_turns([receipt])
    return receipt


def inherited_action_id(root_id, execution_id, source_action_id):
    return _digest(['inherited-new-effect-1', root_id, execution_id, str(source_action_id)])[:32]


def _map_observation_routes(root_id, entry, routes, value):
    from .chat_tool_observation import FIELDS, FORMAT, _tool
    if (not isinstance(value, dict) or set(value) != FIELDS or value.get('brainId') != root_id
            or value.get('turnId') != entry['rootTurnId'] or value.get('format') != FORMAT):
        raise ValueError('new inherited tool evidence does not own this root execution')
    source = routes.get(value.get('neuralActionId'))
    if (source is None or source.get('kind') != 'tool' or _tool(source.get('toolId')) != _tool(value.get('toolId'))
            or source.get('action') != value.get('action')):
        raise ValueError('new inherited observation does not match its newly emitted root action')
    names = ('format', 'brainId', 'turnId', 'neuralActionId', 'actionEventId', 'executionId',
        'toolId', 'action', 'completedAt', 'payloadSha256')
    if _digest_text_parts([value[key] for key in names]) != value['observationId']:
        raise ValueError('inherited root evidence receipt checksum is invalid')
    inward = {**value, 'brainId': entry['parentId'], 'turnId': entry['originalTurnId'], 'neuralActionId': source['actionId']}
    inward['observationId'] = _digest_text_parts([inward[key] for key in names])
    return inward


def map_inherited_observation(root_id, entry, current_actions, original_actions, root_receipt):
    """No neural allocation: exact late-effect alias for an unloaded frontier."""
    prior = {str(action.get('actionId', action.get('id', ''))) for action in original_actions}
    routes = {inherited_action_id(root_id, entry['id'], action['actionId']): action for action in current_actions
        if action.get('actionId') not in prior and isinstance(action.get('actionId'), str)}
    return _map_observation_routes(root_id, entry, routes, root_receipt)


def stage_inherited_observations(root, execution_id, observations):
    """Freeze/read lease required; persist actual late evidence before Unload.

    Only an already existing root owner is used. Cache/model/RNG state is not
    decoded or changed. The original and previous runtime snapshots stay sealed.
    """
    import shutil
    from .chat_tool_observation import ChatToolObservationInbox
    from .connected_execution_owner import private_runtime_engine
    from .persistence import atomic_write_json
    state = validated_connected_executions(getattr(root, 'connected_executions', None))
    entry = next((item for item in state['entries'] if item['id'] == execution_id), None)
    if entry is None or entry['status'] not in {'queued', 'suspended'} or not isinstance(observations, list):
        raise ValueError('late inherited observations have no suspended root owner')
    original_engine = resolve_execution_engine(root.engine_path, entry)
    original = ExecutionCheckpointStore(original_engine, entry['parentId'], policy=root.resource_policy).current(entry['sourceCheckpointId'])
    if _digest(original) != entry['sourceManifestSha256']:
        raise ValueError('late observation lost its original adopted action baseline')
    source = resolve_execution_engine(root.engine_path, entry, resumed='resumeEnginePath' in entry)
    if 'runtimeFiles' in entry:
        _verify_runtime_files(source, entry['runtimeFiles'])
    store = ExecutionCheckpointStore(source, entry['parentId'], policy=root.resource_policy)
    current = store.current(entry.get('resumeCheckpointId', entry['sourceCheckpointId']))
    if _digest(current) != entry.get('resumeManifestSha256', entry['sourceManifestSha256']):
        raise ValueError('late observation lost its committed execution frontier')
    _verify_files(source, current)
    admit = lambda size: not root.resource_policy.status(estimated_ram_bytes=size).get('memoryPressure')
    inbox = ChatToolObservationInbox.from_suspension_state(current['worker']['inbox'], entry['parentId'],
        entry['originalTurnId'], admit, reserve_parse=root.resource_policy.reserve_ram)
    acknowledgements, changed = [], False
    try:
        for observation in observations:
            inward = map_inherited_observation(root.brain_id, entry, current['worker']['actions'],
                original['worker']['actions'], observation)
            acknowledgement = inbox.offer(inward)
            acknowledgements.append({**acknowledgement, 'brainId': root.brain_id, 'turnId': entry['rootTurnId'],
                'observationId': observation['observationId']})
            if not acknowledgement.get('accepted'):
                from .offload import NeuralStateResourcePause
                raise NeuralStateResourcePause('late inherited evidence waits for an admitted metadata boundary',
                    {'recoverable': True, 'paused': True, 'observationId': observation['observationId']})
            changed |= not acknowledgement.get('duplicate', False)
        if not changed:
            return {'brainId': root.brain_id, 'executionId': execution_id, 'committed': True,
                'duplicate': True, 'observations': acknowledgements}
        runtime = _inside(root.storage_path, 'engine/connected-executions/' + execution_id + '/runtime/' + uuid.uuid4().hex + '/engine')
        private_runtime_engine(source, runtime, root.resource_policy)
        for name, path, expected in execution_checkpoint_snapshot_entries(source,
                expected_id=current['checkpointId'], expected_sha256=_digest(current), brain_id=entry['parentId']):
            target = _inside(runtime, name); target.parent.mkdir(parents=True, exist_ok=True)
            root.resource_policy.require_disk(path.stat().st_size + 65536, 'late inherited exact activity copy')
            shutil.copyfile(path, target)
            import os
            with target.open('rb') as synced:
                os.fsync(synced.fileno())
            if file_digest(target) != expected:
                raise ValueError('late inherited evidence copy lost an exact activity file')
        worker = {**current['worker'], 'inbox': inbox.suspension_state()}
        saved = ExecutionCheckpointStore(runtime, entry['parentId'], policy=root.resource_policy).update_worker(current['checkpointId'], worker)
        entry.update(status='suspended', resumeEnginePath=runtime.relative_to(root.storage_path).as_posix(),
            resumeCheckpointId=saved['checkpointId'], resumeManifestSha256=_digest(saved), runtimeFiles=_runtime_files(runtime),
            workerStateSha256=_digest(saved['worker']))
        metadata = read_json(root.engine_path / 'brain.json')
        if metadata.get('brain_id') != root.brain_id or validated_connected_executions(metadata.get('connected_executions')) != validated_connected_executions(root.connected_executions):
            raise ValueError('late inherited evidence began outside the exact root metadata generation')
        metadata['connected_executions'] = state
        root.resource_policy.require_disk(len(canonical(metadata)) * 2 + 65536, 'late inherited evidence root commit')
        atomic_write_json(root.engine_path / 'brain.json', metadata)
        root.connected_executions = state
        return {'brainId': root.brain_id, 'executionId': execution_id, 'committed': True,
            'duplicate': False, 'observations': acknowledgements, 'resume': saved['frontier']}
    finally:
        inbox.close()


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or not relative.parts or any(part in {'', '.', '..'} for part in relative.parts):
        raise ValueError('ancestral execution path must be an owned relative path')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('ancestral execution cannot follow symbolic links')
    if current.resolve() != root and root not in current.resolve().parents:
        raise ValueError('ancestral execution escaped its merged instance')
    return current


def validated_connected_executions(value):
    if value is None:
        return {'format': FORMAT, 'formatVersion': VERSION, 'entries': []}
    if (not isinstance(value, Mapping) or set(value) != {'format', 'formatVersion', 'entries'}
            or value['format'] != FORMAT or value['formatVersion'] != VERSION or not isinstance(value['entries'], list)):
        raise ValueError('connected execution queue format is invalid')
    entries, seen, turns = [], set(), set()
    required = {'id', 'rootTurnId', 'parentId', 'regionPath', 'sourceEnginePath',
        'sourceCheckpointId', 'sourceManifestSha256', 'sourceBindingId', 'originalTurnId',
        'historyParentId', 'historyHumanMessageId', 'rootHumanMessageId', 'status', 'backend', 'lineage'}
    optional = {'completion', 'resumeEnginePath', 'resumeCheckpointId', 'resumeManifestSha256',
        'minimumContextTokens', 'publishedText', 'workerStateSha256', 'runtimeFiles'}
    for raw in value['entries']:
        if not isinstance(raw, Mapping) or not required <= set(raw) or not set(raw) <= required | optional:
            raise ValueError('connected execution entry fields are invalid')
        entry = copy.deepcopy(dict(raw))
        if (not isinstance(entry['id'], str) or not _HASH.fullmatch(entry['id']) or entry['id'] in seen
                or not isinstance(entry['rootTurnId'], str) or not _ID.fullmatch(entry['rootTurnId'])
                or entry['rootTurnId'] in turns or entry['status'] not in _STATUSES
                or not isinstance(entry['sourceCheckpointId'], str) or not _CHECKPOINT.fullmatch(entry['sourceCheckpointId'])
                or any(not isinstance(entry[key], str) or not _HASH.fullmatch(entry[key])
                    for key in ('sourceManifestSha256', 'sourceBindingId'))
                or any(not isinstance(entry[key], str) or not entry[key]
                    for key in ('parentId', 'originalTurnId', 'historyParentId', 'historyHumanMessageId', 'rootHumanMessageId', 'backend'))
                or not _ID.fullmatch(entry['parentId']) or not _ID.fullmatch(entry['originalTurnId'])
                or not isinstance(entry['regionPath'], list) or not entry['regionPath']
                or any(not isinstance(part, str) or not _ID.fullmatch(part) for part in entry['regionPath'])
                or not isinstance(entry['lineage'], list)):
            raise ValueError('connected execution owner/frontier identity is invalid')
        for key in ('sourceEnginePath', 'resumeEnginePath'):
            if key in entry:
                relative = Path(entry[key]) if isinstance(entry[key], str) else Path('/')
                if relative.is_absolute() or not relative.parts or any(part in {'', '.', '..'} for part in relative.parts):
                    raise ValueError('connected execution has an unsafe owned source')
        resume_fields = {'resumeEnginePath', 'resumeCheckpointId', 'resumeManifestSha256'}
        if resume_fields & set(entry) and (not resume_fields <= set(entry)
                or not _CHECKPOINT.fullmatch(str(entry['resumeCheckpointId']))
                or not _HASH.fullmatch(str(entry['resumeManifestSha256']))):
            raise ValueError('connected execution re-suspension binding is incomplete')
        if 'minimumContextTokens' in entry and (type(entry['minimumContextTokens']) is not int or entry['minimumContextTokens'] < 0):
            raise ValueError('ancestral retained context is invalid')
        if 'publishedText' in entry and not isinstance(entry['publishedText'], str):
            raise ValueError('ancestral published prefix must be text')
        if 'workerStateSha256' in entry and not _HASH.fullmatch(str(entry['workerStateSha256'])):
            raise ValueError('ancestral worker identity is invalid')
        if 'runtimeFiles' in entry:
            files = entry['runtimeFiles']
            if (not isinstance(files, Mapping) or set(files) != set(_RUNTIME_FILES)
                    or any(not isinstance(item, Mapping) or set(item) != {'bytes', 'sha256'}
                        or type(item['bytes']) is not int or item['bytes'] < 0 or not _HASH.fullmatch(str(item['sha256']))
                        for item in files.values())):
                raise ValueError('inherited private runtime activity seals are invalid')
        if entry['status'] == 'complete' or 'completion' in entry:
            completion = entry.get('completion')
            if (not isinstance(completion, Mapping) or set(completion) != {'resultPath', 'sha256', 'turnId'}
                    or not _ID.fullmatch(str(completion['turnId']))
                    or completion['turnId'] != entry['rootTurnId'] and not entry['lineage']
                    or not _HASH.fullmatch(str(completion['sha256']))):
                raise ValueError('completed ancestral execution has no root-owned result receipt')
            relative = Path(completion['resultPath'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('ancestral completion path is unsafe')
        seen.add(entry['id']); turns.add(entry['rootTurnId']); entries.append(entry)
    return {'format': FORMAT, 'formatVersion': VERSION, 'entries': entries}


def _verify_files(engine, manifest, *, require_supported=True):
    if require_supported:
        ExecutionCheckpointStore(engine, manifest['brainId']).require_supported_frontier(manifest)
    binding, neural = manifest.get('binding'), manifest.get('neuralFiles')
    metadata = read_json(engine / 'brain.json')
    if (not isinstance(binding, Mapping) or _digest(binding) != manifest.get('bindingId')
            or binding.get('brainId') != manifest['brainId'] or metadata.get('brain_id') != manifest['brainId']
            or binding.get('turnId') != manifest['frontier']['turnId']
            or binding.get('mutable') != metadata.get('mutable_state')
            or binding.get('substrate') != (metadata.get('substrate') or {}).get('persistence')
            or binding.get('attentionEpoch') != int((metadata.get('fresh_attention_boundary') or {}).get('epoch', 0))
            or not isinstance(neural, Mapping) or set(neural) != {'core.safetensors', 'plasticity.safetensors'}):
        raise ValueError('ancestral execution lost its committed native generation binding')
    for name, descriptor in neural.items():
        path = _inside(engine, name)
        if (not isinstance(descriptor, Mapping) or not path.is_file() or path.stat().st_size != descriptor.get('bytes')
                or file_digest(path) != descriptor.get('sha256')):
            raise ValueError('ancestral execution native tensor seal changed')
    directory = engine / '.execution' / manifest['checkpointId']
    for name, descriptor in manifest.get('files', {}).items():
        if not re.fullmatch(r'(?:t-[0-9]+\.safetensors|recall-[0-9]+\.sqlite3)', str(name)):
            raise ValueError('ancestral activity file inventory is invalid')
        path = _inside(directory, name)
        if (not isinstance(descriptor, Mapping) or not path.is_file() or type(descriptor.get('bytes')) is not int
                or path.stat().st_size != descriptor['bytes'] or file_digest(path) != descriptor.get('sha256')):
            raise ValueError('ancestral activity file lost its exact seal')
    diagnostic = _inside(directory, 'diagnostic/manifest.json')
    if not diagnostic.is_file() or file_digest(diagnostic) != manifest['journal']['manifestSha256']:
        raise ValueError('ancestral diagnostic baseline lost its exact seal')
    for name, descriptor in read_json(diagnostic)['files'].items():
        if name not in {'controls.safetensors', 'changed.sqlite'}:
            raise ValueError('ancestral diagnostic inventory is invalid')
        path = _inside(diagnostic.parent, name)
        if not isinstance(descriptor, Mapping) or not path.is_file() or path.stat().st_size != descriptor['bytes'] or file_digest(path) != descriptor['sha256']:
            raise ValueError('ancestral diagnostic bytes changed')


def suspended_execution_identity(engine, parent_id, *, policy=None):
    """Only active computation distinguishes otherwise byte-identical regions."""
    manifest = ExecutionCheckpointStore(engine, parent_id, policy=policy).current()
    if manifest is None or manifest['status'] != 'suspended':
        return None
    _verify_files(Path(engine), manifest)
    return {'checkpointId': manifest['checkpointId'], 'manifestSha256': _digest(manifest),
        'bindingId': manifest['bindingId'], 'workerStateSha256': _digest(manifest['worker'])}


def resolve_execution_engine(root_engine, entry, *, resumed=False):
    """Locate unchanged source identities in live or self-contained snapshots."""
    root_engine = Path(root_engine).resolve()
    logical = entry.get('resumeEnginePath') if resumed else entry['sourceEnginePath']
    if logical is None:
        return resolve_execution_engine(root_engine, entry)
    for candidate in (_inside(root_engine.parent, logical), _inside(root_engine / 'connected-execution-inputs', logical)):
        if (candidate / 'brain.json').is_file():
            return candidate
    from .full_merge import resolve_connected_input, validate_connected_descriptor
    current = root_engine
    if resumed and entry.get('lineage'):
        # A prior merged parent owns this saved private runtime. Its immutable
        # descriptor may have moved to the checked engine-local input alias.
        descriptor = validate_connected_descriptor(read_json(current / 'brain.json')['connected_regions'])
        declaration = next((row for row in descriptor['regions'] if row['regionId'] == entry['regionPath'][0]), None)
        if declaration is None:
            raise ValueError('inherited re-suspension lost its owning parent region')
        parent_engine = resolve_connected_input(current, declaration)
        prior_id = entry['lineage'][-1]['executionId']
        prior = next((row for row in validated_connected_executions(read_json(parent_engine / 'brain.json').get('connected_executions'))['entries']
            if row['id'] == prior_id), None)
        if prior is None:
            raise ValueError('nested inherited runtime lost its original queue receipt')
        return resolve_execution_engine(parent_engine, prior, resumed=True)
    if resumed:
        raise FileNotFoundError('inherited private runtime closure is missing')
    for identifier in entry['regionPath']:
        descriptor = validate_connected_descriptor(read_json(current / 'brain.json')['connected_regions'])
        declaration = next((row for row in descriptor['regions'] if row['regionId'] == identifier), None)
        if declaration is None:
            raise ValueError('inherited native cursor does not address a retained original region')
        current = resolve_connected_input(current, declaration)
    return current


def execution_checkpoint_snapshot_entries(engine, *, expected_id=None, expected_sha256=None, brain_id=None):
    """Authenticated active checkpoint inventory, relative to an engine root.

    Only referenced immutable files are yielded. An unfinished directory,
    arbitrary sibling file, live WAL or executable payload is never adopted.
    """
    engine = Path(engine).resolve()
    brain_id = brain_id or read_json(engine / 'brain.json')['brain_id']
    store = ExecutionCheckpointStore(engine, brain_id)
    manifest = store.current(expected_id)
    if manifest is not None:
        if expected_sha256 is not None and _digest(manifest) != expected_sha256:
            raise ValueError('execution snapshot differs from its adopted manifest')
        # Archives retain authentic older private frontiers without claiming
        # their execution is supported by the current clock/kernel contract.
        _verify_files(engine, manifest, require_supported=False)
        directory = '.execution/' + manifest['checkpointId'] + '/'
        names = ['.execution/active.json', directory + 'm-' + _digest(manifest) + '.json',
            *(directory + name for name in manifest['files']), directory + 'diagnostic/manifest.json',
            directory + 'diagnostic/controls.safetensors', directory + 'diagnostic/changed.sqlite']
        for name in names:
            path = _inside(engine, name)
            if not path.is_file():
                raise ValueError('execution snapshot lost a referenced regular file')
            if path.suffix in {'.sqlite3', '.sqlite'}:
                for suffix in ('-wal', '-journal'):
                    sidecar = Path(str(path) + suffix)
                    if sidecar.exists() and sidecar.stat().st_size:
                        raise ValueError('immutable execution payload has an uncommitted journal')
            yield name, path, file_digest(path)
    elif expected_sha256 is not None:
        raise ValueError('the adopted execution checkpoint is absent')
    unloaded = engine / '.execution/unloaded.json'
    if unloaded.exists():
        if store.explicitly_unloaded():
            yield '.execution/unloaded.json', unloaded, file_digest(unloaded)


def connected_execution_snapshot_entries(source_engine):
    """Closed root-owned runtime/result dependencies for snapshot/export.

    Original immutable neural parents are supplied by copy_connected_inputs.
    Re-suspended private owners are copied below an engine-local alias while
    retaining their original brainId/cache/diagnostic bindings.
    """
    source_engine = Path(source_engine).resolve()
    if (source_engine / '.connected-working-publication.json').exists():
        raise RuntimeError('recover an unfinished inherited working publication before snapshot/export')
    metadata = read_json(source_engine / 'brain.json')
    state = validated_connected_executions(metadata.get('connected_executions'))
    emitted = set()
    for entry in state['entries']:
        original = resolve_execution_engine(source_engine, entry)
        for _checked in execution_checkpoint_snapshot_entries(original, expected_id=entry['sourceCheckpointId'],
                expected_sha256=entry['sourceManifestSha256'], brain_id=entry['parentId']):
            pass
        owned = []
        # Normal original sources are frozen declared parents, whose full
        # closure copy_connected_inputs already carries. A root-owned source
        # (or an imported engine-local alias) has no such external parent copy;
        # retain its explicit authenticated runtime, never its whole directory.
        if source_engine in original.parents:
            owned.append((original, entry['sourceEnginePath'], entry['sourceCheckpointId'], entry['sourceManifestSha256']))
        if 'resumeEnginePath' in entry:
            runtime = resolve_execution_engine(source_engine, entry, resumed=True)
            if 'runtimeFiles' in entry:
                _verify_runtime_files(runtime, entry['runtimeFiles'])
            owned.append((runtime, entry['resumeEnginePath'], entry['resumeCheckpointId'], entry['resumeManifestSha256']))
        for runtime, logical_source, checkpoint_id, checkpoint_sha in owned:
            _runtime_files(runtime)
            prefix = 'connected-execution-inputs/' + logical_source + '/'
            files = [(name, _inside(runtime, name), None) for name in
                _RUNTIME_FILES]
            files.extend(execution_checkpoint_snapshot_entries(runtime, expected_id=checkpoint_id,
                expected_sha256=checkpoint_sha, brain_id=entry['parentId']))
            views = runtime / 'state/concept-id-views'
            if views.exists():
                from .concept_id_views import IdView
                if views.is_symlink() or not views.is_dir():
                    raise ValueError('inherited structural ID closure is unsafe')
                for path in views.iterdir():
                    if path.name.startswith('.'):
                        continue
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('inherited structural ID closure is not regular')
                    with path.open('rb') as stream:
                        header = json.loads(stream.readline(4096))
                    descriptor = {**header, 'path': 'state/concept-id-views/' + path.name,
                        'sha256': path.stem, 'bytes': path.stat().st_size}
                    IdView(runtime, descriptor, brain_id=header['brainId'], turn_id=header['turnId'])
                    files.append((descriptor['path'], path, descriptor['sha256']))
            for name, path, expected in files:
                if not path.is_file() or path.is_symlink():
                    raise ValueError('inherited private runtime lost a required owned file')
                logical = prefix + name
                if logical not in emitted:
                    emitted.add(logical)
                    yield logical, path, expected or file_digest(path)
        completion = entry.get('completion')
        if completion is not None:
            path = _inside(source_engine.parent, completion['resultPath'])
            if not path.is_file():
                path = _inside(source_engine / 'connected-execution-inputs', completion['resultPath'])
            if not path.is_file() or file_digest(path) != completion['sha256']:
                raise ValueError('inherited completion result lost its exact checksum')
            logical = 'connected-execution-inputs/' + completion['resultPath']
            if logical not in emitted:
                emitted.add(logical)
                yield logical, path, completion['sha256']


_RUNTIME_FILES = ('brain.json', 'core.safetensors', 'plasticity.safetensors', 'conversation.sqlite3',
                 'state/replay.sqlite3', 'state/working-memory.sqlite3')


def _runtime_files(engine):
    result = {}
    for name in _RUNTIME_FILES:
        path = _inside(engine, name)
        if not path.is_file():
            raise ValueError('inherited runtime has no exact private activity file')
        if path.suffix == '.sqlite3':
            for suffix in ('-wal', '-journal'):
                sidecar = Path(str(path) + suffix)
                if sidecar.exists() and sidecar.stat().st_size:
                    raise ValueError('inherited runtime must freeze/checkpoint its SQLite writes before suspension')
        result[name] = {'bytes': path.stat().st_size, 'sha256': file_digest(path)}
    return result


def _verify_runtime_files(engine, values):
    if _runtime_files(engine) != values:
        raise ValueError('inherited runtime activity changed after its exact suspension seal')


def adopt_connected_executions(brain, snapshots, index, metadata, *, cancelled=None):
    """No decoding at adoption: install checked executable source addresses."""
    from .connected_runtime import _message_id
    entries = []
    for parent in snapshots:
        if cancelled is not None and cancelled():
            raise InterruptedError('full merge stopped before execution adoption')
        engine = _inside(brain.storage_path, parent.engine_relative_path)
        original = suspended_execution_identity(engine, parent.parent_id, policy=brain.resource_policy)
        if original is not None:
            manifest = ExecutionCheckpointStore(engine, parent.parent_id, policy=brain.resource_policy).current(original['checkpointId'])
            human = manifest['presentation']['inputReceipt']['humanMessageId']
            identifier = _digest(['ancestral-native-execution-1', parent.region_id,
                parent.parent_id, parent.generation_sha256, original])
            entries.append({'id': identifier, 'rootTurnId': 'ancestral-' + _digest([brain.brain_id, identifier]),
                'parentId': parent.parent_id, 'regionPath': [parent.region_id],
                'sourceEnginePath': parent.engine_relative_path, 'sourceCheckpointId': original['checkpointId'],
                'sourceManifestSha256': original['manifestSha256'], 'sourceBindingId': original['bindingId'],
                'originalTurnId': manifest['frontier']['turnId'], 'historyParentId': parent.parent_id,
                'historyHumanMessageId': human, 'rootHumanMessageId': _message_id(index, parent.parent_id, human),
                'status': 'queued', 'backend': manifest['backend'], 'lineage': [],
                'minimumContextTokens': manifest['minimumContextTokens'],
                'publishedText': manifest['presentation']['publishedText'],
                'workerStateSha256': original['workerStateSha256']})
        for prior in validated_connected_executions(metadata[parent.parent_id].get('connected_executions'))['entries']:
            # Parent pending cursors retain the original native owner, nested
            # under this registered parent graph. They are not retargeted to
            # the larger connected sampler.
            value = copy.deepcopy(prior)
            identifier = _digest(['reparented-native-execution-1', parent.region_id, parent.generation_sha256, prior['id']])
            value.update(id=identifier, rootTurnId='ancestral-' + _digest([brain.brain_id, identifier]),
                regionPath=[parent.region_id, *prior['regionPath']], historyParentId=parent.parent_id,
                historyHumanMessageId=prior['rootHumanMessageId'],
                rootHumanMessageId=_message_id(index, parent.parent_id, prior['rootHumanMessageId']),
                lineage=[*prior['lineage'], {'parentId': parent.parent_id, 'executionId': prior['id'],
                    'generationSha256': parent.generation_sha256}])
            source_root = Path(parent.engine_relative_path).parent
            for key in ('sourceEnginePath', 'resumeEnginePath'):
                if key in value:
                    value[key] = (source_root / value[key]).as_posix()
            if value['status'] == 'complete':
                value['completion']['resultPath'] = (source_root / value['completion']['resultPath']).as_posix()
                # This is an inherited closed receipt, not a new completed
                # root turn. Keep the original receipt identity/provenance.
            entries.append(value)
    return validated_connected_executions({'format': FORMAT, 'formatVersion': VERSION, 'entries': entries})


def connected_execution_status(brain):
    state = validated_connected_executions(getattr(brain, 'connected_executions', None))
    pending = [entry for entry in state['entries'] if entry['status'] in {'queued', 'suspended'}]
    return {'format': FORMAT, 'pending': len(pending), 'rootMutationHeld': bool(pending),
        'activeExecutionId': getattr(brain, '_connected_execution_active', None),
        'minimumContextTokens': max((entry.get('minimumContextTokens', 0) for entry in pending), default=0),
        'entries': [{'id': entry['id'], 'turnId': entry['rootTurnId'], 'parentId': entry['parentId'],
            'status': entry['status'], 'backend': entry['backend'], 'publishedText': entry.get('publishedText', '')}
            for entry in state['entries']]}


def stop_unloaded_connected_execution(storage, brain_id, execution_id, policy=None):
    """Metadata/history Stop under the caller's instance write lease.

    No neural constructor, model load, original-cursor rewrite or invented
    learning. A visible partial prefix remains real history; its unfinished
    suffix is tombstoned, not undone. Core/plastic/origin bytes stay untouched.
    """
    from .conversation_ledger import NeuralConversationLedger
    from .persistence import atomic_write_json
    from .brain import _iso_now
    storage = Path(storage).resolve(); engine = _inside(storage, 'engine')
    metadata = read_json(engine / 'brain.json')
    if metadata.get('brain_id') != brain_id:
        raise ValueError('inherited Stop has a different root owner')
    state = validated_connected_executions(metadata.get('connected_executions'))
    entry = next((row for row in state['entries'] if row['id'] == execution_id), None)
    if entry is None:
        raise ValueError('inherited Stop does not own a committed queue entry')
    if entry['status'] in {'complete', 'stopped'}:
        completion = entry.get('completion')
        if completion is None:
            return {'brainId': brain_id, 'turnId': entry['rootTurnId'], 'executionId': execution_id,
                'stopped': True, 'turnCommitted': False, 'idempotentCompletion': True, 'actions': []}
        path = _inside(storage, completion['resultPath'])
        if not path.is_file():
            path = _inside(engine / 'connected-execution-inputs', completion['resultPath'])
        if not path.is_file() or file_digest(path) != completion['sha256']:
            raise ValueError('inherited Stop completion receipt lost its checksum')
        return {**read_json(path), 'brainId': brain_id, 'idempotentCompletion': True, 'actions': []}
    source = resolve_execution_engine(engine, entry, resumed='resumeEnginePath' in entry)
    saved = ExecutionCheckpointStore(source, entry['parentId']).current(entry.get('resumeCheckpointId', entry['sourceCheckpointId']))
    if _digest(saved) != entry.get('resumeManifestSha256', entry['sourceManifestSha256']):
        raise ValueError('inherited Stop cursor differs from its committed root binding')
    prefix = saved['presentation']['publishedText']
    message = None
    if prefix:
        message = {'id': _digest(['inherited-stopped-prefix-1', entry['id']]), 'role': 'brain', 'content': prefix,
            'created_at': _iso_now(), 'turn_id': entry['rootTurnId'],
            'attention_epoch': int((metadata.get('fresh_attention_boundary') or {}).get('epoch', 0)),
            'generation_end': 'stopped', 'inheritedNative': {'executionId': execution_id,
                'parentId': entry['parentId'], 'originalTurnId': entry['originalTurnId']}}
    result = {'brainId': brain_id, 'turnId': entry['rootTurnId'], 'executionId': execution_id, 'text': prefix,
        'message': message, 'stopped': True, 'turnCommitted': False, 'actions': [], 'inheritedNative': True,
        'inputReceiptRetained': True, 'learnedRuntimeResident': False, 'idempotentCompletion': False}
    path = _inside(storage, 'engine/connected-executions/' + entry['id'] + '/stopped.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        policy.require_disk(len(canonical(result)) * 2 + len(canonical(metadata)) * 2 + 65536, 'inherited Stop receipt')
    atomic_write_json(path, result)
    entry.update(status='stopped', completion={'resultPath': path.relative_to(storage).as_posix(),
        'sha256': file_digest(path), 'turnId': entry['rootTurnId']})
    ledger = NeuralConversationLedger(engine / 'conversation.sqlite3', brain_id)
    head = metadata['conversation']
    committed = False
    try:
        ledger.truncate_after_head(head['headSequence'], head['headSha256'])
        if message is not None:
            ledger.append('message', (message,))
        metadata.update(connected_executions=state, conversation=ledger.summary())
        atomic_write_json(engine / 'brain.json', metadata)
        committed = True
    finally:
        if not committed:
            ledger.truncate_after_head(head['headSequence'], head['headSha256'])
        ledger.close()
    return result


def require_joint_mutation_allowed(brain, *, operation):
    status = connected_execution_status(brain)
    if status['rootMutationHeld']:
        from .offload import NeuralStateResourcePause
        raise NeuralStateResourcePause('resume inherited native turns before joint ' + operation,
            {**status, 'recoverable': True, 'paused': True, 'inheritedExecutionRequired': True})
    return status


class ConnectedExecutionQueue:
    """One serialized exact-native frontier at a time inside the merged owner."""
    def __init__(self, brain):
        if getattr(brain, '_connected_regions', None) is None:
            raise ValueError('inherited execution requires a connected native brain')
        self.brain = brain
        brain.connected_executions = validated_connected_executions(getattr(brain, 'connected_executions', None))

    def status(self):
        return connected_execution_status(self.brain)

    def next_pending(self):
        return next((copy.deepcopy(value) for value in self.brain.connected_executions['entries']
            if value['status'] in {'queued', 'suspended'}), None)

    def prepare(self, execution_id=None):
        pending = self.next_pending()
        if execution_id is None:
            if pending is None:
                raise ValueError('the connected native execution queue is empty')
            execution_id = pending['id']
        entry = next((value for value in self.brain.connected_executions['entries'] if value['id'] == execution_id), None)
        if entry is None:
            raise ValueError('inherited execution is not owned by this merged brain')
        if entry['status'] in {'complete', 'stopped'}:
            return CompletedConnectedExecution(self.brain, entry)
        if pending['id'] != execution_id:
            raise ValueError('inherited frontiers drain in their committed queue order')
        if getattr(self.brain, '_connected_execution_active', None) is not None:
            raise RuntimeError('another inherited native execution already owns the writer')
        return ConnectedExecutionSession(self.brain, entry)

    def stop(self, execution_id, *, _active_session=None):
        active = getattr(self.brain, '_connected_execution_active', None)
        if active is not None and active != _active_session:
            raise RuntimeError('inherited Stop must use the active native writer boundary')
        result = stop_unloaded_connected_execution(self.brain.storage_path, self.brain.brain_id,
            execution_id, self.brain.resource_policy)
        self.brain.connected_executions = validated_connected_executions(
            read_json(self.brain.engine_path / 'brain.json').get('connected_executions'))
        if result.get('message') is not None and not result.get('idempotentCompletion'):
            self.brain.messages.append(result['message'])
        result['learnedRuntimeResident'] = True
        return result


class CompletedConnectedExecution:
    def __init__(self, brain, entry):
        self.root, self.entry = brain, copy.deepcopy(entry)
        self.root_turn_id = entry['rootTurnId']
        self.original_turn_id = entry['originalTurnId']

    def run(self, **_callbacks):
        completion = self.entry.get('completion')
        if completion is None:
            return {'brainId': self.root.brain_id, 'turnId': self.root_turn_id,
                'stopped': True, 'turnCommitted': False, 'idempotentCompletion': True, 'actions': []}
        path = _inside(self.root.storage_path, completion['resultPath'])
        if not path.is_file():
            path = _inside(self.root.engine_path / 'connected-execution-inputs', completion['resultPath'])
        if not path.is_file() or file_digest(path) != completion['sha256']:
            raise ValueError('completed inherited execution lost its exact root result')
        result = read_json(path)
        if result.get('turnId') != completion['turnId']:
            raise ValueError('completed inherited result has another turn owner')
        if completion['turnId'] != self.root_turn_id:
            result = {**result, 'turnId': self.root_turn_id, 'turnCommitted': False,
                'inheritedHistoricalReceipt': True, 'inheritedClosedTurnId': completion['turnId'],
                'sourceTurnReceipt': result.get('turnReceipt'), 'turnReceipt': None}
        return {**result, 'brainId': self.root.brain_id, 'idempotentCompletion': True, 'actions': [],
            'metrics': self.root.metrics(), 'runtimeCard': self.root.runtime_card()}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()


def persist_borrowed_generation_clock(brain, turn_id, expected_binding):
    """Publish only the paused clock of a read-only regional frontier.

    A borrowed save cannot serialize the root pager under a parent name. Its
    source neural files/pointers remain exact; the independently advancing
    activity clock must nevertheless be written to the NEW private runtime
    before ExecutionCheckpointStore seals its payload and runtime inventory.
    Never stamp a missing clock onto an old cursor or mutate a frozen source.
    """
    from .execution_checkpoint import binding_for, binding_state_for
    from .persistence import atomic_write_json
    if binding_for(brain, turn_id) != expected_binding:
        raise RuntimeError('inherited generation mutated learned state before its finalization boundary')
    clock = brain._owned_activity_clock()
    if not clock.paused:
        raise ValueError('borrowed generation persistence requires a paused activity clock')
    path = brain.engine_path / 'brain.json'
    policy = getattr(brain, 'resource_policy', None)
    reserve = getattr(policy, 'reserve_ram', None)
    with reserve(path.stat().st_size * 12 + 65536, 'borrowed paused clock publication') if callable(reserve) else nullcontext():
        metadata = read_json(path)
        body = binding_state_for(brain, turn_id)
        if (metadata.get('brain_id') != body['brainId']
                or metadata.get('mutable_state') != body['mutable']
                or (metadata.get('substrate') or {}).get('persistence') != body['substrate']
                or int((metadata.get('fresh_attention_boundary') or {}).get('epoch', 0)) != body['attentionEpoch']
                or 'activity_clock' not in metadata):
            raise ValueError('borrowed runtime lost its source generation or persisted activity clock')
        persist_working_clock = getattr(getattr(brain, 'paged_working_memory', None), 'persist_activity_baseline', None)
        if callable(persist_working_clock):
            persist_working_clock()
        metadata['activity_clock'] = clock.snapshot()
        if policy is not None:
            policy.require_disk(len(canonical(metadata)) + 65536, 'borrowed paused clock publication')
        atomic_write_json(path, metadata)


class ConnectedExecutionSession:
    def __init__(self, root, entry):
        from .connected_execution_owner import BorrowedRegionalOwner, private_runtime_engine
        self.root, self.entry = root, entry
        self.root_turn_id, self.original_turn_id = entry['rootTurnId'], entry['originalTurnId']
        self.closed, self.reader, self.owner = False, None, None
        self._new_runtime = None
        self.new_actions = set()
        self.source_actions = {}
        root._connected_execution_active = entry['id']
        try:
            original = resolve_execution_engine(root.engine_path, entry)
            original_manifest = ExecutionCheckpointStore(original, entry['parentId'], policy=root.resource_policy).current(entry['sourceCheckpointId'])
            if _digest(original_manifest) != entry['sourceManifestSha256']:
                raise ValueError('inherited original adopted action frontier lost its exact seal')
            source = resolve_execution_engine(root.engine_path, entry, resumed='resumeEnginePath' in entry)
            if 'runtimeFiles' in entry:
                _verify_runtime_files(source, entry['runtimeFiles'])
            checkpoint = entry.get('resumeCheckpointId', entry['sourceCheckpointId'])
            expected = entry.get('resumeManifestSha256', entry['sourceManifestSha256'])
            store = ExecutionCheckpointStore(source, entry['parentId'], policy=root.resource_policy)
            manifest = store.current(checkpoint)
            if _digest(manifest) != expected or manifest['bindingId'] != entry['sourceBindingId']:
                raise ValueError('inherited execution differs from its adopted source seal')
            _verify_files(source, manifest)
            runtime = _inside(root.storage_path, 'engine/connected-executions/' + entry['id'] + '/runtime/' + uuid.uuid4().hex + '/engine')
            self._new_runtime = runtime
            private_runtime_engine(source, runtime, root.resource_policy)
            self.owner = BorrowedRegionalOwner(root, entry, source, runtime)
            self.owner.enter()
            self.brain = self.owner.brain
            self.manifest, self.payload, self.reader = store.restore(self.brain, checkpoint)
            self.worker_state = copy.deepcopy(manifest['worker'])
            self.worker_state['ancestralExecutionId'] = entry['id']
            self.worker_state['rootTurnId'] = self.root_turn_id
            self.prior_action_ids = {str(action.get('actionId', action.get('id', '')))
                for action in self.payload['continuation']['ledger'].get('actions', {}).values()}
            original_action_ids = {str(action.get('actionId', action.get('id', '')))
                for action in original_manifest['worker']['actions']}
            for action in self.payload['continuation']['ledger'].get('actions', {}).values():
                source_id = action.get('actionId', action.get('id', ''))
                if source_id not in original_action_ids:
                    self.source_actions[self.action_id(source_id)] = copy.deepcopy(action)
            # CheckpointStore.save cannot serialize the root pager's registry
            # under a parent name. Generation itself is read-only; the source
            # neural seals stay valid until the root finalization transaction.
            def staged_save(**_options):
                if not getattr(self, '_finalizing', False):
                    persist_borrowed_generation_clock(self.brain, self.original_turn_id, entry['sourceBindingId'])
            self.brain.save = staged_save
            self.brain._finish_chat_continuation = self._finish
        except BaseException:
            self.close()
            if self._new_runtime is not None:
                from .connected_execution_owner import discard_new_private_runtime
                discard_new_private_runtime(root, entry['id'], self._new_runtime)
            raise

    def action_id(self, identifier):
        return inherited_action_id(self.root.brain_id, self.entry['id'], identifier)

    def ponder_report_id(self, identifier):
        return _digest(['inherited-ponder-self-report-1', self.root.brain_id, self.entry['id'], str(identifier)])

    def argument_owner(self, mapped_action_id):
        """Trusted worker lookup, never a permission-bearing external handle.

        Source ID views belong to this exact registered region/engine. Host
        execution still uses the root grant and mapped effect identity; only
        neural argument inspection borrows the original source owner.
        """
        if self.closed or mapped_action_id not in self.source_actions:
            raise ValueError('inherited neural argument view is not owned by the active writer')
        return self.brain, copy.deepcopy(self.source_actions[mapped_action_id])

    def inward_observation(self, value):
        return _map_observation_routes(self.root.brain_id, self.entry, self.source_actions, value)

    def _outward(self, value):
        """Map protocol identity fields, never user text or source token data."""
        if isinstance(value, list):
            return [self._outward(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: (copy.deepcopy(item) if str(value.get('toolId', '')).startswith('mcp.') else self._outward_arguments(item))
            if key == 'arguments' and isinstance(item, dict)
            else self._outward(item) for key, item in value.items()}
        for key in ('turnId', 'turn_id', 'sourceTurnId'):
            if result.get(key) == self.original_turn_id:
                result[key] = self.root_turn_id
        for key in ('brainId', 'brain_id'):
            if key in result:
                result[key] = self.root.brain_id
        for key in ('actionId', 'nativeActionId', 'neuralActionId', 'parentPonderActionId'):
            if isinstance(result.get(key), str) and result[key] not in self.prior_action_ids:
                result[key] = self.action_id(result[key])
        if result.get('format') == 'omni-ponder-self-report-delta' and isinstance(result.get('reportId'), str):
            result['reportId'] = self.ponder_report_id(result['reportId'])
        if result.get('format') == 'omni-ponder-self-report' and isinstance(result.get('id'), str):
            result['id'] = self.ponder_report_id(result['id'])
        if 'kind' in result and isinstance(result.get('id'), str) and result['id'] not in self.prior_action_ids:
            result['id'] = self.action_id(result['id'])
        return result

    def _outward_arguments(self, value):
        # Ordinary tool arguments are user/model data, not protocol ownership.
        # A JSON document containing a brainId must not be silently rewritten.
        result = copy.deepcopy(value)
        from .concept_id_views import FORMAT as VIEW_FORMAT, IdView, publish_id_view
        for key in ('assemblyIdView', 'conceptIdView'):
            descriptor = value.get(key)
            if isinstance(descriptor, dict) and descriptor.get('format') == VIEW_FORMAT:
                source = IdView(self.brain.engine_path, descriptor, brain_id=self.entry['parentId'],
                    turn_id=str(value.get('sourceTurnId', self.original_turn_id)))
                result[key] = publish_id_view(self.root.engine_path, source, brain_id=self.root.brain_id,
                    turn_id=self.root_turn_id, reserve=self.root.resource_policy.require_disk)
                result['sourceTurnId'] = self.root_turn_id
        report = value.get('ponderSelfReport')
        if isinstance(report, dict) and report.get('format') == 'omni-ponder-self-report':
            result['ponderSelfReport'] = self._outward(report)
        return result

    def outward_event(self, kind, value):
        if ('action' in str(kind) or kind == 'ponder-episode') and isinstance(value, dict):
            action = value.get('action', value)
            if isinstance(action, dict) and isinstance(action.get('actionId', action.get('id')), str):
                identifier = action.get('actionId', action.get('id'))
                if identifier in self.prior_action_ids:
                    return None
                self.new_actions.add(identifier)
                self.source_actions[self.action_id(identifier)] = copy.deepcopy(action)
        return self._outward(copy.deepcopy(value))

    def stop(self):
        return ConnectedExecutionQueue(self.root).stop(self.entry['id'], _active_session=self.entry['id'])

    def run(self, *, stream_callback=None, cancel_check=None, steer_check=None,
            tool_observation_provider=None, chat_control_callback=None,
            suspension_check=None, suspend_callback=None, worker_state=None):
        if self.closed:
            raise RuntimeError('the borrowed native execution has been released')
        def emit(kind, value):
            mapped = self.outward_event(kind, value)
            if mapped is not None and stream_callback is not None:
                stream_callback(kind, mapped)
        def suspended(context, cursor, continuation):
            state = worker_state() if callable(worker_state) else copy.deepcopy(self.worker_state)
            # The source checkpoint and ledger keep their native identities.
            state = {**state, 'brainId': self.entry['parentId'], 'turnId': self.original_turn_id}
            state.pop('ancestralExecutionId', None); state.pop('rootTurnId', None)
            receipt = self.resuspend(context, cursor, continuation, state)
            return suspend_callback(receipt) if suspend_callback is not None else receipt
        return self.brain.resume_chat(self.payload, stream_callback=emit, cancel_check=cancel_check,
            steer_check=steer_check, tool_observation_provider=tool_observation_provider,
            chat_control_callback=chat_control_callback, suspension_check=suspension_check,
            suspend_callback=suspended)

    def resuspend(self, context, cursor, continuation, worker_state):
        from .execution_rng import restore_execution_rng
        from .persistence import atomic_write_json
        state = copy.deepcopy(worker_state)
        if state.get('turnId') != self.original_turn_id or state.get('brainId') != self.entry['parentId']:
            raise ValueError('ancestral re-suspension must retain its original native frontier identity')
        # The caller supplies the full native prefix and existing action/inbox
        # receipts; never derive those from visible root history.
        store = ExecutionCheckpointStore(self.brain.engine_path, self.brain.brain_id, policy=self.root.resource_policy)
        saved = store.save(self.brain, context, cursor, phase=continuation['phase'],
            worker_state=state, journal=context['before_parameters'], continuation=continuation)
        previous = copy.deepcopy(self.entry)
        self.entry.update(status='suspended', resumeEnginePath=self.brain.engine_path.relative_to(self.root.storage_path).as_posix(),
            resumeCheckpointId=saved['checkpointId'], resumeManifestSha256=_digest(saved),
            minimumContextTokens=saved['minimumContextTokens'], publishedText=saved['presentation']['publishedText'],
            runtimeFiles=_runtime_files(self.brain.engine_path))
        restore_execution_rng(self.owner.rng, self.root.device)
        try:
            self.root.save()
        except BaseException:
            committed = read_json(self.root.engine_path / 'brain.json').get('connected_executions')
            if committed != self.root.connected_executions:
                self.entry.clear(); self.entry.update(previous)
                raise
            self.root._connected_recovery_required = True
        return {'brainId': self.root.brain_id, 'turnId': self.root_turn_id, 'executionId': self.entry['id'],
            'suspended': True, 'turnCommitted': False, 'publishedText': saved['presentation']['publishedText'],
            'resume': {**saved['frontier'], 'turnId': self.root_turn_id}, 'inheritedNative': True}

    def _finish(self, context, *, stream_callback=None, cancel_check=None):
        from .brain import AdaptiveBrain
        from .connected_transaction import ConnectedExperienceTransaction
        from .execution_rng import restore_execution_rng
        from .persistence import atomic_write_json
        from .connected_working_publication import ConnectedWorkingPublication
        root = self.root
        fields = ('messages', 'traces', 'counters', 'pending_chat_slow_learning', 'completed_chat_turns', 'accepted_chat_inputs',
                  'recent_token_context', 'recent_token_activity', 'current_context', 'working_memory', 'workspace_items',
                  'memory_lifecycle', 'connected_executions', '_connected_graph_states', 'mutable_state_manifest',
                  'last_activity_decay', 'last_idle_cycle_at', 'last_idle_visible_action_at')
        from .slow_state_snapshot import admit_snapshot_metadata
        admit_snapshot_metadata(root.resource_policy, {name: getattr(root, name) for name in fields}, 'inherited root finalization metadata')
        before = _copy_activity_metadata({name: getattr(root, name) for name in fields})
        head, replay_count = root.conversation.summary(), len(root.replay)
        result_path = _inside(root.storage_path, 'engine/connected-executions/' + self.entry['id'] + '/result.json')
        committed = False
        self._working_publication = ConnectedWorkingPublication(root, self.entry['id'])
        self._finalizing = True
        self._root_checksum_before_finalization = root.parameter_checksum()
        try:
            with ConnectedExperienceTransaction(root) as transaction:
                self._finalization_transaction = transaction
                result = AdaptiveBrain._finish_chat_continuation(self.brain, context,
                    stream_callback=stream_callback, cancel_check=cancel_check)
                if not result.get('turnCommitted'):
                    return self._outward(result)
                self.owner.region.liquid_state.copy_(self.brain.liquid_state)
                result = self._adopt_completion(result)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                root.resource_policy.require_disk(len(canonical(result)) * 2 + 65536, 'inherited completion receipt')
                atomic_write_json(result_path, result)
                self.entry.update(status='complete', completion={'resultPath': result_path.relative_to(root.storage_path).as_posix(),
                    'sha256': file_digest(result_path), 'turnId': self.root_turn_id})
                transaction.prepare_publication()
                restore_execution_rng(self.owner.rng, root.device)
                try:
                    root.save()
                    committed = True
                except BaseException:
                    persisted = validated_connected_executions(read_json(root.engine_path / 'brain.json').get('connected_executions'))
                    committed = any(item['id'] == self.entry['id'] and item.get('completion') == self.entry.get('completion')
                        and item['status'] == 'complete' for item in persisted['entries'])
                    if not committed:
                        raise
                    root._connected_recovery_required = True
        except BaseException:
            if committed:
                root._connected_recovery_required = True
                return {**read_json(result_path), 'reloadRequired': True}
            for name, value in before.items():
                setattr(root, name, value)
            root.conversation.truncate_after_head(head['headSequence'], head['headSha256'])
            root.replay.truncate(replay_count)
            with root.replay._connect() as database:
                if database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='merge_origins'").fetchone():
                    database.execute('DELETE FROM merge_origins WHERE sequence NOT IN (SELECT sequence FROM replay)')
            root._connected_recovery_required = True
            raise
        finally:
            self._finalizing = False
            try:
                self._working_publication.finish()
            except Exception as error:
                # Keep the exact beforeimage/marker for the cold-load helper;
                # no later warm operation may read a mixed temporary store.
                root._connected_recovery_required = True
                root._connected_working_recovery_error = type(error).__name__
        for kind, payload, job_id in self.brain.events.rows:
            try:
                root.events.append(kind, {**self._outward(payload), 'inheritedExecutionId': self.entry['id']}, job_id)
            except Exception as error:
                # A completed neural/history receipt is not undone by derived
                # telemetry failure, nor may lost ACK replay an external effect.
                result['eventLogWarning'] = type(error).__name__
                break
        result['metrics'], result['runtimeCard'] = root.metrics(), root.runtime_card()
        return result

    def _adopt_completion(self, result):
        """Actual source finalization, mapped once into the root identity."""
        from .connected_runtime import _address, _adopt_replay, _adopt_working_page, _embed
        from .recent_token_activity import RecentTokenActivity, token_hash
        root, view = self.root, self.brain
        path = self.entry['regionPath']; region_key = '/'.join(path)
        offset, connected = 0, root._connected_regions
        for identifier in path:
            for region in connected.decoder.regions:
                if region.identity.region_id == identifier:
                    break
                offset += region.config.idea_dim
            connected = (connected.nested or {}).get(identifier)
        def address(identifier):
            for part in reversed(path):
                identifier = _address(part, identifier)
            return identifier
        def provenance(identifier):
            return {'executionId': self.entry['id'], 'parentId': self.entry['parentId'],
                'originalTurnId': self.original_turn_id, 'originalId': identifier, 'regionPath': list(path)}
        new = self._outward(copy.deepcopy(result))
        original_message, original_trace = result['message']['id'], result['trace']['id']
        message_id = _digest(['inherited-reply-1', self.entry['id'], original_message])
        trace_id = _digest(['inherited-trace-1', self.entry['id'], original_trace])
        new['message'].update(id=message_id, turn_id=self.root_turn_id, attention_epoch=root._attention_epoch(), inheritedNative=provenance(original_message))
        new['trace'].update(id=trace_id, turn_id=self.root_turn_id, attention_epoch=root._attention_epoch(), inheritedNative=provenance(original_trace))
        source_trace = result['trace']
        new['trace']['inheritedNative'].update(sourceParameterChecksumBefore=source_trace['parameter_checksum_before'],
            sourceParameterChecksumAfter=source_trace['parameter_checksum_after'],
            sourceParameterDeltaNorm=source_trace['parameter_delta_norm'])
        root_checksum_after = root.parameter_checksum()
        new['trace'].update(parameter_checksum_before=self._root_checksum_before_finalization,
            parameter_checksum_after=root_checksum_after,
            parameter_delta_norm=self._finalization_transaction.journal.delta_norm())
        new['trace']['core_parameter_delta_norm'] = new['trace']['parameter_delta_norm']
        human = root.conversation.payload_by_id('message', self.entry['rootHumanMessageId'])
        if human is None:
            raise ValueError('inherited native input lost its already-adopted root human row')
        new['humanMessage'] = human
        from .native_audio_chat import observed_input_sha256
        input_alias = {'humanMessageId': self.entry['rootHumanMessageId'],
            'originalTurnId': str(human.get('turn_id', '')), 'attentionEpoch': int(human.get('attention_epoch', 0)),
            'inputSha256': observed_input_sha256(human)}
        if input_alias['inputSha256'] != self.payload['context']['input_sha256']:
            raise ValueError('inherited root human alias changed the actual admitted input')
        new['inheritedInputAlias'] = input_alias
        root.messages.append(new['message']); root.traces.append(new['trace'])
        for key, value in view.counters.items():
            delta = int(value) - int(self.owner.before_counters.get(key, 0))
            if delta < 0:
                raise ValueError('native regional counters moved backwards')
            root.counters[key] = int(root.counters.get(key, 0)) + delta
        _adopt_replay(root, view.replay.path, 'execution-' + self.entry['id'], offset,
            view.config.idea_dim, None, after_sequence=self.owner.before_replay_sequence)
        matches = lambda item: item.get('mergeRegionPath', [item.get('mergeRegionId')]) == path
        keep = [(vector, item) for vector, item in zip(root.working_memory, root.workspace_items) if not matches(item)]
        condition = "json_extract(metadata_json,'$.mergeRegionPath')=?"
        arguments = [json.dumps(path, separators=(',', ':'))]
        if len(path) == 1:
            condition += " OR (json_type(metadata_json,'$.mergeRegionPath') IS NULL AND json_extract(metadata_json,'$.mergeRegionId')=?)"
            arguments.append(path[0])
        with root.paged_working_memory._connect() as database:
            old_cold = database.execute('SELECT COUNT(*) FROM working_pages WHERE ' + condition, arguments).fetchone()[0]
        new_cold = view.paged_working_memory.count()
        overflow = max(0, len(keep) + len(view.working_memory) - root.config.working_memory_slots)
        if old_cold or new_cold or overflow:
            self._working_publication.stage()
            with root.paged_working_memory._connect() as database:
                database.execute('DELETE FROM working_pages WHERE ' + condition, arguments)
            from .offload import PagedWorkingMemory
            with view.paged_working_memory._connect() as database:
                for row in database.execute('SELECT sequence,page_id,assembly_id,metadata_json,sha256,dtype,shape_json,payload,updated_at FROM working_pages ORDER BY sequence'):
                    vector, item = PagedWorkingMemory._decode_page_row(row)
                    item['assemblyId'] = address(item.get('assemblyId'))
                    item.update(mergeRegionId=path[0], mergeRegionPath=list(path), mergeOriginalPageId=row[1])
                    _adopt_working_page(root, _digest([self.entry['id'], 'post-completion-cold', row[1], row[4]]),
                        _embed(vector, root.config.idea_dim, offset, view.config.idea_dim), item, row[8],
                        source_activity_now=view.activity_clock.now(), source_activity_at=view.paged_working_memory.activity_timestamp(item, row[8]))
        root.working_memory, root.workspace_items = [row[0] for row in keep], [row[1] for row in keep]
        for position, (vector, raw) in enumerate(zip(view.working_memory, view.workspace_items)):
            item = copy.deepcopy(raw); item['assemblyId'] = address(item.get('assemblyId'))
            item.update(mergeRegionId=path[0], mergeRegionPath=list(path))
            value = _embed(vector, root.config.idea_dim, offset, view.config.idea_dim)
            if len(root.working_memory) < root.config.working_memory_slots:
                root.working_memory.append(value); root.workspace_items.append(item)
            else:
                _adopt_working_page(root, _digest([self.entry['id'], 'post-completion-hot', position]), value, item, 0.,
                    source_activity_now=view.activity_clock.now())
        retained = [(item, vector) for item, vector in zip(root.memory_lifecycle.afterimage_items, root.memory_lifecycle.afterimage_vectors) if not matches(item)]
        root.memory_lifecycle.afterimage_items = [row[0] for row in retained]
        root.memory_lifecycle.afterimage_vectors = [row[1] for row in retained]
        cycle = max(root.memory_lifecycle.cycle, view.memory_lifecycle.cycle)
        for raw, vector in zip(view.memory_lifecycle.afterimage_items, view.memory_lifecycle.afterimage_vectors):
            item = copy.deepcopy(raw)
            for key in ('id', 'assemblyId'):
                item[key] = address(item.get(key))
            item.update(mergeRegionId=path[0], mergeRegionPath=list(path))
            for key in ('enteredCycle', 'lastActiveCycle', 'lastRescoredCycle', 'lastSettledCycle'):
                if type(item.get(key)) is int:
                    item[key] += cycle - view.memory_lifecycle.cycle
            root.memory_lifecycle.afterimage_items.append(item)
            root.memory_lifecycle.afterimage_vectors.append(_embed(vector, root.config.idea_dim, offset, view.config.idea_dim))
        root.memory_lifecycle.active_focus = [item for item in root.memory_lifecycle.active_focus if not matches(item)]
        for raw in view.memory_lifecycle.active_focus:
            item = copy.deepcopy(raw); item['assemblyId'] = address(item.get('assemblyId'))
            item.update(mergeRegionId=path[0], mergeRegionPath=list(path)); root.memory_lifecycle.active_focus.append(item)
        for key, metadata_key in (('settled_experiences', 'settledExperiences'), ('settled_rest_cycles', 'settledRestCycles'),
                                  ('expired_afterimages', 'expiredAfterimages'), ('reinforcement_events', 'reinforcementEvents')):
            setattr(root.memory_lifecycle, key, getattr(root.memory_lifecycle, key) + getattr(view.memory_lifecycle, key)
                - int(self.owner.before_lifecycle.get(metadata_key, 0)))
        root.memory_lifecycle.cycle = cycle
        source_now, root_now = view.activity_clock.now(), root.activity_clock.now()
        for field in ('last_activity_decay', 'last_idle_cycle_at', 'last_idle_visible_action_at'):
            setattr(root, field, root_now - max(0., source_now - float(getattr(view, field))))
        # Replace only this owner's still-live temporary window. This never
        # retrieves history or invents tokens from a neural association.
        owners = root.current_context.get('connectedTokenOwners', [])
        if not owners and root.recent_token_context:
            raise ValueError('merged temporary context lacks its exact source ownership ranges')
        tokens, spans, rewritten, cursor = [], [], [], 0
        activity = root.recent_token_activity.spans
        span_index = span_cursor = 0
        inserted = False
        for owner in owners:
            count = owner['tokenCount']; old = root.recent_token_context[cursor:cursor + count]
            selected = owner['regionPath'] == path
            old_spans = []
            remaining = count
            while remaining:
                span = dict(activity[span_index]); take = min(remaining, span['tokenCount'] - span_cursor)
                span['tokenCount'] = take; old_spans.append(span); remaining -= take; span_cursor += take
                if span_cursor == activity[span_index]['tokenCount']:
                    span_cursor = 0; span_index += 1
            if selected:
                if not inserted:
                    tokens.extend(view.recent_token_context)
                    spans.extend({**span, 'afterimageId': address(span.get('afterimageId'))} for span in view.recent_token_activity.spans)
                    if view.recent_token_context:
                        rewritten.append({'regionPath': list(path), 'tokenCount': len(view.recent_token_context)})
                    inserted = True
            else:
                tokens.extend(old); spans.extend(old_spans); rewritten.append(copy.deepcopy(owner))
            cursor += count
        if not inserted:
            tokens.extend(view.recent_token_context)
            spans.extend({**span, 'afterimageId': address(span.get('afterimageId'))} for span in view.recent_token_activity.spans)
            if view.recent_token_context:
                rewritten.append({'regionPath': list(path), 'tokenCount': len(view.recent_token_context)})
        if len(tokens) > root.config.max_seq_len:
            from .offload import NeuralStateResourcePause
            raise NeuralStateResourcePause('review a larger working window before completing inherited native turns',
                {'recoverable': True, 'paused': True, 'minimumContextTokens': len(tokens)})
        root.recent_token_context, root.recent_token_activity = tokens, RecentTokenActivity(spans)
        root.current_context.update(tokenCount=len(tokens), recentTokenCount=len(tokens), tokenHash=token_hash(tokens),
            recentTokenHash=token_hash(tokens), connectedTokenOwners=rewritten)
        message_map = {str(result['humanMessage']['id']): self.entry['rootHumanMessageId'], original_message: message_id}
        for raw in view.pending_chat_slow_learning:
            if raw['jobId'] in self.owner.before_pending:
                continue
            job = copy.deepcopy(raw)
            job.update(jobId=_digest(['inherited-chat-learning-1', self.entry['id'], raw['jobId']]),
                turnId=self.root_turn_id + (':self' if raw['turnId'].endswith(':self') else ''),
                humanMessageId=message_map[raw['humanMessageId']], afterimageId=address(raw.get('afterimageId')),
                mergeOriginalJobId=raw['jobId'], mergeParentId=self.entry['parentId'])
            root.pending_chat_slow_learning.append(job)
        receipt = mapped_native_turn_receipt(root, result['turnReceipt'], turn_id=self.root_turn_id,
            human_id=self.entry['rootHumanMessageId'], message_id=message_id, trace_id=trace_id, checksum=root_checksum_after)
        # Keep the stable native receipt closed and independently loadable;
        # ancestry/input aliases live in result/trace provenance, not new keys
        # that a future cold native loader would reject after a valid save.
        root._validated_completed_chat_turns([receipt])
        root.completed_chat_turns.append(receipt)
        admitted = copy.deepcopy(self.payload['context'].get('admitted_input'))
        if admitted is not None:
            admitted.update(turnId=self.root_turn_id, humanMessageId=self.entry['rootHumanMessageId'],
                afterimageId=address(admitted.get('afterimageId')), attentionEpoch=root._attention_epoch())
            if 'slowJobId' in admitted:
                mapped_job = next((job for job in root.pending_chat_slow_learning
                    if job.get('mergeOriginalJobId') == admitted['slowJobId']
                    and job.get('mergeParentId') == self.entry['parentId']), None)
                if mapped_job is not None:
                    admitted['slowJobId'] = mapped_job['jobId']
            root.accepted_chat_inputs.append(admitted)
        new.update(brainId=root.brain_id, turnId=self.root_turn_id, turnReceipt=receipt, inputSha256=receipt['inputSha256'],
            inheritedNative=provenance(self.original_turn_id), executionId=self.entry['id'],
            actions=[self._outward(action) for action in result.get('actions', [])
                if action.get('actionId', action.get('id')) not in self.prior_action_ids])
        new.pop('metrics', None); new.pop('runtimeCard', None)
        return new

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.reader is not None:
            self.reader.close()
        if getattr(self, 'payload', None) is not None:
            journal = self.payload['context'].get('before_parameters')
            if journal is not None:
                journal.close()
        if self.owner is not None:
            self.owner.close()
        if getattr(self.root, '_connected_execution_active', None) == self.entry['id']:
            self.root._connected_execution_active = None

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()
