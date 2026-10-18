"""Exact temporary-page beforeimage across an inherited root publication.

Neural graph caches recover from immutable generations. Cold working pages are
different: removing/replacing a parent's pages needs this protected beforeimage
until root brain.json commits. No model or source corpus is copied here.
"""
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid
from contextlib import closing, nullcontext

from .execution_state_codec import file_digest
from .persistence import atomic_write_json, read_json

FORMAT = 'omni-inherited-working-publication'
MARKER = '.connected-working-publication.json'


def _checked(engine, metadata):
    path = engine / MARKER
    if path.is_symlink() or not path.is_file():
        raise ValueError('inherited working publication marker is unsafe')
    value = read_json(path)
    if (not isinstance(value, dict) or set(value) != {'format', 'version', 'brainId', 'executionId', 'beforeRootSha256', 'backup', 'sha256'}
            or value['format'] != FORMAT or value['version'] != 1 or value['brainId'] != metadata.get('brain_id')
            or not re.fullmatch('[a-f0-9]{64}', str(value['executionId']))
            or not re.fullmatch('[a-f0-9]{64}', str(value['beforeRootSha256']))
            or not re.fullmatch('[a-f0-9]{64}', str(value['sha256']))
            or not re.fullmatch('connected-executions/' + value['executionId'] + '/working-before/[a-f0-9]{32}\\.sqlite3', str(value['backup']))):
        raise ValueError('inherited working publication ownership is invalid')
    backup = engine / value['backup']
    current = engine
    for part in Path(value['backup']).parts:
        current /= part
        if current.is_symlink():
            raise ValueError('inherited working beforeimage cannot follow links')
    if not backup.is_file() or file_digest(backup) != value['sha256']:
        raise ValueError('inherited working beforeimage lost its exact checksum')
    return value, backup


def recover_inherited_working_publication(engine, metadata=None, policy=None):
    """Before constructing/loading any live working-page owner; no model load."""
    engine = Path(engine).resolve()
    marker = engine / MARKER
    if not marker.exists():
        return {'recovered': False, 'pending': False}
    metadata = read_json(engine / 'brain.json') if metadata is None else metadata
    value, backup = _checked(engine, metadata)
    from .connected_execution import validated_connected_executions
    entries = validated_connected_executions(metadata.get('connected_executions'))['entries']
    committed = any(row.get('id') == value['executionId'] and row.get('status') == 'complete' for row in entries)
    if committed:
        from .brain import AdaptiveBrain
        receipts = AdaptiveBrain._validated_completed_chat_turns(metadata.get('completed_chat_turns'))
        selected = next(row for row in entries if row['id'] == value['executionId'])
        receipt = next((row for row in receipts if row['turnId'] == selected['rootTurnId']), None)
        completion = selected['completion']
        from .connected_execution import _inside
        result_path = _inside(engine.parent, completion['resultPath'])
        if (receipt is None or not result_path.is_file() or file_digest(result_path) != completion['sha256']
                or read_json(result_path).get('turnReceipt') != receipt):
            raise ValueError('keep inherited working beforeimage until its committed native receipt is valid')
    if not committed:
        if file_digest(engine / 'brain.json') != value['beforeRootSha256']:
            raise ValueError('inherited working recovery found an unrelated root generation')
        target = engine / 'state/working-memory.sqlite3'
        if target.is_symlink():
            raise ValueError('inherited working recovery target is unsafe')
        temporary = target.with_name('.working-restore-' + uuid.uuid4().hex + '.sqlite3')
        if policy is not None:
            policy.require_disk(backup.stat().st_size + 65536, 'exact inherited working crash recovery')
        try:
            with policy.reserve_ram(2 * 1024**2, 'inherited working recovery transfer') if policy is not None else nullcontext():
                shutil.copyfile(backup, temporary)
            if file_digest(temporary) != value['sha256']:
                raise ValueError('inherited working recovery lost beforeimage bytes')
            for suffix in ('-wal', '-shm', '-journal'):
                sidecar = Path(str(target) + suffix)
                if sidecar.exists():
                    if sidecar.is_symlink() or not sidecar.is_file():
                        raise ValueError('inherited working recovery journal is unsafe')
                    sidecar.unlink()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    marker.unlink()
    backup.unlink()
    return {'recovered': not committed, 'pending': False, 'rootCommitted': committed, 'executionId': value['executionId']}


class ConnectedWorkingPublication:
    def __init__(self, root, execution_id):
        self.root, self.execution_id, self.staged = root, execution_id, False

    def stage(self):
        if self.staged:
            return
        engine = self.root.engine_path
        if (engine / MARKER).exists():
            raise RuntimeError('recover the prior inherited working publication before another writer')
        source = self.root.paged_working_memory.path
        if source.is_symlink() or not source.is_file():
            raise ValueError('inherited working source is not an owned database')
        relative = 'connected-executions/' + self.execution_id + '/working-before/' + uuid.uuid4().hex + '.sqlite3'
        backup = engine / relative; backup.parent.mkdir(parents=True, exist_ok=True)
        size = source.stat().st_size * 2 + 131072
        self.root.resource_policy.require_disk(size, 'inherited cold-working exact beforeimage')
        try:
            with self.root.resource_policy.reserve_ram(4 * 1024**2, 'inherited working backup transfer'), \
                    closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as incoming, \
                    closing(sqlite3.connect(backup)) as outgoing:
                incoming.execute('PRAGMA cache_size=-1024'); outgoing.execute('PRAGMA cache_size=-1024')
                incoming.backup(outgoing, pages=256)
                outgoing.execute('PRAGMA journal_mode=DELETE')
                if outgoing.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('inherited working beforeimage failed integrity verification')
            atomic_write_json(engine / MARKER, {'format': FORMAT, 'version': 1, 'brainId': self.root.brain_id,
                'executionId': self.execution_id, 'beforeRootSha256': file_digest(engine / 'brain.json'),
                'backup': relative, 'sha256': file_digest(backup)})
            self.staged = True
        except BaseException:
            backup.unlink(missing_ok=True)
            raise

    def finish(self):
        if self.staged:
            result = recover_inherited_working_publication(self.root.engine_path, policy=self.root.resource_policy)
            self.staged = False
            return result
        return {'recovered': False, 'pending': False}
