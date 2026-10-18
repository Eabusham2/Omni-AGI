"""Safe bounded persistence for an in-flight net-parameter-delta baseline.

No neural model is cloned. Original changed bytes, control baselines, and
structural bindings are serialized separately from the caller's checkpoint.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import torch

from .bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded


FORMAT = "omni-core-delta-resume-1"
CHUNK_BYTES = 64 * 1024
_FLAGS = ("connection_state", "structural_state", "logical_columns", "implicit_zero_birth")
_FILES = ("controls.safetensors", "changed.sqlite")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _module_bindings(roots):
    by_path, by_id = {}, {}
    for prefix, root in roots:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("journal root names must be nonempty strings")
        for name, module in root.named_modules():
            path = prefix + ("." + name if name else "")
            if path in by_path and by_path[path] is not module:
                raise ValueError("journal root names overlap")
            by_path[path] = module
            by_id.setdefault(id(module), path)
    return by_path, by_id


def _class_name(module):
    return type(module).__module__ + "." + type(module).__qualname__


def _plain_file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise ValueError("diagnostic resume file is missing or not a regular local artifact")
    return path


def _rows(database, descriptors):
    """Validate one sparse original block before exposing it to a new writer."""
    for key, block, positions, codes in database.execute("SELECT owner,block,positions,codes FROM original ORDER BY owner,block"):
        item = descriptors.get(key)
        if item is None or item["kind"] != "packed" or item.get("implicit_zero_birth"):
            raise ValueError("diagnostic original row has no original packed owner")
        if type(block) is not int or block < 0 or not isinstance(positions, bytes) or not isinstance(codes, bytes):
            raise ValueError("diagnostic original row has invalid types")
        if not codes or len(codes) > 4096 or len(positions) != 2 * len(codes):
            raise ValueError("diagnostic original row exceeds one sparse block")
        indices = array.array("H")
        indices.frombytes(positions)
        if list(indices) != sorted(set(indices)) or any(index >= 4096 or block * 4096 + index >= item["elements"] for index in indices):
            raise ValueError("diagnostic original offsets exceed their named owner")
        if not item.get("connection_state") and not item.get("structural_state") and any((code >> shift) & 3 == 3 for code in codes for shift in (0, 2, 4, 6)):
            raise ValueError("diagnostic baseline contains a reserved ternary sign")
        yield key, block, positions, codes


def export_journal_state(journal, destination, *, binding_id):
    if journal.closed:
        raise RuntimeError("cannot export a closed parameter journal")
    if not _sha(binding_id):
        raise ValueError("diagnostic checkpoint binding must be a SHA-256 identity")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("diagnostic resume destination already exists")
    # Verify that no unjournaled mutation is hidden by a suspension boundary.
    if not math.isfinite(journal.delta_norm()):
        raise ValueError("diagnostic baseline contains non-finite controls")
    _named, names = _module_bindings(journal.roots)
    descriptors, controls = [], {}
    for key, item in journal.original.items():
        owner_path = names.get(id(item["module"]))
        if owner_path is None:
            raise RuntimeError("diagnostic original owner is no longer named")
        descriptor = {"key": key, "kind": item["kind"], "owner": owner_path,
                      "class": _class_name(item["module"]), "name": item["name"]}
        if item["kind"] == "control":
            controls[key] = item["baseline"]
            descriptor.update(shape=list(item["baseline"].shape), dtype=str(item["baseline"].dtype))
        else:
            descriptor.update(shape=list(item["shape"]), elements=item["elements"])
            descriptor.update((flag, item[flag]) for flag in _FLAGS if flag in item)
            if item.get("implicit_zero_birth"):
                parent = names.get(id(item["birth_parent"]))
                if parent is None:
                    raise RuntimeError("diagnostic sparse birth parent is not named")
                descriptor.update(birthParent=parent, birthKey=item["birth_key"])
        descriptors.append(descriptor)
    topology, seen = [], set()
    for owner, boundary in journal._parallel_topology:
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        path = names.get(id(owner))
        if path is None:
            raise RuntimeError("diagnostic parallel owner is not named")
        topology.append({"owner": path, "class": _class_name(owner), "boundary": boundary})
    containers = [path for path, _owner in journal._sparse_containers.values()]
    control_bytes = sum(value.numel() * value.element_size() for value in controls.values())
    source_db_bytes = 0 if journal.database is None else (journal.private_directory / "changed.sqlite").stat().st_size
    metadata_allowance = (len(descriptors) + len(topology)) * 2048 + 131072
    journal.reserve_ram(metadata_allowance + 2 * CHUNK_BYTES)
    journal.reserve_disk(control_bytes + source_db_bytes * 2 + metadata_allowance + 131072, "paused parameter diagnostic baseline")
    reserve_spill = getattr(journal.resource_policy, 'reserve_spill', None)
    quotas, private, published = {}, None, False
    try:
        for name, size in ((_FILES[0], control_bytes + metadata_allowance),
                           (_FILES[1], source_db_bytes * 2 + 131072), ('manifest.json', metadata_allowance)):
            if callable(reserve_spill): quotas[name] = reserve_spill(size, 'paused diagnostic ' + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix=".diagnostic-", dir=str(destination.parent)))
        for name, quota in quotas.items(): quota.bind_path(private / name)
        atomic_save_tensors_bounded(private / _FILES[0], controls, metadata={"format": FORMAT},
            chunk_bytes=CHUNK_BYTES, spill_lease=quotas.get(_FILES[0]))
        output = sqlite3.connect(str(private / _FILES[1]))
        try:
            output.execute("PRAGMA cache_size=-64")
            output.execute("PRAGMA journal_mode=DELETE")
            output.execute("CREATE TABLE original (owner TEXT, block INTEGER, positions BLOB, codes BLOB, PRIMARY KEY(owner,block)) WITHOUT ROWID")
            if journal.database is not None:
                for row in _rows(journal.database, {item['key']: item for item in descriptors}):
                    output.execute("INSERT INTO original VALUES (?,?,?,?)", row)
            output.commit()
        finally:
            output.close()
        manifest = {"format": FORMAT, "bindingId": binding_id, "includeState": journal.include_state,
                    "owners": descriptors, "parallelTopology": topology, "sparseContainers": containers,
                    "files": {name: {"sha256": _hash_file(private / name), "bytes": (private / name).stat().st_size} for name in _FILES}}
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with (private / "manifest.json").open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        for name in _FILES:
            with (private / name).open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(private, destination)
        published = True
        for name, quota in quotas.items():
            quota.bind_path(destination / name)
            quota.commit(path=destination / name)
        return {"format": FORMAT, "bindingId": binding_id,
                "manifestSha256": hashlib.sha256(encoded).hexdigest(), "ownerCount": len(descriptors),
                "bytes": sum(item['bytes'] for item in manifest['files'].values()) + len(encoded)}
    except BaseException:
        if private is not None and private.exists():
            shutil.rmtree(private)
        if published:
            # This exact directory was absent at entry and published only by
            # this attempt. A failed accounting commit must not leave an
            # unreceipted partial continuation for a later load to discover.
            shutil.rmtree(destination)
        raise
    finally:
        for quota in quotas.values(): quota.release()


def import_journal_state(cls, roots, source, *, binding_id, expected_manifest_sha256,
                         directory, reserve_ram, reserve_disk, resource_policy=None):
    from .parameter_diagnostics import _OBSERVERS, _CALL

    if not _sha(binding_id) or not _sha(expected_manifest_sha256):
        raise ValueError("diagnostic resume requires verified checkpoint and manifest identities")
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("diagnostic resume source must be a local artifact directory")
    path = _plain_file(source, 'manifest.json')
    reserve_ram(path.stat().st_size * 6 + 2 * CHUNK_BYTES + 131072)
    if _hash_file(path) != expected_manifest_sha256:
        raise ValueError("diagnostic resume manifest checksum mismatch")
    manifest = json.loads(path.read_text(encoding='utf-8'))
    fields = {"format", "bindingId", "includeState", "owners", "parallelTopology", "sparseContainers", "files"}
    if set(manifest) != fields or manifest['format'] != FORMAT or manifest['bindingId'] != binding_id or type(manifest['includeState']) is not bool:
        raise ValueError("diagnostic resume manifest or checkpoint binding is invalid")
    if set(manifest['files']) != set(_FILES):
        raise ValueError("diagnostic resume file inventory is invalid")
    for name in _FILES:
        item = manifest['files'][name]
        path = _plain_file(source, name)
        if set(item) != {'sha256', 'bytes'} or not _sha(item['sha256']) or type(item['bytes']) is not int or item['bytes'] != path.stat().st_size or _hash_file(path) != item['sha256']:
            raise ValueError("diagnostic resume artifact checksum mismatch")
    if not all(isinstance(manifest[name], list) for name in ('owners', 'parallelTopology', 'sparseContainers')):
        raise ValueError("diagnostic resume owner inventory is invalid")
    journal = cls.__new__(cls)
    journal.roots, journal.directory = tuple(roots), Path(directory)
    journal.reserve_ram, journal.reserve_disk = reserve_ram, reserve_disk
    journal.database = journal.private_directory = None
    journal.closed = False
    journal.include_state, journal.resource_policy = manifest['includeState'], resource_policy
    journal._disk_quota, journal._sidecar_quotas = None, []
    journal.original, journal.by_module, journal._sparse_containers, journal._parallel_topology = {}, {}, {}, []
    try:
        inventory = journal.inventory()
        named, _ids = _module_bindings(journal.roots)
        controls = BoundedTensorFile(source / _FILES[0], chunk_bytes=CHUNK_BYTES)
        expected_controls = set()
        reserve_ram(131072 + len(manifest['owners']) * 2048 + sum(spec.byte_count for spec in controls.specs.values()) * 2)
        for descriptor in manifest['owners']:
            key = descriptor.get('key') if isinstance(descriptor, dict) else None
            if not isinstance(key, str) or key in journal.original or key not in inventory:
                raise ValueError("diagnostic original owner binding is missing or duplicated")
            item = dict(inventory[key])
            module = named.get(descriptor.get('owner'))
            if module is not item['module'] or descriptor.get('class') != _class_name(module) or descriptor.get('kind') != item['kind'] or descriptor.get('name') != item['name']:
                raise ValueError("diagnostic original owner identity changed")
            permitted = {'key', 'kind', 'owner', 'class', 'name', 'shape', 'dtype'} if item['kind'] == 'control' else {'key', 'kind', 'owner', 'class', 'name', 'shape', 'elements', *_FLAGS, 'birthParent', 'birthKey'}
            if not set(descriptor) <= permitted:
                raise ValueError("diagnostic owner has unsupported metadata")
            if item['kind'] == 'control':
                value = item['tensor']
                if key not in controls.specs or descriptor['shape'] != list(value.shape) or descriptor['dtype'] != str(value.dtype):
                    raise ValueError("diagnostic control shape or dtype changed")
                spec = controls.specs[key]
                if spec.shape != tuple(value.shape) or spec.dtype != value.dtype:
                    raise ValueError("diagnostic control baseline does not match its loaded owner")
                item['baseline'] = controls.tensor(key)
                if item['baseline'].is_floating_point() and not bool(torch.isfinite(item['baseline']).all()):
                    raise ValueError("diagnostic control baseline must be finite")
                expected_controls.add(key)
            else:
                if descriptor.get('shape') != list(item['shape']) or descriptor.get('elements') != item['elements']:
                    raise ValueError("diagnostic packed owner shape changed")
                for flag in ('connection_state', 'structural_state', 'logical_columns'):
                    if descriptor.get(flag, False) != item.get(flag, False):
                        raise ValueError("diagnostic packed owner representation changed")
                if 'implicit_zero_birth' in descriptor and type(descriptor['implicit_zero_birth']) is not bool:
                    raise ValueError("diagnostic sparse birth flag is invalid")
                if descriptor.get('implicit_zero_birth'):
                    parent = named.get(descriptor.get('birthParent'))
                    block_key = descriptor.get('birthKey')
                    if not getattr(parent, 'packed_synapse_container', False) or not isinstance(block_key, str) or block_key not in parent.blocks or parent.blocks[block_key] is not module:
                        raise ValueError("diagnostic sparse birth no longer has its exact named parent")
                    item.update(implicit_zero_birth=True, birth_parent=parent, birth_key=block_key)
                journal.by_module[id(module), item['name']] = item
                journal.mark_version(item)
            journal.original[key] = item
        if set(controls.specs) != expected_controls:
            raise ValueError("diagnostic resume contains unrelated control baselines")
        seen = set()
        for record in manifest['parallelTopology']:
            module = named.get(record.get('owner')) if isinstance(record, dict) else None
            if module is None or id(module) in seen or set(record) != {'owner', 'class', 'boundary'} or record['class'] != _class_name(module):
                raise ValueError("diagnostic parallel topology owner is invalid")
            capture = getattr(module, 'capture_packed_topology_boundary', None)
            if not callable(capture) or not callable(getattr(module, 'restore_topology_boundary', None)):
                raise ValueError("diagnostic parallel topology cannot bind its native owner")
            owner, current_boundary = capture()
            boundary = record['boundary']
            if owner is not module or not isinstance(boundary, list) or len(boundary) != 2 or type(boundary[0]) is not int or not 0 <= boundary[0] <= current_boundary[0]:
                raise ValueError("diagnostic parallel topology boundary is invalid")
            prior, current = {}, dict(current_boundary[1])
            if not isinstance(boundary[1], list):
                raise ValueError("diagnostic parallel page boundary is invalid")
            for entry in boundary[1]:
                if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or type(entry[1]) is not int or entry[0] in prior or entry[0] not in current or not 0 <= entry[1] <= current[entry[0]]:
                    raise ValueError("diagnostic parallel page boundary is invalid")
                prior[entry[0]] = entry[1]
            journal._parallel_topology.append((module, (boundary[0], tuple(prior.items()))))
            seen.add(id(module))
        for path in manifest['sparseContainers']:
            module = named.get(path) if isinstance(path, str) else None
            if not getattr(module, 'packed_synapse_container', False) or id(module) in journal._sparse_containers:
                raise ValueError("diagnostic sparse container binding is invalid")
            journal._sparse_containers[id(module)] = (path, module)
        database = sqlite3.connect((source / _FILES[1]).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            database.execute('PRAGMA query_only=ON')
            database.execute('PRAGMA cache_size=-64')
            if database.execute("SELECT type FROM sqlite_master WHERE name='original'").fetchone() != ('table',):
                raise ValueError("diagnostic original storage is not a table")
            descriptors = {item['key']: item for item in manifest['owners']}
            # Validate the entire sparse input before creating mutable state.
            for _row in _rows(database, descriptors): pass
            if database.execute('SELECT 1 FROM original LIMIT 1').fetchone() is not None:
                size = manifest['files'][_FILES[1]]['bytes'] * 2 + 131072
                reserve_disk(size, 'resume sparse parameter diagnostic originals')
                journal._open()
                reserve = getattr(resource_policy, 'reserve_spill', None)
                quota = reserve(size, 'resume sparse parameter diagnostic originals') if callable(reserve) else None
                path = journal.private_directory / 'changed.sqlite'
                try:
                    if quota is not None: quota.bind_path(path)
                    for row in _rows(database, descriptors):
                        journal.database.execute('INSERT INTO original VALUES (?,?,?,?)', row)
                    journal.database.commit()
                    if quota is not None:
                        quota.commit(path=path)
                        if journal._disk_quota is not None:
                            journal._sidecar_quotas.append(journal._disk_quota)
                        journal._disk_quota = quota
                except BaseException:
                    if quota is not None: quota.release()
                    raise
        finally:
            database.close()
        # No loaded neural bytes or owner counters have been modified.
        _OBSERVERS.set((*_OBSERVERS.get(), journal))
        call = _CALL.get()
        if call is not None: call.append(journal)
        return journal
    except BaseException:
        journal.close()
        raise
