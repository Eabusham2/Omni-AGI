"""Compact learned strength and metaplasticity for native ternary contacts.

The ternary sign remains authoritative in its existing two-bit stream. One
additional byte holds a strength code (low nibble, integer 1..16) and a
resistance code (high nibble, integer 0..15). Zero bytes therefore reproduce
the original unit-strength native checkpoint exactly. These are discrete
learned values, not a floating master-weight shadow.
"""

from __future__ import annotations

import heapq
import re
import weakref
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from .native_core_paging import NativePackedModuleMixin
from .parameter_diagnostics import packed_diagnostic_write, packed_diagnostic_restore, active_diagnostic_scopes


CONNECTION_STATE_VERSION = 1
CONNECTION_STRENGTH_MAX = 16
CONNECTION_RESISTANCE_MAX = 15
CONTACT_PAGE_CAPACITY = 1024
CONTACT_ROW_BUCKET = 32
CONTACT_PAGE_METADATA_ALLOWANCE = 8192


def connection_strengths(state: torch.Tensor) -> torch.Tensor:
    if state.dtype != torch.uint8:
        raise ValueError("connection state must use uint8 storage")
    return (state & 15).to(torch.int16) + 1


def connection_resistance(state: torch.Tensor) -> torch.Tensor:
    if state.dtype != torch.uint8:
        raise ValueError("connection state must use uint8 storage")
    return state >> 4


def encode_connection_state(strength: torch.Tensor, resistance: torch.Tensor) -> torch.Tensor:
    if strength.shape != resistance.shape or strength.device != resistance.device:
        raise ValueError("connection strength and resistance must align")
    if strength.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("connection strength must be an integer")
    if resistance.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("connection resistance must be an integer")
    if bool(((strength < 1) | (strength > CONNECTION_STRENGTH_MAX)).any()):
        raise ValueError("connection strength must be between 1 and 16")
    if bool(((resistance < 0) | (resistance > CONNECTION_RESISTANCE_MAX)).any()):
        raise ValueError("connection resistance must be between 0 and 15")
    return ((strength - 1).to(torch.uint8) | (resistance.to(torch.uint8) << 4)).contiguous()


def weighted_ternary_levels(levels: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    """Decode a bounded compute tile; never create a whole-model mirror."""
    if state is None:
        return levels
    if levels.shape != state.shape or levels.device != state.device:
        raise ValueError("ternary signs and connection state must align")
    return (levels.to(torch.int16) * connection_strengths(state)).to(torch.int8)


class PackedContactPage(NativePackedModuleMixin, nn.Module):
    """A bounded page of independent contacts; no Python object per contact.

    Metadata rows are [row-in-bucket, presence, little-endian uint64 column].
    Presence 0 is unused, 1 live, 2 a tombstone. Slots are never reused after
    committed removal; the page ID and slot define the stable contact ordinal.
    """
    _packed_structural_fields = frozenset({"_record_meta"})

    def __init__(self, *, pager=None):
        super().__init__()
        self._init_native_paging()
        self._native_core_pager = pager
        self._native_loading_checkpoint = False
        self._native_cached_buffers = frozenset({"_sort_order"})
        self.register_buffer("_packed_sign", self._native_buffer("_packed_sign", (1, CONTACT_PAGE_CAPACITY // 4), fill=0x55))
        self.register_buffer("_connection_state", self._native_buffer("_connection_state", (1, CONTACT_PAGE_CAPACITY), fill=0))
        self.register_buffer("_record_meta", self._native_buffer("_record_meta", (CONTACT_PAGE_CAPACITY, 10), fill=0))
        self.register_buffer("_sort_order", self._native_buffer("_sort_order", (CONTACT_PAGE_CAPACITY, 2), fill=0), persistent=False)
        self._header_version = None
        self._order_version = None
        self._used = self._live = 0
        self._finish_native_paging()

    def authoritative_packed_tensors(self):
        return self._packed_sign, self._connection_state, self._record_meta

    def _apply(self, fn):
        # Endpoint pages are compact CPU/paged authority. A parent .to(cuda)
        # must not transfer every repeated-contact page onto the accelerator.
        return self

    def header(self):
        signature = (id(self._record_meta), int(self._record_meta._version))
        if signature != self._header_version:
            flags = self._record_meta[:, 1]
            if bool((flags > 2).any()):
                raise ValueError("contact page contains an invalid presence flag")
            occupied = torch.nonzero(flags, as_tuple=False).flatten()
            self._used = int(occupied[-1]) + 1 if occupied.numel() else 0
            if self._used and not bool(flags[:self._used].ne(0).all()):
                raise ValueError("contact page has reused or noncontiguous slots")
            self._live = int(flags.eq(1).sum())
            self._header_version = signature
        return self._used, self._live

    def _record(self, slot):
        values = self._record_meta[slot].tolist()
        if values[1] != 1:
            return None
        code = int(self._packed_sign[0, slot // 4]) >> (2 * (slot % 4)) & 3
        if code == 3:
            raise ValueError("contact page contains a reserved ternary code")
        return values[0], int.from_bytes(bytes(values[2:]), "little"), 0x54 | code, int(self._connection_state[0, slot])

    @contextmanager
    def _slot_transaction(self, slot):
        scopes = active_diagnostic_scopes()
        fields = (("_record_meta", slot * 10, 10), ("_packed_sign", slot // 4, 1), ("_connection_state", slot, 1))
        originals = tuple((name, start, self._buffers[name].reshape(-1)[start:start + count].clone()) for name, start, count in fields)
        try:
            yield
        except BaseException:
            for name, start, original in originals:
                target = self._buffers[name]
                with packed_diagnostic_restore(self, target, original, start=start, captured_scopes=scopes):
                    target.reshape(-1)[start:start + original.numel()].copy_(original)
            raise

    @torch.no_grad()
    def write(self, slot, row_offset, column, sign, state, *, remove=False):
        with self._packed_residency_scope(torch.device("cpu")), self._slot_transaction(slot):
            used, live = self.header()
            was_live = int(self._record_meta[slot, 1]) == 1
            shift = 2 * (slot % 4)
            packed = self._packed_sign
            new_sign = torch.tensor([[(int(packed[0, slot // 4]) & ~(3 << shift)) | ((1 if remove else sign & 3) << shift)]], dtype=torch.uint8)
            with packed_diagnostic_write(self, packed, new_sign, start=slot // 4):
                packed[0, slot // 4] = new_sign[0, 0]
            strength = torch.tensor([[0 if remove else state]], dtype=torch.uint8)
            with packed_diagnostic_write(self, self._connection_state, strength, start=slot):
                self._connection_state[0, slot] = strength[0, 0]
            record = torch.tensor([[row_offset, 2 if remove else 1, *int(column).to_bytes(8, "little")]], dtype=torch.uint8)
            if not torch.equal(self._record_meta[slot], record[0]):
                with packed_diagnostic_write(self, self._record_meta, record, start=slot * 10):
                    self._record_meta[slot].copy_(record[0])
                self._used = max(used, slot + 1)
                self._live = live + int(not remove) - int(was_live)
                self._header_version = (id(self._record_meta), int(self._record_meta._version))

    @torch.no_grad()
    def truncate(self, used):
        """Discard only transaction-born tail slots during exact rollback."""
        if not 0 <= used <= self.header()[0]:
            raise ValueError("contact rollback boundary is invalid")
        with self._packed_residency_scope(torch.device("cpu")):
            for name, start, fill in (("_record_meta", used * 10, 0), ("_connection_state", used, 0)):
                target = self._buffers[name]
                replacement = torch.full((target.numel() - start,), fill, dtype=torch.uint8)
                with packed_diagnostic_write(self, target, replacement, start=start):
                    target.reshape(-1)[start:].copy_(replacement)
            start = used // 4
            replacement = torch.full((self._packed_sign.numel() - start,), 0x55, dtype=torch.uint8)
            if used % 4:
                mask = (1 << (2 * (used % 4))) - 1
                replacement[0] = (int(self._packed_sign[0, start]) & mask) | (0x55 & ~mask)
            with packed_diagnostic_write(self, self._packed_sign, replacement, start=start):
                self._packed_sign.reshape(-1)[start:].copy_(replacement)

    def ordered_slots(self):
        signature = (id(self._record_meta), int(self._record_meta._version))
        if signature != self._order_version:
            pager = self._native_core_pager
            policy = getattr(pager, "shared_resource_policy", None)
            reserve = getattr(policy, "reserve_ram", None)
            lease = reserve(CONTACT_PAGE_CAPACITY * 192 + 4096, "contact page sort scratch") if callable(reserve) else nullcontext()
            with lease:
                records = [(record[0], record[1], slot) for slot in range(self.header()[0]) if (record := self._record(slot)) is not None]
                records.sort()
                order = torch.tensor([slot for _row, _column, slot in records], dtype=torch.int32)
                self._sort_order.zero_()
                self._sort_order[:len(records), 0].copy_((order & 255).to(torch.uint8))
                self._sort_order[:len(records), 1].copy_((order >> 8).to(torch.uint8))
                self._order_live = len(records)
            self._order_version = signature
        return self._order_live

    def validate_bounded_state_load(self):
        expected = {"_packed_sign": (1, CONTACT_PAGE_CAPACITY // 4), "_connection_state": (1, CONTACT_PAGE_CAPACITY), "_record_meta": (CONTACT_PAGE_CAPACITY, 10)}
        for name, shape in expected.items():
            value = self._buffers[name]
            if value.dtype != torch.uint8 or tuple(value.shape) != shape:
                raise ValueError("contact page buffer shape or dtype is invalid")
        self.header()
        codes = torch.stack(tuple((self._packed_sign.reshape(-1) >> shift) & 3 for shift in (0, 2, 4, 6)), dim=1).reshape(-1)
        live = self._record_meta[:, 1].eq(1)
        if bool(self._record_meta[live, 0].ge(CONTACT_ROW_BUCKET).any()):
            raise ValueError("contact page row offset exceeds its indexed bucket")
        if bool(codes.eq(3).any()) or bool((codes[~live] != 1).any()) or bool(self._connection_state.reshape(-1)[~live].ne(0).any()):
            raise ValueError("contact page has invalid signs or nonzero inactive state")


class PooledParallelConnections(NativePackedModuleMixin, nn.Module):
    """Paged, row-indexed independent contacts with stable non-reused IDs."""
    _packed_structural_fields = frozenset({"_next_page"})

    def __init__(self, weight_shape, bias_columns=0, *, pager=None):
        super().__init__()
        self.weight_shape = tuple(int(value) for value in weight_shape)
        self.bias_columns = int(bias_columns)
        self._native_core_pager = pager
        self.pages = nn.ModuleDict()
        self.pages._native_sparse_registry = True
        self.pages._native_core_pager_ref = weakref.ref(pager) if pager is not None else None
        self.register_buffer("_next_page", torch.zeros(8, dtype=torch.uint8))
        self._by_bucket = {}
        self._by_id = {}
        self._revision = 0
        self.register_load_state_dict_post_hook(self._after_load)

    def _apply(self, fn):
        return self

    def authoritative_packed_tensors(self):
        return (self._next_page,)

    def __len__(self):
        return sum(page.header()[1] for page in self.pages.values())

    def count(self, *, bias=False):
        role = "b" if bias else "w"
        return sum(self.pages[key].header()[1] for (part, _bucket), keys in self._by_bucket.items() if part == role for key in keys)

    @staticmethod
    def parse_key(key):
        match = re.fullmatch(r"(w|b)_r(0|[1-9][0-9]*)_c(0|[1-9][0-9]*)_n(0|[1-9][0-9]*)", key) if isinstance(key, str) else None
        if match is None:
            raise ValueError("parallel connection key is invalid")
        role, row, column, ordinal = match.groups()
        return role, int(row), int(column), int(ordinal)

    @staticmethod
    def parse_page_key(key):
        match = re.fullmatch(r"(w|b)_r(0|[1-9][0-9]*)_p(0|[1-9][0-9]*)", key)
        if match is None:
            raise ValueError("contact page key is invalid")
        role, bucket, identifier = match.groups()
        bucket, identifier = int(bucket), int(identifier)
        if bucket % CONTACT_ROW_BUCKET or identifier * CONTACT_PAGE_CAPACITY >= 1 << 64:
            raise ValueError("contact page identity or row bucket is invalid")
        return role, bucket, identifier

    def _geometry(self, bias):
        return (1, self.bias_columns) if bias else self.weight_shape

    def _validate_endpoint(self, row, column, bias):
        rows, columns = self._geometry(bias)
        if type(row) is not int or type(column) is not int or not 0 <= row < rows or not 0 <= column < columns:
            raise ValueError("parallel connection endpoint is outside its projection")

    def _counter(self):
        return int.from_bytes(bytes(self._next_page.tolist()), "little")

    @torch.no_grad()
    def _set_counter(self, value):
        replacement = torch.tensor(list(int(value).to_bytes(8, "little")), dtype=torch.uint8)
        with packed_diagnostic_write(self, self._next_page, replacement):
            self._next_page.copy_(replacement)

    def _register_page(self, key):
        role, bucket, identifier = self.parse_page_key(key)
        if identifier in self._by_id and self._by_id[identifier] != key:
            raise ValueError("contact page identifier is duplicated")
        rows, _ = self._geometry(role == "b")
        if bucket >= rows:
            raise ValueError("contact page is outside its projection")
        if key in self.pages:
            return self.pages[key]
        pager = self._native_core_pager
        if pager is not None and pager.reserve_admission is not None:
            pager.reserve_admission(CONTACT_PAGE_METADATA_ALLOWANCE, torch.device("cpu"))
        policy = getattr(pager, "shared_resource_policy", None)
        reserve = getattr(policy, "reserve_ram", None)
        lease = reserve(CONTACT_PAGE_METADATA_ALLOWANCE, "repeated contact page registry") if callable(reserve) else nullcontext()
        with lease as held, pager.construction() if pager is not None else nullcontext():
            page = PackedContactPage(pager=pager)
            self.pages[key] = page
            self._by_id[identifier] = key
            self._by_bucket.setdefault((role, bucket), []).append(key)
            if held is not None:
                held.mark_allocated(CONTACT_PAGE_METADATA_ALLOWANCE)
        return page

    def _retire_page(self, key):
        role, bucket, identifier = self.parse_page_key(key)
        page = self.pages[key]
        if page._native_core_pager is not None:
            page._native_core_pager.retire(page)
        del self.pages[key]
        del self._by_id[identifier]
        self._by_bucket[role, bucket].remove(key)
        if not self._by_bucket[role, bucket]:
            del self._by_bucket[role, bucket]
        self._revision += 1

    def capture_packed_topology_boundary(self):
        return self, (self._counter(), tuple((key, page.header()[0]) for key, page in self.pages.items()))

    def iter_identity_structure(self):
        yield ("next-page", self._counter())
        for key in sorted(self.pages, key=lambda value: self.parse_page_key(value)[2]):
            yield (key, *self.pages[key].header())

    def restore_topology_boundary(self, boundary):
        counter, entries = boundary
        before = dict(entries)
        if not before.keys() <= self.pages.keys():
            raise RuntimeError("an original contact page disappeared before rollback")
        for key in tuple(self.pages):
            if key not in before:
                self._retire_page(key)
            else:
                self.pages[key].truncate(before[key])
        self._set_counter(counter)
        self._revision += 1

    @torch.no_grad()
    def add(self, row, column, sign=0, strength=1, *, bias=False):
        self._validate_endpoint(row, column, bias)
        if type(sign) is not int or sign not in (-1, 0, 1) or type(strength) is not int or not 1 <= strength <= 16:
            raise ValueError("parallel connection sign or strength is invalid")
        role, bucket = "b" if bias else "w", row // CONTACT_ROW_BUCKET * CONTACT_ROW_BUCKET
        keys = self._by_bucket.get((role, bucket), ())
        key = next((value for value in reversed(keys) if self.pages[value].header()[0] < CONTACT_PAGE_CAPACITY), None)
        born = key is None
        if born:
            identifier = self._counter()
            key = "%s_r%d_p%d" % (role, bucket, identifier)
            page = self._register_page(key)
        else:
            page = self.pages[key]
            identifier = self.parse_page_key(key)[2]
        slot = page.header()[0]
        try:
            page.write(slot, row - bucket, column, 0x54 | (sign + 1), strength - 1)
            if born:
                self._set_counter(identifier + 1)
        except BaseException:
            if born:
                self._retire_page(key)
            else:
                page.truncate(slot)
            raise
        self._revision += 1
        return "%s_r%d_c%d_n%d" % (role, row, column, identifier * CONTACT_PAGE_CAPACITY + slot)

    def update(self, key, sign, state):
        role, row, column, ordinal = self.parse_key(key)
        identifier, slot = divmod(ordinal, CONTACT_PAGE_CAPACITY)
        page_key = self._by_id.get(identifier)
        if page_key is None:
            raise ValueError("parallel contact identity is missing")
        expected_role, bucket, _ = self.parse_page_key(page_key)
        page = self.pages[page_key]
        record = page._record(slot)
        if expected_role != role or record is None or (record[0] + bucket, record[1]) != (row, column):
            raise ValueError("parallel contact identity changed endpoints")
        if type(sign) is not int or sign >> 2 != 0x15 or sign & 3 == 3 or type(state) is not int or not 0 <= state <= 255:
            raise ValueError("parallel contact sign or state is invalid")
        page.write(slot, row - bucket, column, sign, state)
        self._revision += 1

    def _page_records(self, key, start, end):
        role, bucket, identifier = self.parse_page_key(key)
        page = self.pages[key]
        with page._packed_residency_scope(torch.device("cpu")):
            count = page.ordered_slots()
        for index in range(count):
            with page._packed_residency_scope(torch.device("cpu")):
                slot = int(page._sort_order[index, 0]) | int(page._sort_order[index, 1]) << 8
                record = page._record(slot)
            if record is None:
                continue
            local_row, column, sign, state = record
            row = bucket + local_row
            if start <= row < end:
                ordinal = identifier * CONTACT_PAGE_CAPACITY + slot
                yield ("%s_r%d_c%d_n%d" % (role, row, column, ordinal), row, column, sign, state)

    def iter_records(self, *, bias=False, start=0, end=None):
        rows, _ = self._geometry(bias)
        end = rows if end is None else end
        role = "b" if bias else "w"
        keys = [key for (part, bucket), values in self._by_bucket.items()
                if part == role and bucket < end and bucket + CONTACT_ROW_BUCKET > start for key in values]
        if not keys:
            return
        # The merge retains one current record per page, not all contacts.
        policy = getattr(self._native_core_pager, "shared_resource_policy", None)
        reserve = getattr(policy, "reserve_ram", None)
        lease = reserve(len(keys) * 512 + 4096, "ordered contact page cursors") if callable(reserve) else nullcontext()
        with lease:
            yield from heapq.merge(*(self._page_records(key, start, end) for key in keys),
                key=lambda item: (item[1], item[2], int(item[0].rsplit("_n", 1)[1])))

    def scratch_bytes(self, *, bias=False, start=0, end=None):
        return 64 + sum(512 + len(record[0]) * 4 for record in self.iter_records(bias=bias, start=start, end=end))

    def replace_records(self, records, *, bias=False, start=0, end=None, prune=True):
        rows, _columns = self._geometry(bias)
        end = rows if end is None else end
        records = tuple(records)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= rows:
            raise ValueError("parallel connection restore range is invalid")
        keep = set()
        identities = set()
        for record in records:
            if not isinstance(record, (tuple, list)) or len(record) != 5:
                raise ValueError("parallel connection record is invalid")
            key, row, column, sign, state = record
            role, expected_row, expected_column, ordinal = self.parse_key(key)
            self._validate_endpoint(row, column, bias)
            if (key in keep or ordinal in identities or (role == "b") != bias or (row, column) != (expected_row, expected_column)
                    or not start <= row < end or ordinal >= 1 << 64 or type(sign) is not int
                    or sign >> 2 != 0x15 or sign & 3 == 3 or type(state) is not int or not 0 <= state <= 255):
                raise ValueError("parallel connection record endpoint, range, identity or code is invalid")
            identifier, slot = divmod(ordinal, CONTACT_PAGE_CAPACITY)
            existing_key = self._by_id.get(identifier)
            if existing_key is not None:
                existing_role, bucket, _ = self.parse_page_key(existing_key)
                old = self.pages[existing_key]._record(slot)
                if existing_role != role or row // CONTACT_ROW_BUCKET * CONTACT_ROW_BUCKET != bucket or old is not None and (old[0] + bucket, old[1]) != (row, column):
                    raise ValueError("parallel contact identity changed endpoints")
            keep.add(key)
            identities.add(ordinal)
        for key, row, column, _sign, _state in self.iter_records(bias=bias, start=start, end=end) if prune else ():
            if key in keep:
                continue
            ordinal = self.parse_key(key)[3]
            identifier, slot = divmod(ordinal, CONTACT_PAGE_CAPACITY)
            page_key = self._by_id[identifier]
            bucket = self.parse_page_key(page_key)[1]
            self.pages[page_key].write(slot, row - bucket, column, 0x55, 0, remove=True)
        for key, row, column, sign, state in records:
            role, _, _, ordinal = self.parse_key(key)
            identifier, slot = divmod(ordinal, CONTACT_PAGE_CAPACITY)
            bucket = row // CONTACT_ROW_BUCKET * CONTACT_ROW_BUCKET
            page_key = "%s_r%d_p%d" % (role, bucket, identifier)
            page = self._register_page(page_key)
            existing = page._record(slot)
            if existing is not None and (existing[0] + bucket, existing[1]) != (row, column):
                raise ValueError("parallel contact identity changed endpoints")
            used = page.header()[0]
            # A filtered collective range may omit earlier, removed slots.
            # Preserve those identities as tombstones instead of reusing them.
            for missing in range(used, slot):
                page.write(missing, 0, 0, 0x55, 0, remove=True)
            page.write(slot, row - bucket, column, sign, state)
            if self._counter() <= identifier:
                self._set_counter(identifier + 1)
        self._revision += 1

    def prepare_bounded_state_load(self, specs, prefix):
        fields = {}
        for name, spec in specs.items():
            if not name.startswith(prefix + "pages."):
                continue
            suffix = name[len(prefix + "pages."):]
            key, separator, field = suffix.partition(".")
            self.parse_page_key(key)
            expected = {"_packed_sign": (1, CONTACT_PAGE_CAPACITY // 4), "_connection_state": (1, CONTACT_PAGE_CAPACITY), "_record_meta": (CONTACT_PAGE_CAPACITY, 10)}
            if not separator or field not in expected or spec.dtype != torch.uint8 or tuple(spec.shape) != expected[field]:
                raise ValueError("contact page checkpoint field, shape or dtype is invalid")
            fields.setdefault(key, set()).add(field)
        if any(value != {"_packed_sign", "_connection_state", "_record_meta"} for value in fields.values()):
            raise ValueError("contact page checkpoint is incomplete")
        for key in sorted(fields):
            self._register_page(key)

    def bounded_optional_state_defaults(self, specs, prefix):
        if prefix + "_next_page" in specs:
            return {}
        if any(name.startswith(prefix + "pages.") for name in specs):
            raise ValueError("contact pages require their saved identity counter")
        return {"_next_page": torch.zeros(8, dtype=torch.uint8)}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.prepare_bounded_state_load(state_dict, prefix)
        for name, value in self.bounded_optional_state_defaults(state_dict, prefix).items():
            state_dict[prefix + name] = value
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _after_load(self, module, incompatible):
        self.validate_bounded_state_load()
        self._revision += 1

    def validate_bounded_state_load(self):
        if self._next_page.dtype != torch.uint8 or tuple(self._next_page.shape) != (8,):
            raise ValueError("contact identity counter is invalid")
        counter = self._counter()
        for key, page in self.pages.items():
            role, bucket, identifier = self.parse_page_key(key)
            if identifier >= counter:
                raise ValueError("contact page is newer than its identity counter")
            page.validate_bounded_state_load()
            for slot in range(page.header()[0]):
                record = page._record(slot)
                if record is not None:
                    self._validate_endpoint(bucket + record[0], record[1], role == "b")

    def storage_status(self):
        live = len(self)
        page_count = len(self.pages)
        learned = page_count * (CONTACT_PAGE_CAPACITY // 4 + CONTACT_PAGE_CAPACITY)
        structural = 8 + page_count * CONTACT_PAGE_CAPACITY * 10
        return {"parallelConnectionCount": live, "parallelConnectionLearnedBytes": learned,
            "parallelConnectionPackedSignBytes": page_count * CONTACT_PAGE_CAPACITY // 4,
            "parallelConnectionActiveLearnedBits": live * 10,
            "parallelConnectionStructuralBytes": structural,
            "parallelConnectionAllocatedTensorBytes": learned + structural,
            "parallelConnectionIndexCacheBytes": page_count * CONTACT_PAGE_CAPACITY * 2,
            "parallelConnectionPageCount": page_count,
            "parallelConnectionPageCapacity": CONTACT_PAGE_CAPACITY,
            "parallelConnectionRegistryAdmissionBytes": page_count * CONTACT_PAGE_METADATA_ALLOWANCE,
            "parallelConnectionRegistryStorage": "packed-record-pages",
            "parallelConnectionBackingAllocationIncluded": False}

    @torch.no_grad()
    def copy_from(self, source):
        """Stream exact page identities/state into a fresh nonshrinking owner."""
        if self.pages or self._counter() or any(a < b for a, b in zip(self.weight_shape, source.weight_shape)) or self.bias_columns < source.bias_columns:
            raise ValueError("contact transfer requires a fresh nonshrinking destination")
        try:
            for key, original in source.pages.items():
                target = self._register_page(key)
                with original._packed_residency_scope(torch.device("cpu")), target._packed_residency_scope(torch.device("cpu")):
                    for name in ("_packed_sign", "_connection_state", "_record_meta"):
                        before, after = target._buffers[name], original._buffers[name]
                        with packed_diagnostic_write(target, before, after):
                            before.copy_(after)
            self._set_counter(source._counter())
            self.validate_bounded_state_load()
            self._revision += 1
        except BaseException:
            for key in tuple(self.pages):
                self._retire_page(key)
            self._set_counter(0)
            raise
