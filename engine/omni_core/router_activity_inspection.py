"""Exact loaded-router contact snapshots for the LIF activity observer.

This reads only allocated recurrent contacts whose source and target channels
are both inside the admitted LIF viewport. It never maps channels to semantic
neurons, classifies a stored contact as firing, or changes recurrence/STDP.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
import weakref
from contextlib import nullcontext

from .packed_connections import connection_resistance, connection_strengths
from .router_state_paging import unpack_tile
from .sparse_router_state import (
    BLOCK_COLUMNS, BLOCK_ROWS, block_key, block_layer, parse_block_key,
    sparse_layout_digest,
)


CONTACT_SOURCE = "actual-allocated-router-contacts"
CONTACT_OWNER = "root.router.synapses"
_RECORD_RAM_ESTIMATE = 512
_PAGE_ALLOWANCE = 256 * 1024
_METADATA_ALLOWANCE = 16 * 1024
_TRANSMISSION_RING_BYTES = 1024 * 1024
_TRANSMISSION_RECORD_ESTIMATE = 768
_TRANSMISSION_CAPACITY = _TRANSMISSION_RING_BYTES // _TRANSMISSION_RECORD_ESTIMATE
_SAFE_INTEGER = (1 << 53) - 1


def empty_router_contact_page(brain_id, *, reason, offset=0, page_size=128):
    return {"source": CONTACT_SOURCE, "owner": CONTACT_OWNER, "available": False,
            "observed": False, "capturedAtUnix": None, "topologyGeneration": None,
            "totalContacts": None, "offset": offset, "returned": 0,
            "pageSize": page_size, "hasMore": False, "contacts": [],
            "reason": reason, "synapseFiringObserved": False}


def empty_router_transmission_page(*, reason, capture_epoch=None,
                                   contact_offset=0, contact_count=0,
                                   event_page_size=128):
    return {"source": "actual-packed-contact-transmissions",
            "owner": CONTACT_OWNER, "available": False, "enabled": False,
            "observed": False, "captureEpoch": capture_epoch, "epoch": None,
            "layoutDigest": None, "selectedContactOffset": contact_offset,
            "selectedContactCount": contact_count,
            "ringCapacity": 0, "lastSequence": 0,
            "firstAvailableSequence": 0, "nextSequence": 0,
            "droppedRecords": 0, "missedRecords": 0,
            "pageSize": event_page_size, "hasMore": False, "events": [],
            "reason": reason, "postsynapticFiringObserved": False}


def _weak_reference(value):
    try:
        return weakref.ref(value)
    except TypeError:  # constructor-free fixtures can use non-weakref objects
        return lambda: value


class RouterContactSnapshot:
    def __init__(self, brain_id, start, count, records=(), *, captured_at=None,
                 generation=None, lease=None, available=True, reason="",
                 synapses=None, layout_digest=None):
        self.brain_id, self.start, self.count = brain_id, start, count
        self.records = tuple(records)
        self.captured_at = captured_at
        self.generation = generation
        self.lease = lease
        self.available = available
        self.reason = reason
        self.closed = False
        self.layout_digest = layout_digest
        self._synapses = _weak_reference(synapses) if synapses is not None else lambda: None
        self._transmission_lock = threading.RLock()
        self._transmission_ring = [None] * _TRANSMISSION_CAPACITY if available else []
        self._transmission_enabled = bool(available)
        self._transmission_reason = ""
        self._capture_epoch = None
        self._selection = None
        self._selected_by_block = {}
        self._transmission_epoch = None
        self._transmission_sequence = 0
        self._operation_sequence = 0

    @classmethod
    def unavailable(cls, brain_id, start, count, reason):
        return cls(brain_id, start, count, available=False, reason=reason)

    def close(self):
        with self._transmission_lock:
            if self.closed:
                return
            self.closed = True
            self._transmission_enabled = False
            self._transmission_reason = "router-contact-snapshot-retired"
            self._transmission_ring = []
            self._selected_by_block = {}
            lease, self.lease = self.lease, None
            self.records = ()
        if lease is not None:
            lease.release()

    def bind_capture(self, capture_epoch, synapses):
        if not isinstance(capture_epoch, str) or len(capture_epoch) != 32:
            raise ValueError("router transmission capture epoch is invalid")
        with self._transmission_lock:
            if not self.available:
                return
            if self.closed or self._synapses() is not synapses:
                raise ValueError("router transmission synapse owner changed")
            self._capture_epoch = capture_epoch

    def retire_transmissions(self, reason):
        with self._transmission_lock:
            if not self._transmission_enabled:
                return
            self._transmission_enabled = False
            self._transmission_reason = str(reason)

    def _select_locked(self, offset, page_size):
        selection = (offset, page_size)
        if selection == self._selection:
            return False
        selected = tuple(enumerate(self.records[offset:offset + page_size], start=offset))
        by_block = {}
        for index, record in selected:
            by_block.setdefault(record[0], []).append((index, record))
        self._selection = selection
        self._selected_by_block = {key: tuple(values) for key, values in by_block.items()}
        self._transmission_epoch = uuid.uuid4().hex
        self._transmission_ring = [None] * _TRANSMISSION_CAPACITY
        self._transmission_sequence = 0
        self._operation_sequence = 0
        return True

    def begin_recurrence(self, synapses):
        with self._transmission_lock:
            if self.closed or not self._transmission_enabled:
                return None
            if self._synapses() is not synapses:
                raise RuntimeError("router transmission synapse owner changed")
            if (self._capture_epoch is None or self._selection is None
                    or not self._selected_by_block):
                return None
            if self._operation_sequence >= _SAFE_INTEGER:
                raise RuntimeError("router transmission operation sequence exhausted")
            self._operation_sequence += 1
            return (self._operation_sequence, time.time(), time.monotonic_ns(),
                    frozenset(self._selected_by_block), set(),
                    self._transmission_epoch, self._selected_by_block)

    def record_packed_tile(self, operation, synapses, identity, region,
                           input_slice, signs, states):
        if operation is None:
            return
        with self._transmission_lock:
            if self.closed or not self._transmission_enabled:
                return
            if self._synapses() is not synapses:
                raise RuntimeError("router transmission synapse owner changed")
            if operation[5] != self._transmission_epoch:
                return
            if (not isinstance(identity, tuple) or len(identity) != 2
                    or not isinstance(identity[0], str)
                    or identity[1] != self.layout_digest):
                raise RuntimeError("router transmission layout changed")
            block_id = identity[0]
            expected_blocks, seen_blocks = operation[3], operation[4]
            if block_id in expected_blocks:
                seen_blocks.add(block_id)
            selected = operation[6].get(block_id, ())
            if not selected:
                return
            r0, r1, c0, c1 = region
            if (states is None or tuple(states.shape) != (r1-r0, c1-c0)
                    or tuple(signs.shape) != (r1-r0, (c1-c0+3)//4)
                    or input_slice.numel() != c1-c0):
                raise RuntimeError("router transmission tile geometry changed")
            for record_index, record in selected:
                target, source = record[2], record[3]
                if not (r0 <= target < r1 and c0 <= source < c1):
                    continue
                row, column = target-r0, source-c0
                code = (int(signs[row, column//4]) >> (2*(column % 4))) & 3
                if code == 3:
                    raise ValueError("router transmission encountered a reserved ternary code")
                sign = code - 1
                state = int(states[row, column])
                strength, resistance = (state & 15) + 1, state >> 4
                input_value = float(input_slice[column])
                contribution = input_value * sign * strength
                if not math.isfinite(input_value) or not math.isfinite(contribution):
                    raise ValueError("router transmission operands are not finite")
                if self._transmission_sequence >= _SAFE_INTEGER:
                    raise RuntimeError("router transmission record sequence exhausted")
                sequence = self._transmission_sequence + 1
                self._transmission_ring[(sequence-1) % _TRANSMISSION_CAPACITY] = (
                    sequence, operation[0], operation[1], operation[2],
                    record_index, input_value, sign, strength, resistance,
                    contribution)
                self._transmission_sequence = sequence

    def finish_recurrence(self, operation):
        if operation is None:
            return
        with self._transmission_lock:
            if self.closed or not self._transmission_enabled:
                return
            if operation[5] != self._transmission_epoch:
                return
            if operation[3] != frozenset(operation[4]):
                raise RuntimeError("router transmission selected topology changed")

    def transmission_page(self, contact_offset=0, contact_page_size=128, *,
                          epoch=None, after=0, event_page_size=128):
        if (type(after) is not int or after < 0 or type(event_page_size) is not int
                or not 1 <= event_page_size <= 128):
            raise ValueError("invalid router transmission page")
        with self._transmission_lock:
            if self.closed or not self.available or self._capture_epoch is None:
                return empty_router_transmission_page(
                    reason=self.reason or "router-transmission-owner-unavailable",
                    capture_epoch=self._capture_epoch, contact_offset=contact_offset,
                    event_page_size=event_page_size)
            if contact_offset > len(self.records):
                raise ValueError("router transmission contact cursor exceeds the exact snapshot")
            changed = self._select_locked(contact_offset, contact_page_size)
            if not changed and epoch is not None and epoch != self._transmission_epoch:
                raise ValueError("router transmission selection epoch changed; discard the old cursor")
            requested_after = 0 if changed else after
            if requested_after > self._transmission_sequence:
                raise ValueError("router transmission cursor is ahead of this selection")
            first = max(1, self._transmission_sequence-_TRANSMISSION_CAPACITY+1) \
                if self._transmission_sequence else 0
            start = max(requested_after+1, first)
            events = []
            for sequence in range(start, min(self._transmission_sequence+1,
                                              start+event_page_size)):
                stored = self._transmission_ring[(sequence-1) % _TRANSMISSION_CAPACITY]
                if stored is None or stored[0] != sequence:
                    raise RuntimeError("router transmission ring identity diverged")
                _, operation_sequence, observed, monotonic, record_index, input_value, sign, strength, resistance, contribution = stored
                block_id, _, target, source, *_ = self.records[record_index]
                contact_id = "%s:root.router.synapses:%s:%d:%d" % (
                    self.brain_id, block_id, target, source)
                classification = "inhibitory" if contribution < 0 else \
                    "excitatory" if contribution > 0 else "zero"
                events.append({"sequence": sequence,
                    "operationSequence": operation_sequence,
                    "observedAtUnix": observed, "monotonicNs": str(monotonic),
                    "contactId": contact_id, "blockId": block_id,
                    "sourceChannel": source, "targetChannel": target,
                    "sourceId": "%s:root.router.population:%d" % (self.brain_id, source),
                    "targetId": "%s:root.router.population:%d" % (self.brain_id, target),
                    "sourceInput": input_value, "effectiveWeight": sign,
                    "connectionStrength": strength,
                    "connectionResistance": resistance,
                    "effectiveContribution": contribution,
                    "classification": classification,
                    "transmissionObserved": True,
                    "postsynapticFiringObserved": False})
            next_sequence = events[-1]["sequence"] if events else requested_after
            selected_count = min(contact_page_size, len(self.records)-contact_offset)
            return {"source": "actual-packed-contact-transmissions",
                    "owner": CONTACT_OWNER, "available": True,
                    "enabled": self._transmission_enabled,
                    "observed": self._transmission_sequence > 0,
                    "captureEpoch": self._capture_epoch,
                    "epoch": self._transmission_epoch,
                    "layoutDigest": self.layout_digest,
                    "selectedContactOffset": contact_offset,
                    "selectedContactCount": selected_count,
                    "ringCapacity": _TRANSMISSION_CAPACITY,
                    "lastSequence": self._transmission_sequence,
                    "firstAvailableSequence": first,
                    "nextSequence": next_sequence,
                    "droppedRecords": max(0, self._transmission_sequence-_TRANSMISSION_CAPACITY),
                    "missedRecords": max(0, first-requested_after-1),
                    "pageSize": event_page_size,
                    "hasMore": next_sequence < self._transmission_sequence,
                    "events": events, "reason": self._transmission_reason,
                    "postsynapticFiringObserved": False}

    def page(self, offset=0, page_size=128):
        if type(offset) is not int or offset < 0 or type(page_size) is not int or not 1 <= page_size <= 256:
            raise ValueError("invalid router contact page")
        if self.closed:
            return empty_router_contact_page(
                self.brain_id, reason="router-contact-snapshot-retired",
                offset=offset, page_size=page_size)
        if not self.available:
            return empty_router_contact_page(self.brain_id, reason=self.reason,
                                             offset=offset, page_size=page_size)
        if offset > len(self.records):
            raise ValueError("router contact cursor exceeds the exact snapshot")
        selected = self.records[offset:offset + page_size]
        contacts = []
        for block_id, parallel_layer, target, source, sign, strength, resistance, eligibility, stability, uses in selected:
            source_id = "%s:root.router.population:%d" % (self.brain_id, source)
            target_id = "%s:root.router.population:%d" % (self.brain_id, target)
            identifier = "%s:root.router.synapses:%s:%d:%d" % (
                self.brain_id, block_id, target, source)
            base = block_key(*parse_block_key(block_id))
            root_id = "%s:root.router.synapses:%s:%d:%d" % (
                self.brain_id, base, target, source)
            contacts.append({"id": identifier, "owner": CONTACT_OWNER,
                             "sourceChannel": source, "targetChannel": target,
                             "sourceId": source_id, "targetId": target_id,
                             "blockId": block_id, "parallelLayer": parallel_layer,
                             "effectiveWeight": sign, "connectionStrength": strength,
                             "connectionResistance": resistance,
                             "effectiveContribution": sign * strength,
                             "eligibility": eligibility, "stability": stability,
                             "uses": uses, "isRepeatedContact": parallel_layer > 0,
                             "parallelOrdinal": parallel_layer,
                             "parallelRootId": root_id, "firingObserved": False})
        end = offset + len(contacts)
        return {"source": CONTACT_SOURCE, "owner": CONTACT_OWNER,
                "available": not self.closed, "observed": not self.closed,
                "capturedAtUnix": self.captured_at,
                "topologyGeneration": self.generation,
                "totalContacts": len(self.records) if not self.closed else 0,
                "offset": offset, "returned": len(contacts), "pageSize": page_size,
                "hasMore": not self.closed and end < len(self.records),
                "contacts": contacts, "reason": self.reason,
                "synapseFiringObserved": False}


def capture_router_contacts(brain, brain_id, start, count):
    """Capture one immutable, resource-admitted selected-channel contact set."""
    router = getattr(brain, "router", None)
    synapses = getattr(router, "synapses", None)
    sparse = getattr(synapses, "_sparse_state", None)
    if synapses is None or sparse is None or not getattr(sparse, "ready", False):
        return RouterContactSnapshot.unavailable(
            brain_id, start, count, "no-exact-loaded-sparse-router-contact-owner")
    if (getattr(synapses, "pre_neurons", None) != getattr(synapses, "post_neurons", None)
            or getattr(synapses, "pre_neurons", None) != getattr(router, "neurons", None)):
        return RouterContactSnapshot.unavailable(
            brain_id, start, count, "router-contact-population-geometry-unavailable")
    end = start + count
    reserve = getattr(getattr(brain, "resource_policy", None), "reserve_ram", None)
    if not callable(reserve):
        raise ValueError("router contact snapshot requires shared RAM admission")
    operation = synapses._operation() if callable(getattr(synapses, "_operation", None)) else nullcontext()
    lease = None
    try:
        with operation:
            keys = sorted(sparse.owner.blocks)
            specs = []
            potential = 0
            for key in keys:
                block = sparse.owner.blocks[key]
                block_row, block_column = parse_block_key(key)
                row_start, column_start = block_row * BLOCK_ROWS, block_column * BLOCK_COLUMNS
                local_r0, local_r1 = max(0, start - row_start), min(block.rows, end - row_start)
                local_c0, local_c1 = max(0, start - column_start), min(block.columns, end - column_start)
                if local_r0 >= local_r1 or local_c0 >= local_c1:
                    continue
                if getattr(block, "_router_unpublished", False):
                    raise RuntimeError("router contact snapshot encountered unpublished topology")
                cells = (local_r1 - local_r0) * (local_c1 - local_c0)
                if potential > (_SAFE_INTEGER - cells):
                    raise ValueError("router contact snapshot exceeds exact accounting")
                potential += cells
                # Packed signs can only be decoded from a byte-aligned first
                # column. Retain the exact requested columns separately so an
                # unaligned viewport never widens the exposed contact set.
                decode_c0 = local_c0 // 4 * 4
                specs.append((key, block, row_start, column_start,
                              (local_r0, local_r1, local_c0, local_c1),
                              (local_r0, local_r1, decode_c0, local_c1)))
            required = (potential * _RECORD_RAM_ESTIMATE + _PAGE_ALLOWANCE
                        + _METADATA_ALLOWANCE + _TRANSMISSION_RING_BYTES
                        + len(specs) * 128)
            if required > _SAFE_INTEGER:
                raise ValueError("router contact snapshot exceeds exact accounting")
            lease = reserve(required, "exact selected router contact inspection snapshot")
            if lease is None or not callable(getattr(lease, "release", None)):
                raise ValueError("router contact snapshot requires a retained RAM lease")
            generation = int(getattr(sparse, "_generation", 0))
            layout_digest = sparse_layout_digest(sparse.owner, keys)
            records = []
            for key, block, row_start, column_start, selected_region, decode_region in specs:
                if callable(getattr(synapses, "_check", None)):
                    synapses._check()
                r0, r1, c0, c1 = selected_region
                _, _, decode_c0, decode_c1 = decode_region
                ram = synapses._ram((r1-r0)*(decode_c1-decode_c0)*64 + 4096,
                                    "selected router contact inspection tile") \
                    if callable(getattr(synapses, "_ram", None)) else nullcontext()
                try:
                    with ram:
                        decoded = unpack_tile(block._packed_weights, *decode_region).detach().to(device="cpu")
                        levels = decoded[:, c0-decode_c0:c1-decode_c0]
                        state = block._connection_state[r0:r1, c0:c1].detach().to(device="cpu")
                        eligibility = block.eligibility_accumulator[r0:r1, c0:c1].detach().to(device="cpu")
                        stability = block.stability[r0:r1, c0:c1].detach().to(device="cpu")
                        uses = block.uses[r0:r1, c0:c1].detach().to(device="cpu")
                        present = (levels.ne(0) | state.ne(0) | eligibility.ne(0)
                                   | stability.ne(0) | uses.ne(0))
                        positions = present.nonzero(as_tuple=False).tolist()
                        strengths = connection_strengths(state)
                        resistances = connection_resistance(state)
                        layer = block_layer(key)
                        for row, column in positions:
                            sign = int(levels[row, column])
                            strength = int(strengths[row, column])
                            resistance = int(resistances[row, column])
                            stable, used = float(stability[row, column]), float(uses[row, column])
                            if sign not in (-1, 0, 1) or not 1 <= strength <= 16 or not 0 <= resistance <= 15 \
                                    or not math.isfinite(stable) or not math.isfinite(used) or used < 0:
                                raise ValueError("router contact inspection record is invalid")
                            records.append((key, layer, row_start+r0+row,
                                            column_start+c0+column, sign, strength,
                                            resistance, int(eligibility[row, column]), stable, used))
                finally:
                    sparse.release(block, decode_region)
            if int(getattr(sparse, "_generation", 0)) != generation:
                raise RuntimeError("router contact topology changed during snapshot")
            return RouterContactSnapshot(brain_id, start, count, records,
                                         captured_at=time.time(), generation=generation,
                                         lease=lease, synapses=synapses,
                                         layout_digest=layout_digest)
    except BaseException:
        if lease is not None:
            lease.release()
        raise
