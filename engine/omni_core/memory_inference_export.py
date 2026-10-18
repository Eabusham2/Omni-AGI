"""Generation-bound native vector inference companions, without a dense table."""
from __future__ import annotations

import hashlib
import struct
import time

import torch

from .streamed_ternary import TernaryPackedSource, PAGE_PACKED_BYTES
from .ternary_packing import (ConnectionStateSource, ParallelConnectionSource,
                              TernaryTensorSpec, decode_ternary_2bit)
from .weighted_vsa_row import CONTACT, make_payload, payload_parts


class _RowScratchAdmission:
    """Reuse an admitted *peak*, not an allocation or permission bypass.

    Rows are consumed one at a time. Identical/smaller scratch cannot increase
    their peak, so expensive OS samples need not run for every scalar row.
    Larger scratch is always checked immediately. Time/count boundaries and
    each completed traversal force fresh pressure checks; cancellation is
    checked on every call, including calls reusing an approval.
    """
    def __init__(self, reserve, cancelled=None, *, clock=None):
        self.reserve, self.cancelled = reserve, cancelled
        self.clock = clock or time.monotonic
        self.approved, self.next_probe, self.requests = 0, 0., 0

    def check(self, amount, operation, *, force=False):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError('native vector inference export cancelled')
        amount = max(1, int(amount))
        self.requests += 1
        now = self.clock()
        if self.reserve is not None and (force or amount > self.approved
                or now >= self.next_probe or self.requests >= 256):
            if self.reserve(amount, operation) is False:
                raise RuntimeError('native vector export awaits row scratch memory')
            self.approved, self.next_probe, self.requests = amount, now + .25, 0


def vector_tensor_spec(vectors, *, name="substrate.neuron_vectors.weight", reserve=None, cancelled=None):
    snapshot = vectors.current_snapshot()
    count, width = len(vectors), int(vectors.dimensions)
    scratch = _RowScratchAdmission(reserve, cancelled)

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('native vector inference export cancelled')
        if vectors.current_snapshot() != snapshot:
            raise ValueError("neural vector generation changed during inference export")

    def rows():
        check()
        seen = 0
        page_rows = getattr(vectors, 'page_rows', None)
        def payloads():
            if not callable(page_rows):
                for identifier in vectors:
                    yield identifier, vectors.row_payload(identifier)
                return
            # The existing native page provider reserves the complete page
            # BEFORE fetching its BLOBs. Its sequential order is identical to
            # native __iter__, and a refused page retries the same cursor at
            # smaller size; not one record is silently dropped or truncated.
            from .paged_packed_vectors import PagedVectorResourcePause
            cursor, page_size = None, 256
            while True:
                check()
                try:
                    page = page_rows(snapshot, cursor, page_size)
                except PagedVectorResourcePause:
                    if page_size == 1:
                        raise
                    page_size = max(1, page_size // 2)
                    continue
                if page.snapshot_id != snapshot:
                    raise ValueError('neural vector export page has another generation')
                for row in page.rows:
                    # Normalize legacy unit rows exactly as row_payload does;
                    # signs, strengths, resistance and contacts stay intact.
                    yield row.assembly_id, make_payload(*payload_parts(row.packed, width), width)
                if not page.has_more:
                    break
                if not page.rows or page.next_cursor is None or page.next_cursor == cursor:
                    raise ValueError('neural vector export page did not advance its exact cursor')
                cursor = page.next_cursor
        from .paged_read_session import ReadSession, keeper_eligible_stores
        scope = ReadSession(keeper_eligible_stores([vectors]), cancelled=cancelled)
        iterator = iter(payloads())
        try:
            while True:
                with scope.activate():
                    try:
                        identifier, payload = next(iterator)
                    except StopIteration:
                        break
                    check()
                    scratch.check(len(payload) * 8 + width * 8 + 4096, 'native vector inference row')
                # No READ ContextVar escapes across this yield. A consumer
                # mutation still follows its real writer path and is caught
                # by the next original generation check, never blocked by an
                # accidentally leaked read-only connection lease.
                yield identifier, payload
                seen += 1
        finally:
            scope.__exit__(None, None, None)
        if seen != count:
            raise ValueError("neural vector export row coverage changed")
        if seen:
            scratch.check(scratch.approved or width * 8 + 4096,
                'native vector inference traversal pressure', force=True)
        check()

    identity, state, repeated = hashlib.sha256(), hashlib.sha256(), 0
    for identifier, payload in rows():
        key = identifier.encode("utf-8")
        identity.update(struct.pack("<Q", len(key)))
        identity.update(key)
        state.update(struct.pack("<Q", len(payload)))
        state.update(payload)
        repeated += len(payload_parts(payload, width)[2]) // CONTACT.size

    def sign_pages():
        for _identifier, payload in rows():
            signs = payload_parts(payload, width)[0]
            for start in range(0, len(signs), PAGE_PACKED_BYTES):
                block = signs[start:start + PAGE_PACKED_BYTES]
                yield decode_ternary_2bit(block, (min(len(block) * 4, width - start * 4),))

    def state_pages():
        for _identifier, payload in rows():
            states = payload_parts(payload, width)[1]
            for start in range(0, len(states), 1 << 20):
                yield torch.frombuffer(bytearray(states[start:start + (1 << 20)]), dtype=torch.uint8)

    def records():
        ordinal_base = 0
        for row, (_identifier, payload) in enumerate(rows()):
            packed = payload_parts(payload, width)[2]
            n = len(packed) // CONTACT.size
            scratch.check(n * 256 + 4096, 'native vector repeated-contact ordering')
            ordered = sorted(enumerate(CONTACT.iter_unpack(packed)), key=lambda item: (item[1][0], item[0]))
            for index, (column, sign, strength) in ordered:
                if cancelled is not None and cancelled():
                    raise InterruptedError('native vector inference export cancelled')
                yield (f"w_r{row}_c{column}_n{ordinal_base + index}", row, column, sign, strength)
            ordinal_base += n
            del ordered
        if ordinal_base != repeated:
            raise ValueError("neural vector repeated-contact count changed")

    spec = TernaryTensorSpec(name=name, values=TernaryPackedSource((count, width), sign_pages),
        kind="dynamic-synapse", source_dtype="packed-2bit",
        connection_state=ConnectionStateSource((count, width), state_pages),
        parallel_connections=ParallelConnectionSource(repeated, records))
    return spec, {"count": count, "dimensions": width, "orderSha256": identity.hexdigest(),
                  "stateSha256": state.hexdigest(), "repeatedContacts": repeated,
                  "orderBasis": "vector-store-native-row-order-v1"}
