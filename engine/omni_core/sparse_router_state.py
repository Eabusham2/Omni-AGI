"""Allocated-only exact ternary recurrence, controls and native migration.

An absent block is exactly zero, including its timing/stability/usage state.
Coactivity may legitimately allocate every block: admission, not a neuron or
block-count cap, is the boundary. Mutable blocks use the shared RAM-first router
pager; this module never adopts a checkpoint as writable backing.
"""
from __future__ import annotations

import math
import re
import hashlib
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from .bounded_tensor_io import TRANSFER_BYTES
from .router_state_paging import (
    WEIGHTED_MATRIX_FIELDS as MATRIX_FIELDS, MATRIX_FIELDS as LEGACY_MATRIX_FIELDS,
    RouterMutationJournal, RouterStatePager, ROUTER_CELL_SCRATCH_BYTES,
    pack_tile, release_router_tensor_chunk, tile_ranges, unpack_tile,
    update_historical_tensor_checksum,
)
from .parameter_diagnostics import (
    packed_diagnostic_write, register_sparse_zero_owner, retire_sparse_zero_owner,
)
from .packed_connections import connection_strengths, connection_resistance, encode_connection_state, weighted_ternary_levels


SPARSE_LAYOUT_VERSION = 2
BLOCK_ROWS = BLOCK_COLUMNS = 64
BLOCK_METADATA_BYTES = 8192
_KEY = re.compile(r"r(0|[1-9][0-9]*)_c(0|[1-9][0-9]*)(?:_p([1-9][0-9]*))?\Z")
_DTYPES = {"_packed_weights": torch.uint8, "eligibility_accumulator": torch.int16,
           "stability": torch.float32, "uses": torch.float32, "_connection_state": torch.uint8}


def block_key(row: int, column: int, layer: int = 0) -> str:
    if type(layer) is not int or layer < 0:
        raise ValueError("parallel router layer must be a nonnegative integer")
    return "r%d_c%d" % (row, column) + ("_p%d" % layer if layer else "")


def parse_block_key(name: str) -> tuple[int, int]:
    match = _KEY.fullmatch(name)
    if match is None:
        raise ValueError("sparse router block name is not canonical")
    return int(match[1]), int(match[2])


def block_layer(name: str) -> int:
    match = _KEY.fullmatch(name)
    if match is None:
        raise ValueError("sparse router block name is not canonical")
    return int(match[3] or 0)


def sparse_layout_digest(owner, keys) -> str:
    """Process-local inspection seal for one exact allocated block layout.

    Object identities deliberately bind the live owner session. This is not a
    checkpoint/provenance digest and must never be persisted as neural state.
    """
    digest = hashlib.sha256(b"omni-router-inspection-layout-v1\0")
    digest.update(str(id(owner)).encode("ascii") + b"\0")
    digest.update(str((int(owner.post_neurons), int(owner.pre_neurons))).encode("ascii") + b"\0")
    for key in keys:
        block = owner.blocks[key]
        if getattr(block, "_router_unpublished", False):
            raise RuntimeError("router inspection encountered unpublished topology")
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str((id(block), int(block.rows), int(block.columns),
                           int(getattr(block, "_router_birth", 0)))).encode("ascii") + b"\0")
    return digest.hexdigest()


def canonical_module_walk(root, *, resource_policy=None):
    """Named-module traversal preserving all old order except sparse registries.

    Live coactivity births and lexical checkpoint headers may insert the same
    block identities in different orders. Only the explicitly marked registry
    is canonicalized, with its actual-key sort admitted before allocation.
    No implicit potential pairs or full-model sorted inventory is built.
    """
    if not hasattr(root, "_modules"):
        yield from root.named_modules()  # Existing non-neural protocol fixtures.
        return
    seen = set()

    def walk(module, path):
        if id(module) in seen:
            return
        seen.add(id(module))
        yield path, module
        children = module._modules
        if getattr(module, "_native_sparse_registry", False):
            pager = getattr(module, "_router_state_pager", None)
            native_ref = getattr(module, "_native_core_pager_ref", None)
            native = native_ref() if callable(native_ref) else None
            policy = resource_policy if resource_policy is not None else (
                getattr(pager, "policy", None) if pager is not None else getattr(native, "shared_resource_policy", None))
            reserve = getattr(policy, "reserve_ram", None)
            lease = reserve(1024 + len(children)*128, "canonical sparse module registry order") if callable(reserve) else nullcontext()
            with lease:
                keys = sorted(children)
                for key in keys:
                    if pager is not None:
                        pager.check()
                    if native is not None and getattr(native, "_closed", False):
                        raise RuntimeError("canonical contact traversal used a closed native pager")
                    child = children[key]
                    if child is not None:
                        yield from walk(child, path + ("." if path else "") + key)
        else:
            for key, child in children.items():
                if child is not None:
                    yield from walk(child, path + ("." if path else "") + key)

    yield from walk(root, "")


class SparseRouterBlock(nn.Module):
    """A registered packed owner, not a floating or dense matrix shadow."""
    def __init__(self, rows, columns, pager=None, *, loading=False):
        super().__init__()
        self.rows, self.columns = int(rows), int(columns)
        self._router_state_pager = pager
        self._router_sparse_block = True
        self._router_unpublished = True
        self._router_birth = 0
        self._router_loaded = not loading
        try:
            for name in MATRIX_FIELDS:
                shape = (rows, (columns + 3) // 4) if name == "_packed_weights" else (rows, columns)
                fill = 0x55 if name == "_packed_weights" else 0
                value = (pager.allocate(self, name, shape, _DTYPES[name], fill=fill, loading=loading)
                         if pager is not None else torch.full(shape, fill, dtype=_DTYPES[name]))
                self.register_buffer(name, value)
        except BaseException:
            if pager is not None:
                pager.discard_unpublished_owner(self)
            raise

    def _apply(self, fn, recurse=True):
        # The page store stays CPU-owned even when small timing vectors and
        # projections move to an accelerator. No complete K×K transfer.
        return self

    def authoritative_packed_tensors(self):
        # Rollback and diagnostics require each tensor's actual registered
        # owner, not a container returning child buffers it cannot resolve.
        packed = self._packed_weights
        if packed.dtype != torch.uint8 or tuple(packed.shape) != (self.rows, (self.columns+3)//4):
            raise ValueError("sparse packed owner shape/dtype is invalid")
        state = self._connection_state
        if state.dtype != torch.uint8 or tuple(state.shape) != (self.rows, self.columns):
            raise ValueError("sparse connection strength/plasticity shape or dtype is invalid")
        return packed, state


def _validate_tile(levels, eligibility, stability, uses, state=None):
    if not bool(((levels >= -1) & (levels <= 1)).all()):
        raise ValueError("sparse router contains an invalid exact ternary level")
    limit = 255 if state is None else 256 * (1 + connection_resistance(state).to(torch.int32)) - 1
    if (bool((eligibility.to(torch.int32).abs() > limit).any())
            or not bool(torch.isfinite(stability).all()) or not bool(torch.isfinite(uses).all())
            or bool(((stability < 0) | (stability > 20) | (uses < 0)).any())):
        raise ValueError("sparse router timing/stability/usage controls are invalid")


def _stdp_replacements(owner, previous, state, eligibility, stability, uses, timing):
    """Signed STDP pressure changes discrete strength before crossing polarity.

    Timing traces and the bounded integer eligibility are control state. The
    learned forward value is exactly ternary sign × compact integer strength.
    Pressure survives subthreshold events; unrelated quiet pairs do not erase it.
    """
    delta = (owner.learning_rate / (1.0 + stability)) * timing
    active = timing.ne(0)
    agreement = previous.sign().eq(delta.sign()) | previous.eq(0)
    stability.add_(torch.where(agreement, torch.full_like(stability, owner.metaplasticity_rate),
        torch.full_like(stability, -owner.metaplasticity_rate * .25)) * active).clamp_(0., 20.)
    uses.add_(active.to(uses.dtype))
    increments = (delta / max(owner.learning_rate, 1e-6) * 256.).round().clamp(-256, 256).to(torch.int32)
    resistance = connection_resistance(state).to(torch.int32)
    threshold = 256 * (1 + resistance)
    pressure = eligibility.to(torch.int32) + increments
    transition = torch.where(pressure >= threshold, torch.ones_like(pressure),
        torch.where(pressure <= -threshold, -torch.ones_like(pressure), torch.zeros_like(pressure)))
    amplitude = previous.to(torch.int32) * connection_strengths(state).to(torch.int32)
    proposed = amplitude + transition
    overflow = torch.where(proposed.abs() > 16, transition, 0).to(torch.int8)
    updated = proposed.clamp(-16, 16)
    difference = updated - amplitude
    pressure = (pressure - transition * threshold).to(torch.int16)
    # Resistance rises with repeated reinforcing updates and falls as opposing
    # evidence weakens a connection. It remains an independently stored integer.
    strengthening = difference.ne(0) & amplitude.ne(0) & amplitude.sign().eq(transition.sign())
    weakening = difference.ne(0) & amplitude.ne(0) & ~amplitude.sign().eq(transition.sign())
    resistance = (resistance + strengthening.to(torch.int32) - weakening.to(torch.int32)).clamp(0, 15)
    # A lower resistance changes the next threshold; retain any excess pressure
    # inside its new representable control interval without fabricating a step.
    bound = 256 * (1 + resistance) - 1
    pressure = torch.maximum(-bound, torch.minimum(bound, pressure.to(torch.int32))).to(torch.int16)
    next_state = torch.where(active, encode_connection_state(updated.abs().clamp_min(1), resistance), state)
    replacements = {"stability": stability, "uses": uses, "eligibility_accumulator": pressure}
    next_sign = updated.sign().to(torch.int8)
    if bool(next_sign.ne(previous).any()):
        replacements["_packed_weights"] = pack_tile(next_sign)
    if not torch.equal(next_state, state):
        replacements["_connection_state"] = next_state
    return delta, active, difference, replacements, overflow


class SparseRouterState:
    def __init__(self, owner, *, loading=False):
        self.owner = owner
        self.ready = not loading
        self._generation = 0
        self._transaction_active = False
        self.migration_proof = None
        self._layers = {}
        owner.packed_synapse_container = True
        registry = nn.ModuleDict()
        registry._native_sparse_registry = True
        registry._router_state_pager = getattr(owner, "_router_state_pager", None)
        owner.add_module("blocks", registry)
        owner.register_buffer("_sparse_layout", torch.tensor([
            SPARSE_LAYOUT_VERSION, BLOCK_ROWS, BLOCK_COLUMNS,
            owner.post_neurons, owner.pre_neurons], dtype=torch.int64))

    @property
    def pager(self):
        return getattr(self.owner, "_router_state_pager", None)

    @property
    def allocated_trits(self):
        return sum(block.rows * block.columns for block in self.owner.blocks.values())

    def status(self):
        return {"mode": "allocated-block-sparse-native-router", "blockRows": BLOCK_ROWS,
                "blockColumns": BLOCK_COLUMNS, "allocatedBlocks": len(self.owner.blocks),
                "allocatedTernarySynapses": self.allocated_trits,
                "potentialConnectivity": self.owner.post_neurons * self.owner.pre_neurons,
                "allocatedStateBytes": sum(value.numel() * value.element_size()
                    for block in self.owner.blocks.values() for value in block.buffers()),
                "metadataBytesEstimate": len(self.owner.blocks) * BLOCK_METADATA_BYTES,
                "parallelLayers": sum(bool(block_layer(key)) for key in self.owner.blocks),
                "connectionStateBytes": sum(block._connection_state.numel() for block in self.owner.blocks.values()),
                "connectionStrengthEncoding": "uint8-low4-strength-minus-one-high4-resistance",
                "connectionStrengthOverflow": "allocate-independent-parallel-layer-contact",
                "parallelContactRegistryStorage": "page-backed-block-layers",
                "implicitConnectionsAreExactZero": True, "checkpointReady": self.ready}

    def shape(self, row, column):
        rows = min(BLOCK_ROWS, self.owner.post_neurons - row * BLOCK_ROWS)
        columns = min(BLOCK_COLUMNS, self.owner.pre_neurons - column * BLOCK_COLUMNS)
        if row < 0 or column < 0 or rows <= 0 or columns <= 0:
            raise ValueError("sparse router block is outside the declared population")
        return rows, columns

    def ensure(self, row, column, *, loading=False, layer=0):
        key = block_key(row, column, layer)
        existing = self.owner.blocks[key] if key in self.owner.blocks else None
        if existing is not None:
            return existing
        from .operation_auto_registry import PRIVATE_OPERATION
        if PRIVATE_OPERATION.get(): raise RuntimeError("private native Auto cannot grow recurrent learned contacts")
        self.owner._check()
        rows, columns = self.shape(row, column)
        if not self._transaction_active:
            self._generation += 1
        # Registered tensor/module/index metadata itself is not free. Its
        # pending allocation shares the same atomic selected-RAM ledger.
        with self.owner._ram(BLOCK_METADATA_BYTES, "sparse router block metadata") as lease:
            block = SparseRouterBlock(rows, columns, self.pager, loading=loading)
            try:
                block._router_birth = self._generation
                RouterStatePager._mark(lease, BLOCK_METADATA_BYTES)
                self.owner.blocks[key] = block
                if not loading:
                    register_sparse_zero_owner(self.owner, key, block)
                self._layers.setdefault((row, column), {})[layer] = key
                block._router_unpublished = self._transaction_active or loading
            except BaseException:
                retire_sparse_zero_owner(self.owner, key, block)
                if self.pager is not None:
                    self.pager.discard_unpublished_owner(block)
                if key in self.owner.blocks:
                    del self.owner.blocks[key]
                raise
        return block

    def _discard_generation(self, generation):
        # No unbounded duplicate inventory during pressure rollback. Modules
        # from this transaction are unpublished and have no external reader.
        while True:
            key = next((key for key, block in self.owner.blocks.items()
                        if block._router_birth == generation and block._router_unpublished), None)
            if key is None:
                break
            block = self.owner.blocks[key]
            retire_sparse_zero_owner(self.owner, key, block)
            if self.pager is not None:
                self.pager.discard_unpublished_owner(block)
            del self.owner.blocks[key]
            self._forget_layer(key)

    def retire_rollback_birth(self, key, block):
        """Called only by the journal proving this owner was absent at start."""
        if key not in self.owner.blocks or self.owner.blocks[key] is not block:
            raise RuntimeError("sparse rollback birth no longer has its exact owner")
        retire_sparse_zero_owner(self.owner, key, block)
        if self.pager is not None:
            if self.pager._active:
                block._router_unpublished = True
                self.pager.discard_unpublished_owner(block)
            else:
                self.pager.release_owner(block)
        del self.owner.blocks[key]
        self._forget_layer(key)

    def _forget_layer(self, key):
        position = parse_block_key(key)
        layers = self._layers.get(position)
        if layers is not None:
            layers.pop(block_layer(key), None)
            if not layers:
                del self._layers[position]

    def restore_topology_boundary(self, generation):
        if self.owner._sparse_state is not self or not self.ready:
            raise RuntimeError("packed rollback lost its exact sparse container generation")
        if self.pager is not None and self.pager._active:
            raise RuntimeError("whole-batch sparse topology restore requires quiescent router operations")
        while True:
            key = next((key for key, block in self.owner.blocks.items()
                        if block._router_birth > generation), None)
            if key is None:
                break
            self.retire_rollback_birth(key, self.owner.blocks[key])

    @contextmanager
    def transaction(self):
        from .operation_auto_registry import PRIVATE_OPERATION
        if PRIVATE_OPERATION.get(): raise RuntimeError("private native Auto cannot begin recurrent learned mutation")
        if self._transaction_active:
            raise RuntimeError("sparse router mutation transactions may not nest")
        self._generation += 1
        generation = self._generation
        journal = RouterMutationJournal(self.owner, self.pager,
            ram_bytes=self.pager.journal_ram_bytes if self.pager is not None else TRANSFER_BYTES)
        self._transaction_active = True
        retained = False
        try:
            yield journal
        except BaseException:
            try:
                journal.rollback()
            except BaseException:
                # Retain every before-image and all allocated blocks if exact
                # rollback cannot be proved. Further activity must fail closed.
                journal.close()
                raise
            self._discard_generation(generation)
            raise
        else:
            for block in self.owner.blocks.values():
                if block._router_birth == generation:
                    block._router_unpublished = False
            from .connected_transaction import retain_router_journal
            retained = retain_router_journal(journal, self, generation)
        finally:
            if not retained:
                journal.close()
            self._transaction_active = False

    def regions(self, block):
        yield from tile_ranges(block.rows, block.columns, self.owner._tile_budget())

    def release(self, block, region):
        if self.pager is not None:
            self.pager.release_tile(block, *region)

    def capture_writes(self, journal, key, block, region, replacements):
        r0, r1, c0, c1 = region
        for name, replacement in replacements.items():
            a, b = (c0 // 4, (c1 + 3) // 4) if name == "_packed_weights" else (c0, c1)
            target = getattr(block, name)[r0:r1, a:b]
            if not torch.equal(target.detach().cpu(), replacement):
                if not block._router_unpublished:
                    journal.capture("blocks." + key + "." + name, r0, r1, a, b)
                if name in {"_packed_weights", "_connection_state"}:
                    for row in range(r0, r1):
                        piece = replacement[row-r0].reshape(-1)
                        target_buffer = getattr(block, name)
                        with packed_diagnostic_write(block, target_buffer, piece,
                                                     start=row*target_buffer.shape[1]+a):
                            target_buffer[row, a:b].copy_(piece)
                else:
                    target.copy_(replacement.to(target.device))

    def validate(self):
        for key, block in self.owner.blocks.items():
            row, column = parse_block_key(key)
            if (block.rows, block.columns) != self.shape(row, column):
                raise ValueError("sparse router block logical dimensions are invalid")
            for name in MATRIX_FIELDS:
                value = getattr(block, name)
                expected = (block.rows, (block.columns + 3) // 4) if name == "_packed_weights" else (block.rows, block.columns)
                if value.dtype != _DTYPES[name] or tuple(value.shape) != expected or not value.is_contiguous():
                    raise ValueError("sparse router block shape/dtype is invalid")
            for region in self.regions(block):
                self.owner._check()
                r0, r1, c0, c1 = region
                with self.owner._ram((r1-r0)*(c1-c0)*32 + 1024, "sparse router state validation"):
                    levels = unpack_tile(block._packed_weights, *region)
                    _validate_tile(levels, block.eligibility_accumulator[r0:r1, c0:c1],
                                   block.stability[r0:r1, c0:c1], block.uses[r0:r1, c0:c1],
                                   block._connection_state[r0:r1, c0:c1])
                    if c1 == block.columns and block.columns % 4:
                        last = block._packed_weights[r0:r1, -1]
                        for lane in range(block.columns % 4, 4):
                            if bool((((last >> (lane * 2)) & 3) != 1).any()):
                                raise ValueError("sparse router block has nonzero packed padding")
                self.release(block, region)

    def iter_effective_tiles(self):
        with self.owner._ram(1024 + len(self.owner.blocks)*128, "canonical router recurrence order"):
            for key in sorted(self.owner.blocks):
                block = self.owner.blocks[key]
                row, column = parse_block_key(key)
                for region in self.regions(block):
                    self.owner._check()
                    r0, r1, c0, c1 = region
                    try:
                        with self.owner._ram((r1-r0)*(c1-c0)*16 + 1024, "allocated router signed tile"):
                            levels = unpack_tile(block._packed_weights, *region)
                            yield (r0 + row*BLOCK_ROWS, r1 + row*BLOCK_ROWS,
                                   c0 + column*BLOCK_COLUMNS, c1 + column*BLOCK_COLUMNS), weighted_ternary_levels(
                                       levels, block._connection_state[r0:r1, c0:c1])
                    finally:
                        self.release(block, region)

    def iter_packed_tiles(self, *, include_identity=False):
        """Admitted raw sign/gain tiles in the same canonical recurrence order.

        Every allocated repeated layer is visited. Missing blocks allocate
        nothing and contribute zero; decoding is exclusively inside the kernel.
        """
        with self.owner._ram(1024 + len(self.owner.blocks)*128, "canonical packed router recurrence order"):
            keys = sorted(self.owner.blocks)
            if include_identity:
                try:
                    layout_digest = sparse_layout_digest(self.owner, keys)
                except Exception:
                    # Identity validation belongs only to the optional
                    # observer. A malformed/stale observation seal must not
                    # change recurrent neural math or its ordinary errors.
                    layout_digest = None
            else:
                layout_digest = None
            for key in keys:
                block = self.owner.blocks[key]
                identity = (key, layout_digest) if include_identity else None
                row, column = parse_block_key(key)
                for region in self.regions(block):
                    self.owner._check()
                    r0, r1, c0, c1 = region
                    try:
                        # Raw byte copies plus activity promotion/output and
                        # validation scratch fit the existing admitted tile.
                        with self.owner._ram((r1-r0)*(c1-c0)*3 + (c1-c0)*8 + (r1-r0)*16 + 1024,
                                             "allocated router packed sign/gain tile"):
                            signs = block._packed_weights[r0:r1, c0//4:(c1+3)//4].detach().contiguous()
                            states = block._connection_state[r0:r1, c0:c1].detach().contiguous()
                            value = ((r0 + row*BLOCK_ROWS, r1 + row*BLOCK_ROWS,
                                      c0 + column*BLOCK_COLUMNS, c1 + column*BLOCK_COLUMNS), signs, states)
                            yield (*value, identity) if include_identity else value
                    finally:
                        self.release(block, region)

    def stability_total(self, rows, columns):
        total = 0.0
        for key, block in self.owner.blocks.items():
            row, column = parse_block_key(key)
            limit_rows = min(block.rows, rows - row*BLOCK_ROWS)
            limit_columns = min(block.columns, columns - column*BLOCK_COLUMNS)
            if limit_rows <= 0 or limit_columns <= 0:
                continue
            for region in tile_ranges(limit_rows, limit_columns, self.owner._tile_budget()):
                r0, r1, c0, c1 = region
                with self.owner._ram((r1-r0)*(c1-c0)*16 + 1024, "allocated router stability reduction"):
                    total += float(block.stability[r0:r1, c0:c1].to(torch.float64).sum())
                self.release(block, region)
        return total

    def set_levels(self, levels):
        with self.transaction() as journal:
            for row in range(math.ceil(self.owner.post_neurons / BLOCK_ROWS)):
                for column in range(math.ceil(self.owner.pre_neurons / BLOCK_COLUMNS)):
                    rows, columns = self.shape(row, column)
                    key = block_key(row, column)
                    for region in tile_ranges(rows, columns, self.owner._tile_budget()):
                        self.owner._check()
                        r0, r1, c0, c1 = region
                        with self.owner._ram((r1-r0)*(c1-c0)*16 + 1024, "explicit sparse router ternary tile"):
                            source = levels[row*BLOCK_ROWS+r0:row*BLOCK_ROWS+r1,
                                            column*BLOCK_COLUMNS+c0:column*BLOCK_COLUMNS+c1].detach().cpu()
                            replacement = pack_tile(source)
                            if key not in self.owner.blocks and not bool(source.ne(0).any()):
                                continue
                            block = self.ensure(row, column)
                            self.capture_writes(journal, key, block, region, {"_packed_weights": replacement})
                        self.release(block, region)

    def step(self, pre, post, old_pre, old_post):
        absolute = signed = 0.0
        level_abs = level_signed = changed_count = active_count = 0
        contact_updates = new_contacts = 0
        with self.transaction() as journal:
            vector_block = max(1, self.owner._tile_budget() // 32)
            for name in ("pre_trace", "post_trace"):
                for start in range(0, getattr(self.owner, name).numel(), vector_block):
                    journal.capture(name, start, min(getattr(self.owner, name).numel(), start + vector_block))
            journal.capture("plasticity_events")
            # Block support, not every possible pair or an NxN mask. Row/column
            # support is read directly from the already admitted timing vectors.
            for row in range(math.ceil(self.owner.post_neurons / BLOCK_ROWS)):
                self.owner._check()
                base_row = row * BLOCK_ROWS
                current_row = bool(post[base_row:base_row+BLOCK_ROWS].ne(0).any())
                historical_row = bool(old_post[base_row:base_row+BLOCK_ROWS].ne(0).any())
                if not (current_row or historical_row):
                    continue
                for column in range(math.ceil(self.owner.pre_neurons / BLOCK_COLUMNS)):
                    self.owner._check()
                    base_column = column * BLOCK_COLUMNS
                    if not ((current_row and bool(old_pre[base_column:base_column+BLOCK_COLUMNS].ne(0).any()))
                            or (historical_row and bool(pre[base_column:base_column+BLOCK_COLUMNS].ne(0).any()))):
                        continue
                    rows, columns = self.shape(row, column)
                    key = block_key(row, column)
                    for region in tile_ranges(rows, columns, self.owner._tile_budget()):
                        self.owner._check()
                        r0, r1, c0, c1 = region
                        with self.owner._ram((r1-r0)*(c1-c0)*ROUTER_CELL_SCRATCH_BYTES + 1024, "exact allocated causal/anti-causal STDP tile"):
                            timing = self.owner.a_plus * (post[base_row+r0:base_row+r1, None] * old_pre[None, base_column+c0:base_column+c1])
                            timing = timing - self.owner.a_minus * (old_post[base_row+r0:base_row+r1, None] * pre[None, base_column+c0:base_column+c1])
                            if not bool(timing.ne(0).any()):
                                continue
                            # Even zero-rate activity changes uses and stability.
                            # Do not prune it by a weight-only sparsity test.
                            self.ensure(row, column)
                            layers = self._layers[(row, column)]
                            changed = torch.zeros_like(timing, dtype=torch.bool)
                            growing = torch.zeros_like(timing, dtype=torch.int8)
                            with self.owner._ram(1024 + len(layers)*128, "parallel router layer order"):
                                for layer in sorted(layers):
                                    key = layers[layer]
                                    block = self.owner.blocks[key]
                                    previous = unpack_tile(block._packed_weights, *region)
                                    state = block._connection_state[r0:r1, c0:c1].clone()
                                    eligibility = block.eligibility_accumulator[r0:r1, c0:c1].clone()
                                    stability = block.stability[r0:r1, c0:c1].clone()
                                    uses = block.uses[r0:r1, c0:c1].clone()
                                    present = previous.ne(0) | state.ne(0) | eligibility.ne(0) | stability.ne(0) | uses.ne(0)
                                    local_timing = timing if not layer else timing * present
                                    delta, active, difference, replacements, overflow = _stdp_replacements(
                                        self.owner, previous, state, eligibility, stability, uses, local_timing)
                                    absolute += float(delta.abs().to(torch.float64).sum())
                                    signed += float(delta.to(torch.float64).sum())
                                    contact_updates += int(active.sum())
                                    if not layer:
                                        active_count += int(active.sum())
                                        growing = overflow
                                    else:
                                        amplitude = previous.to(torch.int32) * connection_strengths(state) + difference
                                        available = present & (amplitude * growing.to(torch.int32) < 16)
                                        growing = torch.where(available, 0, growing)
                                    level_abs += int(difference.abs().sum())
                                    level_signed += int(difference.sum())
                                    changed |= difference.ne(0)
                                    self.capture_writes(journal, key, block, region, replacements)
                                    self.release(block, region)
                            if bool(growing.ne(0).any()):
                                count = self._grow_contacts(row, column, region, growing, journal)
                                new_contacts += count
                                level_abs += count
                                level_signed += int(growing.to(torch.int32).sum())
                                changed |= growing.ne(0)
                            changed_count += int(changed.sum())
            self.owner._check()
            self.owner.pre_trace.mul_(self.owner.pre_decay).add_(pre.to(self.owner.pre_trace.device))
            self.owner.post_trace.mul_(self.owner.post_decay).add_(post.to(self.owner.post_trace.device))
            self.owner.plasticity_events.add_(active_count)
        return absolute, signed, level_abs, level_signed, changed_count, active_count, contact_updates, new_contacts

    @staticmethod
    def _present(block, region, levels=None):
        r0, r1, c0, c1 = region
        levels = unpack_tile(block._packed_weights, *region) if levels is None else levels
        return (levels.ne(0) | block._connection_state[r0:r1, c0:c1].ne(0)
                | block.eligibility_accumulator[r0:r1, c0:c1].ne(0)
                | block.stability[r0:r1, c0:c1].ne(0) | block.uses[r0:r1, c0:c1].ne(0))

    def _grow_contacts(self, row, column, region, directions, journal):
        """Fill free contact positions, then append an admitted parallel layer.

        One module stores a block of contacts, never one Python object per edge.
        Each repeated contact keeps its own sign, strength and timing controls.
        No layer count or connectivity ceiling is introduced.
        """
        r0, r1, c0, c1 = region
        remaining = directions.clone()
        count = int(remaining.ne(0).sum())
        layers = self._layers[(row, column)]
        with self.owner._ram(1024 + len(layers)*128, "parallel router available layer order"):
            for layer in sorted((*layers, max(layers) + 1)):
                if not layer or not bool(remaining.ne(0).any()):
                    continue
                block = self.ensure(row, column, layer=layer)
                key = block_key(row, column, layer)
                levels = unpack_tile(block._packed_weights, *region)
                free = remaining.ne(0) & ~self._present(block, region, levels)
                if bool(free.any()):
                    levels = torch.where(free, remaining, levels)
                    uses = block.uses[r0:r1, c0:c1].clone()
                    stability = block.stability[r0:r1, c0:c1].clone()
                    uses[free] = 1.0
                    stability[free] = max(0.0, min(20.0, self.owner.metaplasticity_rate))
                    self.capture_writes(journal, key, block, region, {
                        "_packed_weights": pack_tile(levels), "uses": uses, "stability": stability})
                    remaining = torch.where(free, 0, remaining)
                self.release(block, region)
        if bool(remaining.ne(0).any()):
            raise RuntimeError("parallel router contact admission did not cover its exact requested pairs")
        return count

    def decay_unused(self, amount):
        with self.transaction() as journal:
            journal.capture("decay_cycles")
            self.owner.decay_cycles.add_(1)
            tick = int(self.owner.decay_cycles)
            for key, block in self.owner.blocks.items():
                row, column = parse_block_key(key)
                for region in self.regions(block):
                    self.owner._check()
                    r0, r1, c0, c1 = region
                    with self.owner._ram((r1-r0)*(c1-c0)*96 + 1024, "exact allocated router decay tile"):
                        if amount:
                            levels = unpack_tile(block._packed_weights, *region)
                            positions = ((torch.arange(r0, r1, dtype=torch.int64) + row*BLOCK_ROWS)[:, None] * self.owner.pre_neurons
                                         + (torch.arange(c0, c1, dtype=torch.int64) + column*BLOCK_COLUMNS)[None, :])
                            draw = ((positions * 1664525 + tick * 1013904223 + block_layer(key)*2246822519) & 0xFFFFFFFF).float() / 4294967296.0
                            uses = block.uses[r0:r1, c0:c1]
                            decay = levels.ne(0) & (draw < amount / (1.0 + uses))
                            replacements = {"stability": block.stability[r0:r1, c0:c1] * (1.0 - amount * .1)}
                            if bool(decay.any()):
                                state = block._connection_state[r0:r1, c0:c1]
                                strengths = connection_strengths(state).to(torch.int32)
                                resistance = connection_resistance(state).to(torch.int32)
                                faded = torch.where(decay, strengths - 1, strengths)
                                resistance = torch.where(decay, (resistance - 1).clamp_min(0), resistance)
                                replacements["_packed_weights"] = pack_tile(torch.where(faded.eq(0), 0, levels).to(torch.int8))
                                replacements["_connection_state"] = encode_connection_state(faded.clamp_min(1), resistance)
                                bound = 256*(1+resistance)-1
                                pressure = block.eligibility_accumulator[r0:r1, c0:c1].to(torch.int32)
                                replacements["eligibility_accumulator"] = torch.maximum(-bound, torch.minimum(bound, pressure)).to(torch.int16)
                            self.capture_writes(journal, key, block, region, replacements)
                    self.release(block, region)

    def dense_field_chunks(self, name):
        """Historical row-major bytes only; recurrence never takes this path."""
        dtype = torch.int8 if name == "weights" else _DTYPES[name]
        elements = max(4, (self.owner._tile_budget() - 1024) // 32 // 4 * 4)
        for row in range(self.owner.post_neurons):
            for start in range(0, self.owner.pre_neurons, elements):
                end = min(self.owner.pre_neurons, start + elements)
                self.owner._check()
                with self.owner._ram((end-start)*32 + 1024, "historical sparse router virtual row bytes"):
                    result = torch.zeros(end-start, dtype=dtype)
                    position = start
                    while position < end:
                        column = position // BLOCK_COLUMNS
                        stop = min(end, (column+1)*BLOCK_COLUMNS)
                        key = block_key(row // BLOCK_ROWS, column)
                        if key in self.owner.blocks:
                            block = self.owner.blocks[key]
                            local_row, local_column = row % BLOCK_ROWS, position % BLOCK_COLUMNS
                            if name == "weights":
                                source = unpack_tile(block._packed_weights, local_row, local_row+1,
                                                     local_column, local_column+stop-position).reshape(-1)
                            else:
                                source = getattr(block, name)[local_row, local_column:local_column+stop-position]
                            result[position-start:stop-start].copy_(source)
                            self.release(block, (local_row, local_row+1, local_column, local_column+stop-position))
                        position = stop
                    yield result

    def update_control_checksum(self, digest, name):
        digest.update(str((self.owner.post_neurons, self.owner.pre_neurons)).encode("ascii"))
        digest.update(str(_DTYPES[name]).encode("ascii"))
        for value in self.dense_field_chunks(name):
            digest.update(memoryview(value.view(torch.uint8).numpy()))

    def historical_checksum(self, *, controls):
        """Explicit legacy equivalence proof, never normal chat integrity."""
        digest = hashlib.sha256()
        digest.update(str((self.owner.post_neurons, self.owner.pre_neurons)).encode("ascii"))
        digest.update(str(torch.int8).encode("ascii"))
        for value in self.dense_field_chunks("weights"):
            digest.update(memoryview(value.view(torch.uint8).numpy()))
        if controls:
            self.update_control_checksum(digest, "stability")
            self.update_control_checksum(digest, "uses")
            update_historical_tensor_checksum(digest, self.owner.plasticity_events, self.owner)
        return digest.hexdigest()

    def canonical_checksum(self, *, controls):
        """Dimension-bound allocated state only, independent of insertion order."""
        digest = hashlib.sha256(b"omni-router-allocated-block-state-v1\0")
        update_historical_tensor_checksum(digest, self.owner._sparse_layout, self.owner)
        # Sorting actual block identities is admitted proportional metadata,
        # not a potential-connectivity bitmap or matrix.
        with self.owner._ram(1024 + len(self.owner.blocks)*128, "canonical allocated router block order"):
            keys = sorted(self.owner.blocks)
            for key in keys:
                self.owner._check()
                block = self.owner.blocks[key]
                digest.update(key.encode("ascii") + b"\0")
                for field in (MATRIX_FIELDS if controls else ("_packed_weights", "_connection_state")):
                    update_historical_tensor_checksum(digest, getattr(block, field), self.owner)
            if controls:
                for name in ("pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
                    update_historical_tensor_checksum(digest, getattr(self.owner, name), self.owner)
        return digest.hexdigest()

    def _source_bytes(self, reader, name):
        spec = reader.specs[name]
        step = max(8, min(reader.chunk_bytes, max(8, (self.owner._tile_budget()-1024)//3)))
        with reader.path.open("rb", buffering=0) as handle:
            handle.seek(spec.offset)
            for start in range(0, spec.byte_count, step):
                self.owner._check()
                count = min(step, spec.byte_count-start)
                with self.owner._ram(count*3+1024, "native migration source fingerprint bytes"):
                    raw = handle.read(count)
                    if len(raw) != count:
                        raise ValueError("native router source fingerprint was truncated")
                    reader.peak_transfer_bytes = max(reader.peak_transfer_bytes, count)
                    yield raw

    def _source_historical_checksum(self, reader, prefix):
        digest = hashlib.sha256()
        digest.update(str((self.owner.post_neurons, self.owner.pre_neurons)).encode("ascii"))
        digest.update(str(torch.int8).encode("ascii"))
        step = max(4, (self.owner._tile_budget()-1024)//32//4*4)
        for row in range(self.owner.post_neurons):
            for column in range(0, self.owner.pre_neurons, step):
                end = min(self.owner.pre_neurons, column+step)
                with self.owner._ram((end-column)*32+1024, "historical dense-native source trits"):
                    packed = self._read_region(reader, prefix+"_packed_weights", row, row+1, column//4, (end+3)//4)
                    value = unpack_tile(packed, 0, 1, 0, end-column).reshape(-1)
                    digest.update(memoryview(value.view(torch.uint8).numpy()))
        for field in ("stability", "uses", "plasticity_events"):
            spec = reader.specs[prefix+field]
            digest.update(str(spec.shape).encode("ascii")); digest.update(str(spec.dtype).encode("ascii"))
            for raw in self._source_bytes(reader, prefix+field):
                digest.update(raw)
        return digest.hexdigest()

    def _dense_equivalence_proof(self, reader, prefix):
        fingerprints = {}
        for name in (*(field for field in MATRIX_FIELDS if prefix+field in reader.specs), "pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
            spec = reader.specs[prefix+name]
            source, restored = hashlib.sha256(), hashlib.sha256()
            for digest in (source, restored):
                digest.update(str(spec.shape).encode("ascii")); digest.update(str(spec.dtype).encode("ascii"))
            for raw in self._source_bytes(reader, prefix+name):
                source.update(raw)
            if name in MATRIX_FIELDS:
                field = "weights" if name == "_packed_weights" else name
                for value in self.dense_field_chunks(field):
                    if name == "_packed_weights":
                        value = pack_tile(value.reshape(1, -1)).reshape(-1)
                    restored.update(memoryview(value.view(torch.uint8).numpy()))
            else:
                # Reuse the bounded persistence checksum encoder for vectors;
                # it includes the same shape/dtype prefix exactly once.
                restored = hashlib.sha256()
                update_historical_tensor_checksum(restored, getattr(self.owner, name), self.owner)
            if source.digest() != restored.digest():
                raise ValueError("dense-native router migration changed authoritative field: " + name)
            fingerprints[name] = {"sha256": source.hexdigest(), "shape": list(spec.shape), "dtype": str(spec.dtype)}
        before = self._source_historical_checksum(reader, prefix)
        after = self.historical_checksum(controls=True)
        if before != after:
            raise ValueError("dense-native router migration changed its historical trit/control checksum")
        return {"format": "omni-router-dense-native-migration-v1", "sourceMode": "dense-packed-native",
                "destinationMode": "allocated-block-sparse-native", "legacyDenseMigration": True,
                "sourceHistoricalChecksum": before, "destinationHistoricalChecksum": after,
                "exactNativeFieldsPreserved": True, "nativeFieldFingerprints": fingerprints,
                "populationShape": [self.owner.post_neurons, self.owner.pre_neurons],
                "sourceLogicalTrits": self.owner.post_neurons*self.owner.pre_neurons,
                "destinationAllocatedTrits": self.allocated_trits, "allocatedBlocks": len(self.owner.blocks),
                "implicitDefaults": {"ternary": 0, "eligibility": 0, "stability": 0, "uses": 0},
                "verifiedAtLoad": True, "checkpointWritableMapped": False}

    def copy_from(self, source):
        """Grow/copy allocated blocks, retaining old border trits and controls."""
        if self.owner.blocks or self.owner.pre_neurons < source.owner.pre_neurons or self.owner.post_neurons < source.owner.post_neurons:
            raise ValueError("sparse router prefix copy requires a fresh nonshrinking destination")
        with self.transaction():
            for key, old in source.owner.blocks.items():
                row, column = parse_block_key(key)
                new = self.ensure(row, column, layer=block_layer(key))
                for region in source.regions(old):
                    source.owner._check(); self.owner._check()
                    r0, r1, c0, c1 = region
                    with self.owner._ram((r1-r0)*(c1-c0)*32 + 1024, "exact sparse router growth prefix"):
                        for name in MATRIX_FIELDS:
                            a, b = (c0//4, (c1+3)//4) if name == "_packed_weights" else (c0, c1)
                            getattr(new, name)[r0:r1, a:b].copy_(getattr(old, name)[r0:r1, a:b])
                    source.release(old, region); self.release(new, region)

    def release_all(self):
        if self.pager is not None:
            for block in self.owner.blocks.values():
                self.pager.release_owner(block)

    def _checkpoint_inventory(self, reader, prefix):
        """Validate all native headers before allocating or copying a block."""
        specs = reader.specs
        if prefix + "weights" in specs:
            raise ValueError("floating learned router weights are incompatible with packed-native state")
        vectors = {"pre_trace": ((self.owner.pre_neurons,), torch.float32),
                   "post_trace": ((self.owner.post_neurons,), torch.float32),
                   "plasticity_events": ((), torch.int64), "decay_cycles": ((), torch.int64)}
        for name, (shape, dtype) in vectors.items():
            spec = specs.get(prefix + name)
            if spec is None or spec.shape != shape or spec.dtype != dtype:
                raise ValueError("router checkpoint timing vector/counter shape or dtype is invalid: " + name)
        sparse = prefix + "_sparse_layout" in specs
        source_version = 2 if prefix + "_connection_state" in specs else 1
        if sparse:
            layout_spec = specs[prefix + "_sparse_layout"]
            if layout_spec.shape != (5,) or layout_spec.dtype != torch.int64:
                raise ValueError("sparse router layout header is invalid")
            with self.owner._ram(1024, "sparse router layout read"):
                layout = torch.empty(5, dtype=torch.int64)
                self._copy_flat(reader, prefix + "_sparse_layout", layout)
                source_version = int(layout[0])
                if source_version not in (1, SPARSE_LAYOUT_VERSION) or not torch.equal(layout[1:], self.owner._sparse_layout.detach().cpu()[1:]):
                    raise ValueError("sparse router layout does not match its declared population")
        fields = MATRIX_FIELDS if source_version >= 2 else LEGACY_MATRIX_FIELDS
        if not sparse:
            for name in fields:
                spec = specs.get(prefix + name)
                shape = (self.owner.post_neurons, (self.owner.pre_neurons+3)//4) if name == "_packed_weights" else (self.owner.post_neurons, self.owner.pre_neurons)
                if spec is None or spec.shape != shape or spec.dtype != _DTYPES[name]:
                    raise ValueError("dense-native router checkpoint field shape or dtype is invalid: " + name)
        for full_name, spec in specs.items():
            if not full_name.startswith(prefix):
                continue
            name = full_name[len(prefix):]
            if name in vectors or sparse and name == "_sparse_layout" or not sparse and name in fields:
                continue
            parts = name.split(".")
            if not sparse or len(parts) != 3 or parts[0] != "blocks" or parts[2] not in fields:
                raise ValueError("unexpected recurrent router checkpoint field: " + name)
            row, column = parse_block_key(parts[1])
            if block_layer(parts[1]) and (source_version < 2 or prefix+"blocks."+block_key(row, column)+"._packed_weights" not in specs):
                raise ValueError("parallel router contact layer has no native base block")
            rows, columns = self.shape(row, column)
            expected = (rows, (columns+3)//4) if parts[2] == "_packed_weights" else (rows, columns)
            if spec.shape != expected or spec.dtype != _DTYPES[parts[2]]:
                raise ValueError("sparse router checkpoint block shape/dtype is invalid: " + name)
            for field in fields:
                if prefix + "blocks." + parts[1] + "." + field not in specs:
                    raise ValueError("sparse router checkpoint block is incomplete")
        return sparse, source_version, fields

    def _copy_flat(self, reader, name, target):
        spec = reader.specs[name]
        if spec.dtype != target.dtype or spec.shape != tuple(target.shape) or not target.is_contiguous():
            raise ValueError("bounded router copy target geometry is invalid")
        item = target.element_size()
        step = max(item, min(reader.chunk_bytes, max(item, (self.owner._tile_budget()-1024)//3)) // item * item)
        flat = target.reshape(-1)
        with reader.path.open("rb", buffering=0) as handle:
            handle.seek(spec.offset)
            for start in range(0, spec.byte_count, step):
                self.owner._check()
                count = min(step, spec.byte_count-start)
                with self.owner._ram(count*3 + 1024, "bounded sparse router checkpoint bytes"):
                    raw = bytearray(count)
                    if handle.readinto(raw) != count:
                        raise ValueError("router checkpoint was truncated during load")
                    flat[start//item:(start+count)//item].copy_(torch.frombuffer(raw, dtype=spec.dtype).to(target.device))
                    reader.peak_transfer_bytes = max(reader.peak_transfer_bytes, count)
                release_router_tensor_chunk(target, start, count)

    def _read_region(self, reader, name, r0, r1, c0, c1):
        """One admitted rectangle, without reader.tensor or a matrix mapping."""
        spec = reader.specs[name]
        if len(spec.shape) != 2 or not (0 <= r0 <= r1 <= spec.shape[0] and 0 <= c0 <= c1 <= spec.shape[1]):
            raise ValueError("router checkpoint rectangle is outside the source")
        item = torch.empty((), dtype=spec.dtype).element_size()
        result = torch.empty((r1-r0, c1-c0), dtype=spec.dtype)
        with reader.path.open("rb", buffering=0) as handle:
            raw = bytearray((c1-c0)*item)
            for index, row in enumerate(range(r0, r1)):
                self.owner._check()
                handle.seek(spec.offset + (row*spec.shape[1]+c0)*item)
                if handle.readinto(raw) != len(raw):
                    raise ValueError("router checkpoint was truncated during native migration")
                result[index].copy_(torch.frombuffer(raw, dtype=spec.dtype))
                reader.peak_transfer_bytes = max(reader.peak_transfer_bytes, len(raw))
        return result

    def load_bounded(self, reader, prefix):
        """Only a fresh owner may load; failure leaves it unusable, not partial.

        The outer loader excludes this prefix. All saved blocks and controls
        are loaded here, or a dense-native archive is scanned once in bounded
        tiles. Nonzero metadata on a zero edge is still allocated and retained.
        """
        if self.owner.blocks:
            raise RuntimeError("recurrent checkpoint loading requires a fresh sparse owner")
        self.ready = False
        sparse, source_version, fields = self._checkpoint_inventory(reader, prefix)
        with self.transaction():
            if sparse:
                for name in reader.specs:
                    if not name.startswith(prefix + "blocks.") or not name.endswith("._packed_weights"):
                        continue
                    key = name[len(prefix + "blocks."):].split(".")[0]
                    row, column = parse_block_key(key)
                    block = self.ensure(row, column, layer=block_layer(key), loading=True)
                    for field in fields:
                        self._copy_flat(reader, prefix + "blocks." + key + "." + field, getattr(block, field))
                    if "_connection_state" not in fields:
                        for region in self.regions(block):
                            r0, r1, c0, c1 = region
                            block._connection_state[r0:r1, c0:c1].zero_()
                            self.release(block, region)
                    block._router_loaded = True
            else:
                for row in range(math.ceil(self.owner.post_neurons/BLOCK_ROWS)):
                    for column in range(math.ceil(self.owner.pre_neurons/BLOCK_COLUMNS)):
                        rows, columns = self.shape(row, column)
                        for region in tile_ranges(rows, columns, self.owner._tile_budget()):
                            self.owner._check()
                            r0, r1, c0, c1 = region
                            a, b = row*BLOCK_ROWS, column*BLOCK_COLUMNS
                            with self.owner._ram((r1-r0)*(c1-c0)*96 + 1024, "bounded dense-native router migration tile"):
                                packed = self._read_region(reader, prefix + "_packed_weights", a+r0, a+r1, (b+c0)//4, (b+c1+3)//4)
                                levels = unpack_tile(packed, 0, r1-r0, 0, c1-c0)
                                if b+c1 == self.owner.pre_neurons and self.owner.pre_neurons % 4:
                                    for lane in range(self.owner.pre_neurons % 4, 4):
                                        if bool((((packed[:, -1] >> (2*lane)) & 3) != 1).any()):
                                            raise ValueError("dense-native router has nonzero packed padding")
                                values = {field: self._read_region(reader, prefix + field, a+r0, a+r1, b+c0, b+c1)
                                          for field in fields if field != "_packed_weights"}
                                _validate_tile(levels, values["eligibility_accumulator"], values["stability"], values["uses"], values.get("_connection_state"))
                                nondefault = bool(levels.ne(0).any()) or any(bool(value.ne(0).any()) for value in values.values())
                                if nondefault:
                                    block = self.ensure(row, column)
                                    block._packed_weights[r0:r1, c0//4:(c1+3)//4].copy_(packed)
                                    for field, value in values.items():
                                        getattr(block, field)[r0:r1, c0:c1].copy_(value)
                                    self.release(block, region)
            for name in ("pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
                self._copy_flat(reader, prefix + name, getattr(self.owner, name))
            self.validate()
            for name in ("pre_trace", "post_trace"):
                vector = getattr(self.owner, name)
                for start in range(0, vector.numel(), max(1, self.owner._tile_budget()//32)):
                    part = vector[start:start + max(1, self.owner._tile_budget()//32)]
                    if not bool(torch.isfinite(part).all()) or bool((part < 0).any()):
                        raise ValueError("router checkpoint timing vector is invalid")
            if int(self.owner.plasticity_events) < 0 or int(self.owner.decay_cycles) < 0:
                raise ValueError("router checkpoint counters must be nonnegative")
            proof = self._dense_equivalence_proof(reader, prefix) if not sparse else {
                "format": "omni-router-sparse-native-load-v2", "sourceMode": "allocated-block-sparse-native",
                "destinationMode": "allocated-block-sparse-native", "legacyDenseMigration": False,
                "populationShape": [self.owner.post_neurons, self.owner.pre_neurons],
                "allocatedBlocks": len(self.owner.blocks), "destinationAllocatedTrits": self.allocated_trits,
                "allNativeHeadersValidated": True, "verifiedAtLoad": True, "checkpointWritableMapped": False}
            proof.update({"sourceConnectionStateVersion": source_version,
                          "destinationConnectionStateVersion": SPARSE_LAYOUT_VERSION,
                          "unitStrengthMigration": source_version == 1})
        if self.pager is not None:
            for block in self.owner.blocks.values():
                self.pager.finish_owner_load(block, flush=False)
            self.pager.flush()
        self.owner._router_loading_checkpoint = False
        self.ready = True
        self.migration_proof = proof
        return proof
