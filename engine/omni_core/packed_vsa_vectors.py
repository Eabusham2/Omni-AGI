"""Packed, authoritative ternary vectors for the adaptive VSA substrate.

This is a storage primitive, not a second memory or an answering path. Learned
state is canonical two-bit ``{-1, 0, +1}`` signs, one integer strength/resistance
byte per base contact, and explicit packed repeated contacts. Decoded activity
tensors are short-lived copies. The per-row counter resumes deterministic
small-rate discrete adaptations after a checkpoint.

The mapping returns transient unit-norm float32 rows for VSA activation math;
``levels()`` returns base int8 signs and ``weighted_levels()`` returns actual
summed integer contributions. No float learned row or scale is retained here.
``export_state`` splits JSON-safe metadata from CPU uint8 tensors suitable for
``safetensors``; callers must save both parts together.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Iterator

import torch


FORMAT = "omni-packed-vsa-vectors"
FORMAT_VERSION = 2
MAX_DECODE_ROWS = 64
_MAX_UPDATES = (1 << 63) - 1
_MASK64 = (1 << 64) - 1
_FULL_BYTE_VALID = tuple(
    all(((byte >> shift) & 3) != 3 for shift in (0, 2, 4, 6))
    for byte in range(256)
)


def _valid_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("packed vector dimensions must be a positive integer")
    return value


def _valid_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("packed vector seed must be an integer")
    return value


def _valid_deadband(value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("packed vector deadband must be finite and in [0, 1)")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("packed vector deadband must be finite and in [0, 1)") from error
    if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError("packed vector deadband must be finite and in [0, 1)")
    return numeric


def _valid_identifier(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("packed vector identifiers must be nonempty strings")
    return value


def quantize_vsa_vector(
    vector: torch.Tensor | Sequence[float],
    dimensions: int,
    *,
    zero_deadband: float = 0.25,
) -> torch.Tensor:
    """Deterministically quantize one finite row with a relative deadband.

    A fixed absolute threshold would turn unit-norm sensory vectors into all
    zeroes as dimensions increase. Here coordinates strictly above/below
    ``zero_deadband * max(abs(row))`` become +/-1; ties and small coordinates
    become zero. Exact ternary input is unchanged for any allowed deadband.
    """

    width = _valid_dimension(dimensions)
    deadband = _valid_deadband(zero_deadband)
    try:
        source = torch.as_tensor(vector)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("packed vector input must be numeric") from error
    if source.is_complex() or source.dtype == torch.bool or source.numel() != width:
        raise ValueError("packed vector input must be real and match dimensions")
    try:
        values = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("packed vector input must be numeric") from error
    if not bool(torch.isfinite(values).all()):
        raise ValueError("packed vector input must contain only finite values")
    maximum = float(values.abs().max().item())
    threshold = deadband * maximum
    return torch.where(
        values > threshold,
        1,
        torch.where(values < -threshold, -1, 0),
    ).to(torch.int8)


def _pack_exact_row(levels: torch.Tensor, dimensions: int) -> bytes:
    if (
        not isinstance(levels, torch.Tensor)
        or levels.dtype != torch.int8
        or levels.shape != (dimensions,)
        or not bool(((levels >= -1) & (levels <= 1)).all())
    ):
        raise ValueError("packed vector levels must be exact int8 ternary rows")
    row = bytearray([0x55] * ((dimensions + 3) // 4))
    for index, level in enumerate(levels.tolist()):
        shift = (index % 4) * 2
        row[index // 4] = (row[index // 4] & ~(3 << shift)) | ((level + 1) << shift)
    return bytes(row)


def _validate_packed_row(row: bytes | bytearray | memoryview, dimensions: int) -> None:
    width = (dimensions + 3) // 4
    if len(row) != width:
        raise ValueError("packed vector row width is invalid")
    complete_bytes = dimensions // 4
    if any(not _FULL_BYTE_VALID[byte] for byte in row[:complete_bytes]):
        raise ValueError("packed vector contains a reserved two-bit code")
    remainder = dimensions % 4
    if remainder:
        last = row[-1]
        for lane in range(remainder):
            if ((last >> (lane * 2)) & 3) == 3:
                raise ValueError("packed vector contains a reserved two-bit code")
        for lane in range(remainder, 4):
            if ((last >> (lane * 2)) & 3) != 1:
                raise ValueError("packed vector has non-canonical zero padding")


def _decode_exact_row(row: bytes | bytearray | memoryview, dimensions: int) -> torch.Tensor:
    _validate_packed_row(row, dimensions)
    return torch.tensor(
        [(row[index // 4] >> ((index % 4) * 2) & 3) - 1 for index in range(dimensions)],
        dtype=torch.int8,
    )


def _splitmix64(state: int) -> tuple[int, float]:
    """Stable 53-bit draw without dependence on process-global RNG state."""

    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK64
    mixed ^= mixed >> 31
    return state, (mixed >> 11) / float(1 << 53)


class PackedTernaryVectors(MutableMapping[str, torch.Tensor]):
    """Insertion-ordered mapping backed only by packed ternary row bytes.

    ``__getitem__``, ``get``, ``items``, and ``values`` decode one row at a
    time. ``decode_rows`` is capped at 64 rows. Deletion wipes and recycles a
    slot; export compacts only live rows in mapping order.
    """

    packed_authoritative = True

    @staticmethod
    def tensor_names(metadata, *, prefix=""):
        version = metadata.get("formatVersion") if isinstance(metadata, Mapping) else None
        if type(version) is not int or version not in (1, FORMAT_VERSION):
            raise ValueError("packed vector format is incompatible")
        names = {"packed_rows", "update_counters_le"}
        if version == 2:
            names |= {"connection_state", "parallel_offsets_le", "parallel_records"}
        return {prefix + name for name in names}

    def __init__(
        self,
        dimensions: int,
        *,
        seed: int = 0,
        zero_deadband: float = 0.25,
        initial: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.dimensions = _valid_dimension(dimensions)
        self.seed = _valid_seed(seed)
        self.zero_deadband = _valid_deadband(zero_deadband)
        self.row_bytes = (self.dimensions + 3) // 4
        self._locations: dict[str, int] = {}
        self._rows = bytearray()
        self._update_counters = bytearray()
        self._connection_state = bytearray()
        self._parallel_records: dict[int, bytes] = {}
        self._parallel_record_bytes = 0
        self.source_format_version = FORMAT_VERSION
        # Execution guard only, never a checkpoint/checksum field. A stream
        # must fail if live state changed even when it was later changed back.
        self._revision = 0
        self._free_slots: list[int] = []
        if initial is not None:
            self.update(initial)

    @property
    def storage_bytes(self) -> int:
        """Actual allocated packed payload and counter bytes, including holes."""

        return len(self._rows) + len(self._update_counters) + len(self._connection_state) + self._parallel_record_bytes

    def parameter_accounting(self) -> dict[str, int]:
        count = len(self)
        repeated = self._parallel_record_bytes // 10
        learned = count * (self.row_bytes + self.dimensions) + repeated * 2
        structural = count * 8 + repeated * 8
        return {"rows": count, "baseContacts": count * self.dimensions,
            "repeatedContacts": repeated, "logicalParameters": count * self.dimensions + repeated,
            "learnedBytes": learned, "structuralBytes": structural,
            "connectionStateBytes": count * self.dimensions + repeated,
            "rowEnvelopeBytes": 0, "allocatedStateBytes": self.storage_bytes}

    @property
    def revision(self) -> int:
        return self._revision

    def current_snapshot(self) -> str:
        """Process-local identity for generation-bound streamed reads."""
        return "resident-packed-vsa:%x:%d:%d" % (id(self), self._revision, len(self))

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self) -> Iterator[str]:
        return iter(self._locations)

    def __contains__(self, key: object) -> bool:
        return key in self._locations

    def _row(self, slot: int) -> memoryview:
        start = slot * self.row_bytes
        return memoryview(self._rows)[start : start + self.row_bytes]

    def _counter(self, slot: int) -> int:
        return struct.unpack_from("<Q", self._update_counters, slot * 8)[0]

    def _set_counter(self, slot: int, value: int) -> None:
        struct.pack_into("<Q", self._update_counters, slot * 8, value)

    def __getitem__(self, key: str) -> torch.Tensor:
        from .weighted_vsa_row import activity
        return activity(self.row_payload(key), self.dimensions)

    def row_payload(self, key: str) -> bytes:
        from .weighted_vsa_row import make_payload
        slot = self._locations[key]
        return make_payload(self.packed_row(key), self.connection_state(key), self.parallel_records(key), self.dimensions)

    def connection_state(self, key: str) -> bytes:
        slot = self._locations[key]
        return bytes(self._connection_state[slot * self.dimensions:(slot + 1) * self.dimensions])

    def parallel_records(self, key: str) -> bytes:
        return self._parallel_records.get(self._locations[key], b"")

    def weighted_levels(self, key: str) -> torch.Tensor:
        from .weighted_vsa_row import weighted_levels
        return weighted_levels(self.row_payload(key), self.dimensions)

    def set_connection_state(self, key: str, state: bytes, repeats: bytes = b"") -> None:
        from .weighted_vsa_row import make_payload
        make_payload(self.packed_row(key), state, repeats, self.dimensions)
        slot = self._locations[key]
        self._parallel_record_bytes += len(repeats) - len(self._parallel_records.get(slot, b""))
        self._connection_state[slot * self.dimensions:(slot + 1) * self.dimensions] = state
        if repeats:
            self._parallel_records[slot] = repeats
        else:
            self._parallel_records.pop(slot, None)
        self._revision += 1

    def append_row_payload(self, key: str, payload: bytes, updates: int = 0) -> None:
        """Append verified native bytes without decoding or resetting counters."""
        from .weighted_vsa_row import payload_parts
        identifier = _valid_identifier(key)
        if identifier in self._locations or type(updates) is not int or not 0 <= updates <= _MAX_UPDATES:
            raise ValueError("packed vector append identity or update count is invalid")
        signs, states, repeats = payload_parts(payload, self.dimensions)
        slot = len(self._rows) // self.row_bytes
        self._rows.extend(signs)
        self._connection_state.extend(states)
        self._update_counters.extend(struct.pack("<Q", updates))
        self._locations[identifier] = slot
        if repeats:
            self._parallel_records[slot] = repeats
            self._parallel_record_bytes += len(repeats)
        self._revision += 1

    def levels(self, key: str) -> torch.Tensor:
        """Return an independent exact int8 copy of one row."""

        return _decode_exact_row(self._row(self._locations[key]), self.dimensions)

    def packed_row(self, key: str) -> bytes:
        """Return immutable canonical bytes for one row without decoding it."""

        return bytes(self._row(self._locations[key]))

    def update_count(self, key: str) -> int:
        return self._counter(self._locations[key])

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        identifier = _valid_identifier(key)
        packed = _pack_exact_row(
            quantize_vsa_vector(value, self.dimensions, zero_deadband=self.zero_deadband),
            self.dimensions,
        )
        if identifier in self._locations:
            slot = self._locations[identifier]
        elif self._free_slots:
            slot = self._free_slots.pop()
            self._locations[identifier] = slot
        else:
            slot = len(self._rows) // self.row_bytes
            self._rows.extend(bytes([0x55]) * self.row_bytes)
            self._update_counters.extend(b"\x00" * 8)
            self._connection_state.extend(bytes(self.dimensions))
            self._locations[identifier] = slot
        self._row(slot)[:] = packed
        self._set_counter(slot, 0)
        self._connection_state[slot * self.dimensions:(slot + 1) * self.dimensions] = bytes(self.dimensions)
        self._parallel_record_bytes -= len(self._parallel_records.pop(slot, b""))
        self._revision += 1

    def __delitem__(self, key: str) -> None:
        slot = self._locations.pop(key)
        self._row(slot)[:] = bytes([0x55]) * self.row_bytes
        self._set_counter(slot, 0)
        self._connection_state[slot * self.dimensions:(slot + 1) * self.dimensions] = bytes(self.dimensions)
        self._parallel_record_bytes -= len(self._parallel_records.pop(slot, b""))
        self._free_slots.append(slot)
        self._revision += 1

    def clear(self) -> None:
        if self._locations or self._rows:
            self._revision += 1
        self._locations.clear()
        self._rows.clear()
        self._update_counters.clear()
        self._connection_state.clear()
        self._parallel_records.clear()
        self._parallel_record_bytes = 0
        self._free_slots.clear()

    def decode_rows(self, keys: Sequence[str]) -> torch.Tensor:
        """Decode at most ``MAX_DECODE_ROWS`` requested rows as a copy."""

        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise ValueError("packed vector row request must be a sequence of IDs")
        if len(keys) > MAX_DECODE_ROWS:
            raise ValueError("packed vector row decode exceeds its bounded window")
        if not keys:
            return torch.empty((0, self.dimensions), dtype=torch.float32)
        return torch.stack([self[key] for key in keys])

    def adapt(self, key: str, target: torch.Tensor, rate: float) -> torch.Tensor:
        """Apply a resumable stochastic-rounded ternary blend to one row.

        The draw is derived from ``seed, ID, row update count`` and coordinate;
        it is not a float master or ambient random state. The expected new
        level equals ``old + rate * (quantized_target - old)``. A small rate
        can therefore change discrete levels across repeated exposures.
        """

        if isinstance(rate, (bool, str, bytes)):
            raise ValueError("packed vector adaptation rate must be in [0, 1]")
        try:
            alpha = float(rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("packed vector adaptation rate must be in [0, 1]") from error
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("packed vector adaptation rate must be in [0, 1]")
        slot = self._locations[key]
        from .weighted_vsa_row import adapt_payload, payload_parts
        counter = self._counter(slot)
        if counter >= _MAX_UPDATES:
            raise OverflowError("packed vector update counter is exhausted")
        payload, attempted = adapt_payload(self.row_payload(key), target, alpha, self.dimensions,
            seed=self.seed, identifier=key, updates=counter, zero_deadband=self.zero_deadband)
        if not attempted:
            return self[key]
        signs, state, repeats = payload_parts(payload, self.dimensions)
        self._row(slot)[:] = signs
        self.set_connection_state(key, state, repeats)
        self._set_counter(slot, counter + 1)
        return self[key]

    def export_state(
        self, *, prefix: str = "", keys: Sequence[str] | None = None
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Snapshot chosen rows as JSON metadata and independent CPU uint8 tensors.

        A bounded shard can export its own ID subset without decoding vectors
        or rewriting the other shards' vector state.
        """

        if not isinstance(prefix, str):
            raise ValueError("packed vector tensor prefix must be a string")
        if keys is None:
            ids = list(self._locations)
        else:
            if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
                raise ValueError("packed vector export keys must be a sequence")
            ids = list(keys)
            if any(not isinstance(identifier, str) for identifier in ids):
                raise ValueError("packed vector export keys are invalid")
            if len(set(ids)) != len(ids) or any(
                identifier not in self._locations for identifier in ids
            ):
                raise ValueError("packed vector export keys are invalid")
        rows = bytearray(len(ids) * self.row_bytes)
        counters = bytearray(len(ids) * 8)
        connection_state = bytearray(len(ids) * self.dimensions)
        offsets = bytearray((len(ids) + 1) * 8)
        parallel = bytearray()
        for index, identifier in enumerate(ids):
            slot = self._locations[identifier]
            row = self._row(slot)
            _validate_packed_row(row, self.dimensions)
            if self._counter(slot) > _MAX_UPDATES:
                raise ValueError("packed vector update counter is invalid")
            rows[index * self.row_bytes : (index + 1) * self.row_bytes] = row
            counters[index * 8 : (index + 1) * 8] = self._update_counters[
                slot * 8 : (slot + 1) * 8
            ]
            connection_state[index * self.dimensions:(index + 1) * self.dimensions] = self.connection_state(identifier)
            parallel.extend(self.parallel_records(identifier))
            struct.pack_into("<Q", offsets, (index + 1) * 8, len(parallel) // 10)
        packed_tensor = (
            torch.frombuffer(rows, dtype=torch.uint8).reshape(len(ids), self.row_bytes)
            if rows
            else torch.empty((0, self.row_bytes), dtype=torch.uint8)
        )
        counter_tensor = (
            torch.frombuffer(counters, dtype=torch.uint8).reshape(len(ids), 8)
            if counters
            else torch.empty((0, 8), dtype=torch.uint8)
        )
        ids_bytes = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        metadata = {
            "format": FORMAT,
            "formatVersion": FORMAT_VERSION,
            "dimensions": self.dimensions,
            "seed": self.seed,
            "zeroDeadband": self.zero_deadband,
            "rowCount": len(ids),
            "ids": ids,
            "idsSha256": hashlib.sha256(ids_bytes).hexdigest(),
            "packedSha256": hashlib.sha256(rows).hexdigest(),
            "countersSha256": hashlib.sha256(counters).hexdigest(),
            "connectionStateSha256": hashlib.sha256(connection_state).hexdigest(),
            "parallelOffsetsSha256": hashlib.sha256(offsets).hexdigest(),
            "parallelRecordsSha256": hashlib.sha256(parallel).hexdigest(),
        }
        return metadata, {
            prefix + "packed_rows": packed_tensor,
            prefix + "update_counters_le": counter_tensor,
            prefix + "connection_state": torch.frombuffer(connection_state, dtype=torch.uint8).reshape(len(ids), self.dimensions) if connection_state else torch.empty((0, self.dimensions), dtype=torch.uint8),
            prefix + "parallel_offsets_le": torch.frombuffer(offsets, dtype=torch.uint8).reshape(len(ids) + 1, 8),
            prefix + "parallel_records": torch.frombuffer(parallel, dtype=torch.uint8).reshape(-1, 10) if parallel else torch.empty((0, 10), dtype=torch.uint8),
        }

    def update_packed(self, other: "PackedTernaryVectors") -> None:
        """Append validated packed rows without decode or counter reset.

        This is intentionally stricter than ordinary mapping ``update``: a
        duplicate ID across persisted shards is corruption, not last-wins.
        """

        if not isinstance(other, PackedTernaryVectors) or (
            self.dimensions != other.dimensions
            or self.seed != other.seed
            or self.zero_deadband != other.zero_deadband
        ):
            raise ValueError("packed vector stores have incompatible parameters")
        if any(identifier in self._locations for identifier in other):
            raise ValueError("packed vector shards contain duplicate IDs")
        for identifier in other:
            _validate_packed_row(other.packed_row(identifier), self.dimensions)
            if other.update_count(identifier) > _MAX_UPDATES:
                raise ValueError("packed vector update counter is invalid")
        for identifier in other:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                slot = len(self._rows) // self.row_bytes
                self._rows.extend(bytes([0x55]) * self.row_bytes)
                self._update_counters.extend(b"\x00" * 8)
                self._connection_state.extend(bytes(self.dimensions))
            self._locations[identifier] = slot
            self._row(slot)[:] = other.packed_row(identifier)
            self._set_counter(slot, other.update_count(identifier))
            self.set_connection_state(identifier, other.connection_state(identifier), other.parallel_records(identifier))

    @classmethod
    def from_state(
        cls,
        metadata: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
        *,
        prefix: str = "",
    ) -> "PackedTernaryVectors":
        """Reject malformed, altered, noncanonical, or shape-mismatched state."""

        expected_fields = {
            "format", "formatVersion", "dimensions", "seed", "zeroDeadband",
            "rowCount", "ids", "idsSha256", "packedSha256", "countersSha256",
        }
        version = metadata.get("formatVersion") if isinstance(metadata, Mapping) else None
        cls.tensor_names(metadata)
        if version == 2:
            expected_fields |= {"connectionStateSha256", "parallelOffsetsSha256", "parallelRecordsSha256"}
        if not isinstance(metadata, Mapping) or set(metadata) != expected_fields:
            raise ValueError("packed vector metadata fields are invalid")
        if (
            metadata["format"] != FORMAT
            or type(metadata["formatVersion"]) is not int
            or metadata["formatVersion"] not in (1, FORMAT_VERSION)
        ):
            raise ValueError("packed vector format is incompatible")
        result = cls(
            _valid_dimension(metadata["dimensions"]),
            seed=_valid_seed(metadata["seed"]),
            zero_deadband=_valid_deadband(metadata["zeroDeadband"]),
        )
        ids = metadata["ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
            or isinstance(metadata["rowCount"], bool)
            or not isinstance(metadata["rowCount"], int)
            or metadata["rowCount"] != len(ids)
        ):
            raise ValueError("packed vector IDs or row count are invalid")
        ids_bytes = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if metadata["idsSha256"] != hashlib.sha256(ids_bytes).hexdigest():
            raise ValueError("packed vector ID checksum mismatch")
        if not isinstance(prefix, str):
            raise ValueError("packed vector tensor prefix must be a string")
        if not isinstance(tensors, Mapping):
            raise ValueError("packed vector tensors are invalid")
        packed = tensors.get(prefix + "packed_rows")
        counters = tensors.get(prefix + "update_counters_le")
        for name, tensor, shape in (
            ("packed_rows", packed, (len(ids), result.row_bytes)),
            ("update_counters_le", counters, (len(ids), 8)),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.uint8
                or tuple(tensor.shape) != shape
                or tensor.layout != torch.strided
            ):
                raise ValueError("packed vector %s tensor is invalid" % name)
        row_bytes = packed.detach().cpu().contiguous().numpy().tobytes(order="C")
        counter_bytes = counters.detach().cpu().contiguous().numpy().tobytes(order="C")
        if metadata["packedSha256"] != hashlib.sha256(row_bytes).hexdigest():
            raise ValueError("packed vector row checksum mismatch")
        if metadata["countersSha256"] != hashlib.sha256(counter_bytes).hexdigest():
            raise ValueError("packed vector counter checksum mismatch")
        for index in range(len(ids)):
            _validate_packed_row(
                memoryview(row_bytes)[
                    index * result.row_bytes : (index + 1) * result.row_bytes
                ],
                result.dimensions,
            )
            if struct.unpack_from("<Q", counter_bytes, index * 8)[0] > _MAX_UPDATES:
                raise ValueError("packed vector update counter is invalid")
        result._locations = {identifier: index for index, identifier in enumerate(ids)}
        result._rows = bytearray(row_bytes)
        result._update_counters = bytearray(counter_bytes)
        result._connection_state = bytearray(len(ids) * result.dimensions)
        result.source_format_version = version
        if version == 2:
            companions = {}
            for name, shape, digest_name in (
                ("connection_state", (len(ids), result.dimensions), "connectionStateSha256"),
                ("parallel_offsets_le", (len(ids) + 1, 8), "parallelOffsetsSha256"),
                ("parallel_records", None, "parallelRecordsSha256"),
            ):
                tensor = tensors.get(prefix + name)
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.layout != torch.strided or (shape is not None and tuple(tensor.shape) != shape) or (shape is None and (tensor.ndim != 2 or tensor.shape[1] != 10)):
                    raise ValueError("weighted vector %s tensor is invalid" % name)
                raw = tensor.detach().cpu().contiguous().numpy().tobytes()
                if hashlib.sha256(raw).hexdigest() != metadata[digest_name]:
                    raise ValueError("weighted vector %s checksum mismatch" % name)
                companions[name] = raw
            boundaries = [value[0] for value in struct.iter_unpack("<Q", companions["parallel_offsets_le"])]
            if boundaries[0] != 0 or boundaries[-1] * 10 != len(companions["parallel_records"]) or any(a > b for a, b in zip(boundaries, boundaries[1:])):
                raise ValueError("weighted vector repeated-contact offsets are invalid")
            from .weighted_vsa_row import validate_contacts
            result._connection_state[:] = companions["connection_state"]
            for index, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
                records = companions["parallel_records"][first * 10:last * 10]
                validate_contacts(records, result.dimensions)
                if records:
                    result._parallel_records[index] = records
                    result._parallel_record_bytes += len(records)
        return result


class PackedTernaryVectorView(MutableMapping[str, torch.Tensor]):
    """Named subset of one authoritative packed store, without copied rows.

    OmniCortex assemblies are also neurons. Their inspector mapping can use
    this view while both names read and mutate the very same packed row.
    Deleting a view member only unlinks the view; the neuron row remains.
    """

    def __init__(self, backing: MutableMapping[str, torch.Tensor]) -> None:
        required = ("dimensions", "levels", "packed_row", "adapt", "update_count")
        if (
            not isinstance(backing, MutableMapping)
            or getattr(backing, "packed_authoritative", False) is not True
            or any(not hasattr(backing, name) for name in required)
        ):
            raise TypeError("packed vector view needs an authoritative packed backing store")
        self.backing = backing
        self._members: dict[str, None] = {}

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __contains__(self, key: object) -> bool:
        return key in self._members and key in self.backing

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._members:
            raise KeyError(key)
        return self.backing[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.backing[key] = value
        self._members[key] = None

    def __delitem__(self, key: str) -> None:
        del self._members[key]

    def link(self, key: str) -> None:
        """Link an already-loaded neuron row to its assembly view."""

        identifier = _valid_identifier(key)
        if identifier not in self.backing:
            raise ValueError("assembly vector has no authoritative neuron row")
        if identifier in self._members:
            raise ValueError("assembly vector ID is duplicated")
        self._members[identifier] = None

    def adapt(self, key: str, target: torch.Tensor, rate: float) -> torch.Tensor:
        if key not in self._members:
            raise KeyError(key)
        return self.backing.adapt(key, target, rate)
