"""Leased, bounded text representations; these limits never discard input."""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


INLINE_TEXT_BYTES = 64 * 1024
TEXT_BLOCK_CHARS = 16 * 1024
LEARNING_WINDOW_CHARS = 4_000
READING_HIERARCHY_FANOUT = 128
_ADMISSION: ContextVar[Optional[Callable[..., Any]]] = ContextVar("dataset_admission", default=None)


class DatasetResourcePause(RuntimeError):
    """A physical allocation was not admitted; no input was rejected."""

    def __init__(self, message: str, status: Optional[dict] = None):
        super().__init__(message)
        self.status = {"recoverable": True, "sourceRecordsSkipped": False, **(status or {})}


@contextmanager
def parser_admission(callback):
    if callback is None:
        yield
        return
    token = _ADMISSION.set(callback)
    try:
        yield
    finally:
        _ADMISSION.reset(token)


def require_parser_resources(stage: str, *, ram_bytes: int = 0, disk_bytes: int = 0) -> None:
    callback = _ADMISSION.get()
    if callback is not None:
        callback(stage, max(0, int(ram_bytes)), max(0, int(disk_bytes)))
        return
    if ram_bytes:
        try:
            import psutil
            memory = psutil.virtual_memory()
        except ImportError as error:
            raise DatasetResourcePause("%s needs a measured native decoder RAM admission" % stage) from error
        reserve = max(128 * 1024 * 1024, int(memory.total * 0.1))
        if ram_bytes > max(0, int(memory.available) - reserve):
            raise DatasetResourcePause("%s exceeds available decoder RAM" % stage, {
                "availableMemoryBytes": int(memory.available), "estimatedRamBytes": ram_bytes,
                "nativeDecoderAllocationBound": "conservative-metadata-estimate-not-a-hard-allocation-cap",
            })


def bounded_json_sha256(value):
    """The canonical JSON digest without a giant escaped/UTF-8 mirror."""
    digest = hashlib.sha256()
    def emit(item):
        if isinstance(item, str):
            yield '"'
            for offset in range(0, len(item), TEXT_BLOCK_CHARS):
                yield json.dumps(item[offset:offset + TEXT_BLOCK_CHARS], ensure_ascii=False)[1:-1]
            yield '"'
        elif isinstance(item, Mapping):
            require_parser_resources("canonical provenance key ordering", ram_bytes=len(item) * 128)
            yield "{"
            for index, key in enumerate(sorted(item)):
                if index:
                    yield ","
                if isinstance(key, str):
                    key_text = key
                elif key is None:
                    key_text = "null"
                elif isinstance(key, (int, float, bool)):
                    key_text = json.dumps(key, allow_nan=False)
                else:
                    raise TypeError("canonical provenance has an unsupported JSON key")
                yield from emit(key_text)
                yield ":"
                yield from emit(item[key])
            yield "}"
        elif isinstance(item, (list, tuple)):
            yield "["
            for index, child in enumerate(item):
                if index:
                    yield ","
                yield from emit(child)
            yield "]"
        else:
            yield json.dumps(item, ensure_ascii=False, allow_nan=False, default=str)
    for piece in emit(value):
        digest.update(piece.encode("utf-8"))
    return digest.hexdigest()


@dataclass
class TextPayload:
    text: str = ""
    path: str = ""
    offset: int = 0
    bytes: int = 0
    sha256: str = ""
    dialogue: Any = None

    def windows(self, *, start_bytes: int = 0, max_chars: int = LEARNING_WINDOW_CHARS):
        if not 0 <= start_bytes <= self.bytes or max_chars < 1:
            raise ValueError("text spool window cursor is invalid")
        if not self.path:
            # Inline payloads are already bounded by the representation switch.
            encoded = self.text.encode("utf-8")
            selected = encoded[start_bytes:].decode("utf-8")
            position = start_bytes
            for offset in range(0, len(selected), max_chars):
                piece = selected[offset:offset + max_chars]
                position += len(piece.encode("utf-8"))
                yield piece, position
            return
        position = start_bytes
        with open(self.path, "rb") as binary:
            binary.seek(self.offset + start_bytes)
            # Never let read-ahead expose text outside the normalized payload.
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            pending = ""
            remaining = self.bytes - start_bytes
            while remaining:
                block = binary.read(min(TEXT_BLOCK_CHARS, remaining))
                if not block:
                    raise RuntimeError("leased text spool ended before its declared byte count")
                remaining -= len(block)
                pending += decoder.decode(block, final=remaining == 0)
                while len(pending) >= max_chars:
                    piece, pending = pending[:max_chars], pending[max_chars:]
                    position += len(piece.encode("utf-8"))
                    yield piece, position
            if pending:
                position += len(pending.encode("utf-8"))
                yield pending, position
        if position != self.bytes:
            raise RuntimeError("text window traversal did not exhaust its declared bytes")

    def close(self):
        if self.dialogue is not None:
            self.dialogue.close()
            self.dialogue = None
        if self.path:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class SpoolLearningWindow:
    """Bounded source text plus one non-labelled preceding byte of context."""
    text: str
    context_byte: Optional[int]
    first: bool
    final: bool

    def token_windows(self, tokenizer, max_length):
        if max_length < 2:
            raise ValueError("a learning window needs at least two labelled sequence positions")
        payload = self.text.encode("utf-8")
        context = tokenizer.bos_id if self.first else (
            self.context_byte + tokenizer.byte_offset if self.context_byte is not None else None
        )
        if context is None:
            raise ValueError("a continuation source window has no byte context")
        sequence = [context] + [value + tokenizer.byte_offset for value in payload]
        if self.final:
            sequence.append(tokenizer.eos_id)
        stride = max_length - 1
        for offset in range(0, max(0, len(sequence) - 1), stride):
            yield sequence[offset:offset + max_length]


def previous_payload_byte(payload: TextPayload, position: int):
    if not position:
        return None
    if not 0 < position <= payload.bytes:
        raise ValueError("text payload context offset is invalid")
    if payload.path:
        with open(payload.path, "rb") as source:
            source.seek(payload.offset + position - 1)
            value = source.read(1)
    else:
        value = payload.text.encode("utf-8")[position - 1:position]
    if len(value) != 1:
        raise RuntimeError("text payload has no committed preceding byte")
    return value[0]


class TextBuilder:
    """Switch from bounded inline bytes to disk without retaining a giant copy."""

    def __init__(self, *, clean=False):
        self.clean = clean
        self._inline = bytearray()
        self._stream = None
        self.path = ""
        self.size = 0
        self.first = None
        self.last = 0

    def write(self, text: str):
        for offset in range(0, len(text), TEXT_BLOCK_CHARS):
            piece = text[offset:offset + TEXT_BLOCK_CHARS]
            if self.clean is True:
                piece = piece.replace("\x00", "")
            data = piece.encode("utf-8", errors="replace")
            if self.clean:
                left = len(piece) - len(piece.lstrip())
                right = len(piece.rstrip())
                if left < len(piece):
                    if self.first is None:
                        self.first = self.size + len(piece[:left].encode("utf-8", errors="replace"))
                    self.last = self.size + len(piece[:right].encode("utf-8", errors="replace"))
            else:
                self.first = 0
                self.last = self.size + len(data)
            if self._stream is None and len(self._inline) + len(data) > INLINE_TEXT_BYTES:
                require_parser_resources("text record spool", disk_bytes=len(self._inline) + len(data))
                descriptor, self.path = tempfile.mkstemp(prefix="omni-record-", suffix=".utf8")
                self._stream = os.fdopen(descriptor, "wb")
                self._stream.write(self._inline)
                self._inline.clear()
            if self._stream is None:
                self._inline.extend(data)
            else:
                require_parser_resources("text record spool", disk_bytes=len(data))
                self._stream.write(data)
            self.size += len(data)

    def finish(self) -> TextPayload:
        start = self.first if self.first is not None else 0
        end = self.last if self.first is not None else 0
        if self._stream is None:
            value = bytes(self._inline[start:end])
            return TextPayload(text=value.decode("utf-8"), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        self._stream.close()
        self._stream = None
        digest = hashlib.sha256()
        with open(self.path, "rb") as source:
            source.seek(start)
            remaining = end - start
            while remaining:
                block = source.read(min(TEXT_BLOCK_CHARS, remaining))
                if not block:
                    raise RuntimeError("text spool was truncated while sealing")
                digest.update(block)
                remaining -= len(block)
        return TextPayload(path=self.path, offset=start, bytes=end - start, sha256=digest.hexdigest())

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self.path:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class TypedDialogueLease:
    """Literal dialogue targets, leased on disk with a bounded pair index.

    The index contains byte ranges and hashes, not another dialogue copy.
    A consumer must finish a pair before advancing the record iterator.
    """

    def __init__(self):
        descriptor, self.path = tempfile.mkstemp(prefix="omni-dialogue-", suffix=".utf8")
        self.stream = os.fdopen(descriptor, "w+b")
        descriptor, self.index_path = tempfile.mkstemp(prefix="omni-dialogue-", suffix=".sqlite3")
        os.close(descriptor)
        self.index = sqlite3.connect(self.index_path)
        self.index.execute("PRAGMA journal_mode=DELETE")
        self.index.execute("CREATE TABLE pairs (ordinal INTEGER PRIMARY KEY, human_offset INTEGER, human_bytes INTEGER, human_sha TEXT, response_offset INTEGER, response_bytes INTEGER, response_sha TEXT)")
        self.transcript = TextBuilder(clean=True)
        self.latest_human = None
        self.pair_count = 0
        self.message_count = 0
        self.excluded = {}
        self.digest = hashlib.sha256()
        self.closed = False

    def _append(self, payload):
        self.stream.seek(0, os.SEEK_END)
        start = self.stream.tell()
        for piece, _end in payload.windows(max_chars=TEXT_BLOCK_CHARS):
            data = piece.encode("utf-8")
            require_parser_resources("typed dialogue target lease", disk_bytes=len(data) + 512)
            self.stream.write(data)
        self.stream.flush()
        return start, payload.bytes, payload.sha256

    def add(self, role, payload):
        role = str(role or "invalid").strip().lower()
        if payload is None or not payload.bytes or role not in {"user", "human", "assistant", "brain"}:
            category = role if role in {"system", "developer"} else ("invalid" if payload is None or not payload.bytes else "unknown")
            self.excluded[category] = self.excluded.get(category, 0) + 1
            return
        normalized = "human" if role in {"user", "human"} else "brain"
        self.transcript.write(normalized + ": ")
        for piece, _end in payload.windows(max_chars=TEXT_BLOCK_CHARS):
            self.transcript.write(piece)
        self.transcript.write("\n")
        reference = self._append(payload)
        self.message_count += 1
        if normalized == "human":
            self.latest_human = reference
        elif self.latest_human is not None:
            require_parser_resources("typed dialogue pair index", disk_bytes=4096)
            self.index.execute("INSERT INTO pairs VALUES (?,?,?,?,?,?,?)", (self.pair_count, *self.latest_human, *reference))
            self.digest.update(json.dumps([self.pair_count, *self.latest_human[1:], *reference[1:]], separators=(",", ":")).encode("ascii"))
            self.pair_count += 1

    def finish(self):
        self.index.commit()
        self.stream.flush()
        payload = self.transcript.finish()
        payload.dialogue = self
        return payload, {
            "format": "typed-dialogue", "selectedField": "messages", "spooledRecord": True,
            "dialoguePairCount": self.pair_count, "dialogueTargetStreamSha256": self.digest.hexdigest(),
            "excludedRoles": dict(self.excluded),
        }

    @property
    def sha256(self):
        return self.digest.hexdigest()

    def pair(self, ordinal):
        row = self.index.execute("SELECT human_offset,human_bytes,human_sha,response_offset,response_bytes,response_sha FROM pairs WHERE ordinal=?", (int(ordinal),)).fetchone()
        if row is None:
            raise ValueError("typed dialogue pair cursor exceeds its literal target index")
        return (
            TextPayload(path=self.path, offset=row[0], bytes=row[1], sha256=row[2]),
            TextPayload(path=self.path, offset=row[3], bytes=row[4], sha256=row[5]),
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.transcript.close()
        self.index.close()
        self.stream.close()
        for path in (self.path, self.index_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def leased_dialogue_windows(response, tokenizer, max_length, *, start_target=0, context_id=None):
    """Every response byte and final EOS once; repeated context is unlabelled."""
    if max_length < 2 or not 0 <= start_target <= response.bytes + 1:
        raise ValueError("leased dialogue target cursor is invalid")
    byte_position = min(start_target, response.bytes)
    context = (tokenizer.brain_id if context_id is None else context_id) if byte_position == 0 else previous_payload_byte(response, byte_position) + tokenizer.byte_offset
    # Byte reads are bounded by the live token window, not the whole response.
    if response.path:
        source = open(response.path, "rb")
        source.seek(response.offset + byte_position)
    else:
        source = io.BytesIO(response.text.encode("utf-8"))
        source.seek(byte_position)
    try:
        position = start_target
        while position < response.bytes + 1:
            count = min(max_length - 1, response.bytes - byte_position)
            data = source.read(count)
            if len(data) != count:
                raise RuntimeError("typed dialogue target lease ended before its byte cursor")
            targets = [value + tokenizer.byte_offset for value in data]
            byte_position += count
            if byte_position == response.bytes and len(targets) < max_length - 1:
                targets.append(tokenizer.eos_id)
            if not targets:
                targets = [tokenizer.eos_id]
            ids = [context, *targets]
            position += len(targets)
            yield ids, position
            context = targets[-1]
    finally:
        source.close()


class ReadingWindowHierarchy:
    """All section IDs enter a bounded-fanout neural tree, never a giant list."""

    def __init__(self, fingerprint, integrate, state=None):
        self.fingerprint = fingerprint
        self.integrate = integrate
        restored = self.validate(state) if state is not None else {"leaves": 0, "levels": [], "groups": []}
        self.leaves = restored["leaves"]
        self.levels = restored["levels"]
        self.groups = restored["groups"]

    @staticmethod
    def validate(state):
        if not isinstance(state, dict) or set(state) != {"leaves", "levels", "groups"}:
            raise ValueError("reading hierarchy state is invalid")
        leaves = state["leaves"]
        levels, groups = state["levels"], state["groups"]
        if type(leaves) is not int or not 0 <= leaves <= (1 << 63) - 1:
            raise ValueError("reading hierarchy leaf count is invalid")
        if not isinstance(levels, list) or not isinstance(groups, list) or len(groups) != len(levels):
            raise ValueError("reading hierarchy levels are invalid")
        if len(levels) > max(1, leaves.bit_length() // 7 + 2):
            raise ValueError("reading hierarchy contains unsupported empty levels")
        cloned = []
        for bucket in levels:
            if not isinstance(bucket, list) or len(bucket) >= READING_HIERARCHY_FANOUT:
                raise ValueError("reading hierarchy page exceeds its bounded fanout")
            copied = []
            for entry in bucket:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("reading hierarchy entry is invalid")
                identity, weight = entry
                if not isinstance(identity, str) or not identity or len(identity) > 256 or any(char.isspace() for char in identity):
                    raise ValueError("reading hierarchy neural identity is invalid")
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight <= 0:
                    raise ValueError("reading hierarchy weight is invalid")
                copied.append([identity, float(weight)])
            cloned.append(copied)
        if any(type(count) is not int or count < 0 for count in groups):
            raise ValueError("reading hierarchy group counts are invalid")
        if sum(len(bucket) * (READING_HIERARCHY_FANOUT ** level) for level, bucket in enumerate(levels)) != leaves:
            raise ValueError("reading hierarchy omits committed section identities")
        return {"leaves": leaves, "levels": cloned, "groups": list(groups)}

    def state(self):
        return self.validate({"leaves": self.leaves, "levels": self.levels, "groups": self.groups})

    def _place(self, level, entry):
        while len(self.levels) <= level:
            self.levels.append([])
            self.groups.append(0)
        self.levels[level].append(entry)
        if len(self.levels[level]) == READING_HIERARCHY_FANOUT:
            entries = self.levels[level]
            parent = self._merge(level, entries)
            self.levels[level] = []
            self._place(level + 1, parent)

    def _merge(self, level, entries):
        if len(entries) == 1:
            return entries[0]
        group = self.groups[level]
        identity = self.integrate(
            [entry[0] for entry in entries], [entry[1] for entry in entries],
            "%s:%d:%d" % (self.fingerprint, level, group),
        )
        self.groups[level] += 1
        return [identity, sum(entry[1] for entry in entries)]

    def add(self, identity, weight):
        self.leaves += 1
        self._place(0, [str(identity), max(0.05, float(weight))])

    def finish(self):
        if not self.leaves:
            return None
        level = 0
        while level < len(self.levels):
            entries = self.levels[level]
            if entries:
                parent = self._merge(level, entries)
                self.levels[level] = []
                if level == len(self.levels) - 1:
                    return parent[0]
                self._place(level + 1, parent)
            level += 1
        return None


def validate_active_record_window(value):
    fields = {"recordOrdinal", "textSha256", "textBytes", "committedTextBytes", "completedWindows", "windowChars", "hierarchy"}
    if not isinstance(value, dict) or set(value) not in (fields, fields | {"typedTargets"}):
        raise ValueError("active record window cursor is invalid")
    for field in ("recordOrdinal", "textBytes", "committedTextBytes", "completedWindows", "windowChars"):
        if type(value[field]) is not int or not 0 <= value[field] <= (1 << 63) - 1:
            raise ValueError("active record window %s is invalid" % field)
    if value["recordOrdinal"] < 1 or value["completedWindows"] < 1 or value["windowChars"] != LEARNING_WINDOW_CHARS:
        raise ValueError("active record window boundary is invalid")
    if not 0 < value["committedTextBytes"] <= value["textBytes"]:
        raise ValueError("active record window byte coverage is invalid")
    if not (value["completedWindows"] - 1) * value["windowChars"] < value["committedTextBytes"] <= value["completedWindows"] * value["windowChars"] * 4:
        raise ValueError("active record window count cannot cover its byte offset")
    digest = value["textSha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("active record window text hash is invalid")
    hierarchy = ReadingWindowHierarchy.validate(value["hierarchy"])
    if hierarchy["leaves"] != value["completedWindows"]:
        raise ValueError("active record window hierarchy omits learned sections")
    clean = {**value, "hierarchy": hierarchy}
    if "typedTargets" in value:
        target = value["typedTargets"]
        target_fields = {"streamSha256", "totalPairs", "pairIndex", "passes", "passIndex", "committedTargetTokens", "completedWindows"}
        if not isinstance(target, dict) or set(target) != target_fields or value["committedTextBytes"] != value["textBytes"]:
            raise ValueError("typed target cursor precedes complete transcript coverage")
        if any(type(target[key]) is not int or not 0 <= target[key] <= (1 << 63) - 1 for key in target_fields - {"streamSha256"}):
            raise ValueError("typed target cursor counters are invalid")
        if target["passes"] not in {1, 2} or target["pairIndex"] > target["totalPairs"] or target["passIndex"] >= target["passes"]:
            raise ValueError("typed target cursor pair/pass is invalid")
        if target["pairIndex"] == target["totalPairs"] and (target["passIndex"] or target["committedTargetTokens"]):
            raise ValueError("exhausted typed target cursor has an active target")
        if not isinstance(target["streamSha256"], str) or len(target["streamSha256"]) != 64 or any(char not in "0123456789abcdef" for char in target["streamSha256"]):
            raise ValueError("typed target stream hash is invalid")
        clean["typedTargets"] = dict(target)
    return clean


def active_record_window_sha256(value):
    clean = validate_active_record_window(value)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def write_json_string_piece(builder: TextBuilder, piece: str) -> None:
    # json.dumps sees only a bounded piece, never an entire giant string field.
    for offset in range(0, len(piece), TEXT_BLOCK_CHARS):
        builder.write(json.dumps(piece[offset:offset + TEXT_BLOCK_CHARS], ensure_ascii=False)[1:-1])


class BoundedCharacters:
    def __init__(self, source):
        self.source = source
        self.block = ""
        self.offset = 0
        self.back = ""
        self.byte_position = 0

    def get(self):
        if self.back:
            value, self.back = self.back, ""
            self.byte_position += len(value.encode("utf-8", errors="replace"))
            return value
        if self.offset >= len(self.block):
            self.block = self.source.read(TEXT_BLOCK_CHARS)
            self.offset = 0
            if not self.block:
                return ""
        value = self.block[self.offset]
        self.offset += 1
        self.byte_position += len(value.encode("utf-8", errors="replace"))
        return value

    def unread(self, value):
        if self.back:
            raise RuntimeError("bounded JSON scanner pushback overflow")
        self.back = value
        self.byte_position -= len(value.encode("utf-8", errors="replace"))

    def nonspace(self):
        value = self.get()
        while value and value in " \t\r\n":
            value = self.get()
        return value


def capture_json_value(chars: BoundedCharacters, first: str, *, store=True) -> Optional[TextPayload]:
    """Frame one value without building a giant scalar/container in Python."""
    if not first:
        raise ValueError("JSON source ended before a value")
    class Discard:
        def write(self, _text): pass
        def finish(self): return None
        def close(self): pass
    builder = TextBuilder() if store else Discard()
    pending = [first]
    stack = [first] if first in {"{", "["} else []
    string = first == '"'
    escaped = False
    try:
        while True:
            if not stack and first == '"' and not string:
                break
            if not stack and first in {"{", "["}:
                break
            value = chars.get()
            if not value:
                if stack or string:
                    raise ValueError("unterminated JSON value")
                break
            if not string and not stack and value in {",", "]", "}"}:
                chars.unread(value)
                break
            if not string and not stack and value in " \t\r\n":
                chars.unread(value)
                break
            pending.append(value)
            if string:
                if escaped:
                    escaped = False
                elif value == "\\":
                    escaped = True
                elif value == '"':
                    string = False
            elif value == '"':
                string = True
            elif value in {"{", "["}:
                stack.append(value)
                if len(stack) % 64 == 0:
                    require_parser_resources("JSON framing stack", ram_bytes=len(stack) * 512)
            elif value in {"}", "]"}:
                if not stack or (stack[-1], value) not in {("{", "}"), ("[", "]")}:
                    raise ValueError("JSON container boundary does not match")
                stack.pop()
            if len(pending) >= TEXT_BLOCK_CHARS:
                builder.write("".join(pending))
                pending.clear()
        if pending:
            builder.write("".join(pending))
        return builder.finish()
    except BaseException:
        builder.close()
        raise


def parse_spooled_json(source, selected_fields, metadata_fields):
    """Validate one JSON value while streaming strings and containers to disk.

    This route never builds a row-sized dict/list/string. Typed dialogue owns
    literal leased targets, rather than being relabelled as generic JSON.
    """
    chars = BoundedCharacters(source)
    canonical = TextBuilder()
    candidates = {}
    messages_present = False
    messages_kind = ""
    messages_empty = False
    metadata_only = True
    root_fields = 0
    root_type = ""
    dialogue = None
    transferred_dialogue = False
    speech_fields = {"format", "audioPath", "audioSha256"}

    def string(output, capture=None, key=False):
        pending = []
        key_parts = []
        key_size = 0
        high_surrogate = None
        def flush():
            nonlocal key_size
            if not pending:
                return
            piece = "".join(pending)
            pending.clear()
            write_json_string_piece(output, piece)
            if capture is not None:
                capture.write(piece)
            if key and key_size <= 256:
                key_size += len(piece)
                if key_size <= 256:
                    key_parts.append(piece)
        output.write('"')
        while True:
            value = chars.get()
            if not value:
                raise ValueError("unterminated JSON string")
            if value == '"':
                if high_surrogate is not None:
                    raise ValueError("unpaired JSON Unicode surrogate")
                flush()
                output.write('"')
                return "".join(key_parts) if key_size <= 256 else None
            if value == "\\":
                escaped = chars.get()
                if high_surrogate is not None and escaped != "u":
                    raise ValueError("unpaired JSON Unicode surrogate")
                escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
                if escaped == "u":
                    digits = "".join(chars.get() for _ in range(4))
                    if len(digits) != 4 or re.fullmatch(r"[0-9a-fA-F]{4}", digits) is None:
                        raise ValueError("invalid JSON Unicode escape")
                    codepoint = int(digits, 16)
                    if 0xD800 <= codepoint <= 0xDBFF:
                        if high_surrogate is not None:
                            raise ValueError("unpaired JSON Unicode surrogate")
                        high_surrogate = codepoint
                        continue
                    if 0xDC00 <= codepoint <= 0xDFFF:
                        if high_surrogate is None:
                            raise ValueError("unpaired JSON Unicode surrogate")
                        value = chr(0x10000 + ((high_surrogate - 0xD800) << 10) + codepoint - 0xDC00)
                        high_surrogate = None
                    else:
                        if high_surrogate is not None:
                            raise ValueError("unpaired JSON Unicode surrogate")
                        value = chr(codepoint)
                elif escaped in escapes:
                    value = escapes[escaped]
                else:
                    raise ValueError("invalid JSON escape")
            elif ord(value) < 32:
                raise ValueError("unescaped JSON control character")
            elif high_surrogate is not None:
                raise ValueError("unpaired JSON Unicode surrogate")
            pending.append(value)
            if len(pending) >= TEXT_BLOCK_CHARS:
                flush()

    def value(first, *, root_field=None, depth=0, message=None, message_field=None):
        nonlocal messages_present, messages_kind, messages_empty, metadata_only, root_type, root_fields, dialogue
        if depth % 64 == 0:
            require_parser_resources("JSON structural stack", ram_bytes=(depth + 1) * 2048)
        if depth == 0:
            root_type = first
        if root_field == "messages":
            messages_kind = first
            messages_empty = False
            if dialogue is not None:
                dialogue.close()
            dialogue = TypedDialogueLease() if first == "[" else None
        if first != '"' and root_field in candidates:
            candidates.pop(root_field).close()
        if message is not None and message_field in message:
            message.pop(message_field).close()
        if first == '"':
            capture = TextBuilder(clean="strip" if message_field in {"content", "role"} else False) if (
                root_field in selected_fields or root_field in speech_fields or depth == 0
                or message is not None and message_field in {"role", "content"}
            ) else None
            try:
                string(canonical, capture=capture)
                if capture is not None:
                    sealed = capture.finish()
                    if message is not None and message_field in {"role", "content"}:
                        message[message_field] = sealed
                    else:
                        if root_field in candidates:
                            candidates[root_field].close()
                        candidates[root_field] = sealed
            except BaseException:
                if capture is not None:
                    capture.close()
                raise
        elif first == "{":
            canonical.write("{")
            key = chars.nonspace()
            if key == "}":
                canonical.write("}")
                return
            while True:
                if key != '"':
                    raise ValueError("JSON object key is not a string")
                name = string(canonical, key=True)
                if chars.nonspace() != ":":
                    raise ValueError("JSON object is missing its colon")
                canonical.write(":")
                if depth == 0:
                    root_fields += 1
                    metadata_only = metadata_only and name is not None and name.strip().lower() in metadata_fields
                    messages_present = messages_present or name == "messages"
                value(chars.nonspace(), root_field=name if depth == 0 else None, depth=depth + 1,
                      message=message if depth == 2 else None,
                      message_field=name if message is not None and depth == 2 else None)
                following = chars.nonspace()
                if following == "}":
                    canonical.write("}")
                    break
                if following != ",":
                    raise ValueError("JSON object is missing a comma or end")
                canonical.write(",")
                key = chars.nonspace()
        elif first == "[":
            canonical.write("[")
            next_value = chars.nonspace()
            if next_value == "]":
                if root_field == "messages":
                    messages_empty = True
                canonical.write("]")
                return
            while True:
                if root_field == "messages":
                    captured_message = {} if next_value == "{" else None
                    try:
                        value(next_value, depth=depth + 1, message=captured_message)
                        role_payload = captured_message.get("role") if captured_message is not None else None
                        if role_payload is not None and role_payload.bytes <= 16:
                            role = "".join(piece for piece, _ in role_payload.windows())
                        else:
                            role = "unknown" if role_payload is not None else "invalid"
                        dialogue.add(role, captured_message.get("content") if captured_message is not None else None)
                    finally:
                        for captured in (captured_message or {}).values():
                            captured.close()
                else:
                    value(next_value, depth=depth + 1)
                following = chars.nonspace()
                if following == "]":
                    canonical.write("]")
                    break
                if following != ",":
                    raise ValueError("JSON array is missing a comma or end")
                canonical.write(",")
                next_value = chars.nonspace()
        elif first in {"t", "f", "n"}:
            token = {"t": "true", "f": "false", "n": "null"}[first]
            if "".join(chars.get() for _ in range(len(token) - 1)) != token[1:]:
                raise ValueError("invalid JSON literal")
            canonical.write(token)
        elif first and first in "-0123456789":
            # A numeric scalar is streamed too. Finite-state validation avoids
            # a giant integer/Decimal/list-of-digits allocation.
            state = "minus" if first == "-" else ("zero" if first == "0" else "integer")
            canonical.write(first)
            while True:
                next_char = chars.get()
                if next_char and next_char.isascii() and next_char.isdigit():
                    if state == "zero":
                        raise ValueError("JSON number contains a leading zero")
                    state = ("zero" if next_char == "0" else "integer") if state == "minus" else ("fraction" if state in {"dot", "fraction"} else ("exponent" if state in {"exponent-sign", "exponent-start", "exponent"} else "integer"))
                elif next_char == "." and state in {"zero", "integer"}:
                    state = "dot"
                elif next_char in {"e", "E"} and state in {"zero", "integer", "fraction"}:
                    state = "exponent-start"
                elif next_char in {"+", "-"} and state == "exponent-start":
                    state = "exponent-sign"
                else:
                    chars.unread(next_char)
                    break
                canonical.write(next_char)
            if state in {"minus", "dot", "exponent-start", "exponent-sign"}:
                raise ValueError("incomplete JSON number")
        else:
            raise ValueError("invalid JSON value")

    # Import is local to keep the ordinary small-record route minimal.
    import re
    try:
        value(chars.nonspace())
        if chars.nonspace():
            raise ValueError("trailing JSON content")
        fallback = canonical.finish()
        format_payload = candidates.get("format")
        speech_pair = format_payload is not None and format_payload.bytes == len("omni-speech-pair-1") and "".join(piece for piece, _ in format_payload.windows()) == "omni-speech-pair-1"
        if messages_present and not speech_pair:
            if messages_kind != "[" or messages_empty:
                fallback.close()
                return None, {"selectedField": "messages", "invalidMessages": True}, "dialogue row has no trainable human or brain messages"
            payload, provenance = dialogue.finish()
            fallback.close()
            if not payload.bytes:
                payload.close()
                return None, provenance, "dialogue row has no trainable human or brain messages"
            transferred_dialogue = True
            return payload, provenance, None
        selected = None
        field = None
        if root_type == '"':
            selected = candidates.pop(None)
        for name in ((None,) if root_type == '"' else (("text",) if speech_pair else selected_fields)):
            if name in candidates and candidates[name].bytes:
                selected, field = candidates.pop(name), name
                break
        if selected is not None:
            fallback.close()
        elif root_type == "{" and root_fields and metadata_only:
            fallback.close()
            return None, {"selectedField": "metadata-only"}, "metadata-only row has no trainable content"
        else:
            selected = fallback
        provenance = {"selectedField": field or "structured-row", "spooledRecord": True}
        if speech_pair:
            descriptor = {"format": "omni-speech-pair-1"}
            for key in ("audioPath", "audioSha256"):
                item = candidates.get(key)
                if item is not None:
                    require_parser_resources("speech pair descriptor", ram_bytes=item.bytes * 8)
                    descriptor[key] = "".join(piece for piece, _ in item.windows())
            provenance["speechPair"] = descriptor
        if "speechPair" not in provenance:
            normalized = TextBuilder(clean=True)
            try:
                for piece, _end in selected.windows(max_chars=TEXT_BLOCK_CHARS):
                    normalized.write(piece)
                original, selected = selected, normalized.finish()
                original.close()
            except BaseException:
                normalized.close()
                selected.close()
                raise
        return selected, provenance, None
    except RecursionError as error:
        canonical.close()
        raise DatasetResourcePause("JSON nesting exceeds the bounded parser stack; record remains uncommitted") from error
    except BaseException:
        canonical.close()
        raise
    finally:
        for candidate in candidates.values():
            candidate.close()
        if dialogue is not None and not transferred_dialogue:
            dialogue.close()
