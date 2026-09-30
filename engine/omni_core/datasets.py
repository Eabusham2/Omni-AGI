"""Streaming dataset readers used by OmniCortex ingestion.

The readers deliberately expose records instead of loading a complete corpus
into memory.  Optional columnar formats use PyArrow, while archives, SQLite,
CSV, JSONL, Hugging Face-style manifests, and ordinary text use the standard
library.
"""

from __future__ import annotations

import bz2
import codecs
import csv
import errno
import gzip
import hashlib
import io
import ipaddress
import json
import lzma
import os
import re
import socket
import sqlite3
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from collections.abc import Sequence as SequenceView
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)
from xml.parsers import expat
from .text_spool import (
    BoundedCharacters,
    DatasetResourcePause,
    INLINE_TEXT_BYTES,
    TEXT_BLOCK_CHARS,
    TextBuilder,
    TextPayload,
    TypedDialogueLease,
    capture_json_value,
    parse_spooled_json,
    parser_admission,
    require_parser_resources,
    write_json_string_piece,
)
from .columnar_admission import (
    admit_parquet_footer, admit_parquet_row_group, ipc_allocation_frames,
)


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".mdx",
    ".rst",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".sql",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".yaml",
    ".yml",
    ".toml",
}
ARCHIVE_EXTENSIONS = {
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".epub",
    ".docx",
    ".pptx",
    ".xlsx",
    ".odt",
    ".ods",
    ".odp",
}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp"}
COLUMNAR_EXTENSIONS = {".parquet", ".arrow", ".feather", ".ipc"}
MANIFEST_SUFFIXES = {
    ".hf.json",
    ".dataset.json",
    ".manifest.json",
    "dataset_info.json",
    "dataset_infos.json",
}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
    ".jp2",
    ".j2k",
    ".jpf",
    ".jpx",
    ".jxl",
    ".raw",
    ".dng",
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
    ".caf",
    ".alac",
    ".amr",
    ".au",
    ".snd",
    ".mka",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".flv",
    ".3gp",
    ".3g2",
    ".ogv",
    ".mts",
    ".m2ts",
    ".vob",
}
_INTERNAL_DATASET_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "__pycache__",
    "node_modules",
}
_AUXILIARY_DATASET_FILES = {
    ".ds_store",
    ".gitattributes",
    ".gitignore",
    ".gitmodules",
    "desktop.ini",
    "thumbs.db",
}
_AUXILIARY_DATASET_SUFFIXES = (
    ".lock",
    ".metadata",
    ".incomplete",
    ".inprogress",
    ".pending",
    ".partial",
    ".part",
    ".crdownload",
    ".download",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
)
_MANIFEST_REFERENCE_KEYS = {
    "url",
    "uri",
    "href",
    "path",
    "file",
    "filename",
}
_MANIFEST_CONTAINER_KEYS = {
    "data_files",
    "files",
    "paths",
    "shards",
    "splits",
}
_MANIFEST_METADATA_KEYS = {
    "sha256",
    "checksum",
    "hash",
    "license",
    "license_url",
    "source",
    "source_url",
    "revision",
    "split",
}

# Common content columns used by Hugging Face, web-corpus, source-code and
# instruction datasets. Columnar/JSON readers select these values instead of
# teaching the model a serialization of unrelated row metadata.
_TRAINING_TEXT_FIELDS = (
    "text",
    "content",
    "document",
    "body",
    "article",
    "story",
    "code",
    "completion",
    "response",
    "answer",
)
_METADATA_ONLY_FIELDS = frozenset(
    {
        "blob_id",
        "repo_name",
        "repository",
        "path",
        "file_path",
        "filename",
        "length",
        "length_bytes",
        "score",
        "int_score",
        "id",
        "url",
        "dump",
        "language",
        "language_score",
        "token_count",
        "prompt_id",
        "sha256",
        "license",
        "split",
    }
)


@dataclass
class DatasetRecord:
    text: str
    name: str
    bytes_read: int
    kind: str = "text"
    # Binary archive members are spooled rather than retained in RAM. The
    # path remains valid only until the record iterator advances or closes.
    local_path: Optional[str] = None
    content_sha256: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    # This is a lease, not retained source text. It lasts until the record
    # iterator advances/closes, and its digest—not its path—binds checkpoints.
    text_payload: Optional[TextPayload] = None


@dataclass(frozen=True)
class _ManifestShard:
    reference: str
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RemoteDownload:
    path: Path
    requested_url: str
    final_url: str
    sha256: str
    bytes_read: int
    content_type: str


@dataclass(frozen=True)
class SQLiteSnapshot:
    """A leased, immutable SQLite backup and the bytes that identify it."""

    path: Path
    sha256: str


class ColumnarTextValue:
    """A native StringScalar buffer view; never an as_py giant string copy."""

    def __init__(self, scalar):
        # Apache Arrow StringScalar.as_buffer() returns a view, not a copy.
        self.buffer = memoryview(scalar.as_buffer())

    def chunks(self):
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        for offset in range(0, len(self.buffer), TEXT_BLOCK_CHARS):
            end = min(len(self.buffer), offset + TEXT_BLOCK_CHARS)
            yield decoder.decode(self.buffer[offset:end], final=end == len(self.buffer))

    def usable(self):
        return any(piece and not piece.isspace() for piece in self.chunks())

    def metadata(self):
        digest = hashlib.sha256()
        for offset in range(0, len(self.buffer), TEXT_BLOCK_CHARS):
            digest.update(self.buffer[offset:offset + TEXT_BLOCK_CHARS])
        return {"sha256": digest.hexdigest(), "bytes": len(self.buffer)}


class ColumnarSequence(SequenceView):
    """A lazy Arrow list/map view, not a row-sized Python list."""
    def __init__(self, scalar, arrow):
        self.values, self.arrow = scalar.values, arrow
        self.is_map = arrow.types.is_map(scalar.type)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("columnar sequence slices must be traversed, not copied")
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        value = self.values[index]
        if self.is_map:
            return (_columnar_value(value[0], self.arrow), _columnar_value(value[1], self.arrow))
        return _columnar_value(value, self.arrow)


class ColumnarMapping(Mapping):
    """Retain native struct children until each child is actually visited."""
    def __init__(self, scalar, arrow):
        self.scalar, self.arrow = scalar, arrow

    def __iter__(self):
        return iter(self.scalar)

    def __len__(self):
        return self.scalar.type.num_fields

    def __getitem__(self, key):
        return _columnar_value(self.scalar[key], self.arrow)


class ColumnarBinaryValue:
    """Bounded original byte-literal encoding, without a giant bytes copy."""
    def __init__(self, scalar):
        self.buffer = memoryview(scalar.as_buffer()).cast("B")

    def literal_chunks(self):
        # Match bytes repr's quotation choice without materializing bytes.
        has_single, has_double = False, False
        for value in self.buffer:
            has_single |= value == 39
            has_double |= value == 34
            if has_single and has_double: break
        quote = '"' if has_single and not has_double else "'"
        yield "b" + quote
        pending = []
        for value in self.buffer:
            if value == ord(quote) or value == 92: piece = "\\" + chr(value)
            elif value == 9: piece = "\\t"
            elif value == 10: piece = "\\n"
            elif value == 13: piece = "\\r"
            elif 32 <= value < 127: piece = chr(value)
            else: piece = "\\x%02x" % value
            pending.append(piece)
            if len(pending) >= TEXT_BLOCK_CHARS:
                yield "".join(pending)
                pending = []
        if pending: yield "".join(pending)
        yield quote


def _columnar_value(scalar, arrow):
    if not scalar.is_valid: return None
    kind = scalar.type
    if arrow.types.is_dictionary(kind) or arrow.types.is_union(kind):
        return _columnar_value(scalar.value, arrow)
    if arrow.types.is_struct(kind): return ColumnarMapping(scalar, arrow)
    if any(test(kind) for test in (arrow.types.is_list, arrow.types.is_large_list,
                                   arrow.types.is_fixed_size_list, arrow.types.is_map)):
        return ColumnarSequence(scalar, arrow)
    if arrow.types.is_string(kind) or arrow.types.is_large_string(kind) or \
            (hasattr(arrow.types, "is_string_view") and arrow.types.is_string_view(kind)):
        viewed = ColumnarTextValue(scalar)
        return viewed if len(viewed.buffer) > INLINE_TEXT_BYTES else scalar.as_py()
    if arrow.types.is_binary(kind) or arrow.types.is_large_binary(kind) or arrow.types.is_fixed_size_binary(kind):
        return ColumnarBinaryValue(scalar)
    # Numeric/time/null leaves are small native scalars. Unknown extension
    # types still need explicit admission; do not pretend all decoders are bounded.
    require_parser_resources("columnar scalar leaf conversion", ram_bytes=512)
    return scalar.as_py()


@dataclass
class DatasetCoverage:
    discovered_files: int = 0
    completed_files: int = 0
    rejected_files: int = 0
    discovered_records: int = 0
    processed_records: int = 0
    rejected_records: int = 0
    processed_bytes: int = 0
    shards: int = 0
    modality_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    errors_truncated: bool = False
    # A parser that fails after yielding a prefix has not proved traversal of
    # its unread tail. File/record rejection counts alone cannot certify it.
    traversal_incomplete: bool = False

    # Diagnostics are samples, not a traversal limit. Keeping counts separate
    # prevents a malformed multi-million-row shard from turning its coverage
    # report into another multi-gigabyte in-memory dataset.
    _ERROR_SAMPLE_LIMIT = 256

    def _record_error(
        self,
        source: str,
        message: str,
        *,
        count: int = 1,
    ) -> None:
        occurrences = max(0, int(count))
        if occurrences == 0:
            return
        self.error_count += occurrences
        if len(self.errors) >= self._ERROR_SAMPLE_LIMIT:
            self.errors_truncated = True
            return
        entry: Dict[str, Any] = {"source": source, "message": message}
        if occurrences != 1:
            entry["count"] = occurrences
        self.errors.append(entry)

    def reject(
        self,
        source: str,
        message: str,
        *,
        already_discovered: bool = False,
    ) -> None:
        """Account for one visited record that could not be processed.

        Parsers call this directly when a malformed row/member is discovered.
        Callers that already admitted the record (for example ``_record`` or
        the binary-media iterator) opt out of incrementing discovery again.
        Keeping the three counters mutually exhaustive lets every downstream
        coverage report prove traversal instead of asserting completion.
        """

        if not already_discovered:
            self.discovered_records += 1
        self.rejected_records += 1
        self._record_error(source, message)

    def reject_many(
        self,
        source: str,
        message: str,
        count: int,
        *,
        processed_bytes: int = 0,
    ) -> None:
        """Classify a schema-proven invalid row set without retaining each row.

        This is used only when the columnar schema proves that no row can
        contain trainable content. It does not skip a potentially valid record.
        """

        occurrences = max(0, int(count))
        self.discovered_records += occurrences
        self.rejected_records += occurrences
        self.processed_bytes += max(0, int(processed_bytes))
        self._record_error(source, message, count=occurrences)

    def reject_processed(self, source: str, message: str) -> None:
        """Reclassify a provisionally processed record as explicitly rejected."""

        if self.processed_records <= 0:
            raise RuntimeError("cannot reject a record that was not processed")
        self.processed_records -= 1
        self.reject(source, message, already_discovered=True)

    def reject_file(
        self,
        source: str,
        message: str,
        *,
        already_discovered: bool = False,
        record_already_discovered: bool = False,
    ) -> None:
        """Classify one discovered file/member and its failed record.

        File-level parser failures still represent a visited input record. The
        file and record counters therefore advance together while sharing one
        diagnostic entry.
        """

        if not already_discovered:
            self.discovered_files += 1
        self.rejected_files += 1
        self.reject(
            source,
            message,
            already_discovered=record_already_discovered,
        )

    def as_dict(self) -> Dict[str, Any]:
        processed_files = max(0, int(self.completed_files))
        discovered_files = max(0, int(self.discovered_files))
        rejected_files = max(0, int(self.rejected_files))
        files_complete = discovered_files == (processed_files + rejected_files)
        records_complete = self.discovered_records == (
            self.processed_records + self.rejected_records
        )
        return {
            "discoveredFiles": self.discovered_files,
            "completedFiles": self.completed_files,
            "processedFiles": processed_files,
            "rejectedFiles": rejected_files,
            "discoveredRecords": self.discovered_records,
            "processedRecords": self.processed_records,
            "rejectedRecords": self.rejected_records,
            "processedBytes": self.processed_bytes,
            "shards": self.shards,
            "modalityCounts": dict(self.modality_counts),
            "errors": list(self.errors),
            "errorCount": self.error_count,
            "errorsTruncated": self.errors_truncated,
            "complete": (
                files_complete and records_complete and not self.traversal_incomplete
            ),
        }


class DatasetTraversalIncomplete(RuntimeError):
    """A readable source stopped before all of its records were visited."""


def _parser_resource_failure(error: Exception) -> bool:
    return isinstance(error, (MemoryError, RecursionError)) or (
        isinstance(error, OSError)
        and error.errno in {
            errno.ENOMEM,
            errno.ENOSPC,
            getattr(errno, "EDQUOT", errno.ENOSPC),
        }
    )


def _media_kind(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def _auxiliary_dataset_file_rejection(path: Path) -> Optional[str]:
    """Classify filesystem/download bookkeeping before content sniffing.

    A UTF-8 lock or downloader metadata file must never become a text training
    record merely because it happens to decode cleanly. Explicitly classifying
    it also keeps coverage exhaustive: the file is visited and rejected rather
    than silently presented to neural learning.
    """

    name = path.name.lower()
    if (
        name.startswith(".")
        or name in _AUXILIARY_DATASET_FILES
        or name.endswith("~")
        or any(name.endswith(suffix) for suffix in _AUXILIARY_DATASET_SUFFIXES)
    ):
        return "transient, operating-system, or repository metadata is not training data"
    return None


def _binary_record(
    *,
    name: str,
    local_path: Path,
    bytes_read: int,
    content_sha256: str,
    kind: str,
    coverage: DatasetCoverage,
    provenance: Optional[Dict[str, Any]] = None,
) -> DatasetRecord:
    coverage.discovered_records += 1
    coverage.processed_records += 1
    coverage.processed_bytes += max(0, int(bytes_read))
    coverage.modality_counts[kind] = coverage.modality_counts.get(kind, 0) + 1
    return DatasetRecord(
        text="",
        name=name,
        bytes_read=max(0, int(bytes_read)),
        kind=kind,
        local_path=str(local_path),
        content_sha256=content_sha256,
        provenance=dict(provenance or {}),
    )


def _normalize_hf_url(raw_url: str) -> str:
    """Translate a declarative ``hf://`` dataset reference into HTTPS.

    Supported form:
    ``hf://datasets/<owner>/<repo>[@revision]/<path>``. A missing revision
    resolves through Hugging Face's ``main`` branch. This only resolves data
    URLs; no repository code or setup script is ever executed.
    """

    parsed = urllib.parse.urlsplit(raw_url)
    if parsed.scheme.lower() != "hf":
        return raw_url
    if parsed.netloc.lower() != "datasets":
        raise ValueError("hf:// references must target the datasets namespace")
    pieces = [urllib.parse.unquote(value) for value in parsed.path.split("/") if value]
    if len(pieces) < 3:
        raise ValueError(
            "hf:// dataset references require owner, repository, and shard path"
        )
    owner, repository_and_revision = pieces[0], pieces[1]
    repository, separator, revision = repository_and_revision.partition("@")
    if not owner or not repository:
        raise ValueError("hf:// dataset owner and repository cannot be empty")
    revision = revision if separator and revision else "main"
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in pieces[2:])
    return (
        "https://huggingface.co/datasets/"
        + urllib.parse.quote(owner, safe="")
        + "/"
        + urllib.parse.quote(repository, safe="")
        + "/resolve/"
        + urllib.parse.quote(revision, safe="")
        + "/"
        + quoted_path
    )


def _allow_local_remote_urls() -> bool:
    return os.environ.get("OMNI_ALLOW_LOCAL_URLS", "").strip() == "1"


def _is_loopback_url(raw_url: str) -> bool:
    hostname = (urllib.parse.urlsplit(raw_url).hostname or "").rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _validate_remote_url(raw_url: str) -> str:
    """Validate a remote data URL before every request and redirect.

    HTTPS is mandatory. Loopback HTTP is available only to local tests and
    explicitly controlled installations through ``OMNI_ALLOW_LOCAL_URLS=1``.
    DNS answers are rejected when any address is private, reserved, link-local,
    multicast, or otherwise non-global.
    """

    normalized = _normalize_hf_url(raw_url)
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.username or parsed.password:
        raise ValueError("remote dataset URLs containing credentials are not allowed")
    if parsed.scheme.lower() not in {"https", "http"}:
        raise ValueError("remote dataset shards require HTTPS")
    if not parsed.hostname:
        raise ValueError("remote dataset URL is missing a hostname")

    hostname = parsed.hostname.rstrip(".").lower()
    allow_local = _allow_local_remote_urls()
    loopback_name = hostname == "localhost"
    try:
        literal_address = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal_address = None
    literal_loopback = bool(literal_address and literal_address.is_loopback)
    if parsed.scheme.lower() != "https" and not (
        allow_local and (loopback_name or literal_loopback)
    ):
        raise ValueError("remote dataset shards require HTTPS")

    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise ValueError("remote dataset hostname could not be resolved") from error
    if not addresses:
        raise ValueError("remote dataset hostname did not resolve")
    for address in addresses:
        raw_address = str(address[4][0]).split("%", 1)[0]
        resolved = ipaddress.ip_address(raw_address)
        if allow_local and resolved.is_loopback:
            continue
        if not resolved.is_global:
            raise ValueError(
                "remote dataset URL resolves to a private or reserved network address"
            )
    return urllib.parse.urlunsplit(parsed)


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: BinaryIO,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Optional[urllib.request.Request]:
        resolved = urllib.parse.urljoin(request.full_url, new_url)
        validated = _validate_remote_url(resolved)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            validated,
        )


def _remote_suffix(url: str, content_type: str = "") -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    lower_name = name.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower_name.endswith(compound):
            return compound
    suffix = Path(name).suffix.lower()
    if suffix and len(suffix) <= 16 and suffix[1:].replace("_", "").isalnum():
        return suffix
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    return {
        "application/json": ".json",
        "application/jsonl": ".jsonl",
        "application/x-ndjson": ".jsonl",
        "application/x-parquet": ".parquet",
        "application/vnd.apache.arrow.file": ".arrow",
        "application/x-tar": ".tar",
        "application/zip": ".zip",
        "application/gzip": ".gz",
        "application/x-gzip": ".gz",
        "application/x-bzip2": ".bz2",
        "application/x-xz": ".xz",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.oasis.opendocument.text": ".odt",
        "application/vnd.oasis.opendocument.spreadsheet": ".ods",
        "application/vnd.oasis.opendocument.presentation": ".odp",
    }.get(normalized_type, ".data")


def _download_remote_shard(
    raw_url: str,
    destination_directory: Path,
    expected_sha256: str = "",
) -> _RemoteDownload:
    fetch_url = _validate_remote_url(raw_url)
    request = urllib.request.Request(
        fetch_url,
        headers={
            "Accept": "application/octet-stream, application/json, text/plain;q=0.9, */*;q=0.5",
            "User-Agent": "Omni-AGI-Studio/1.0 dataset-ingestion",
        },
        method="GET",
    )
    handlers: List[Any] = [_ValidatedRedirectHandler()]
    if _allow_local_remote_urls() and _is_loopback_url(fetch_url):
        # Explicitly allowed local dataset servers must not be intercepted by
        # a machine- or CI-level HTTP proxy. Ordinary global HTTPS downloads
        # keep urllib's normal proxy discovery.
        handlers.insert(0, urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    timeout = max(
        1.0,
        float(os.environ.get("OMNI_DATASET_HTTP_TIMEOUT_SECONDS", "30")),
    )
    destination_directory.mkdir(parents=True, exist_ok=True)
    part_path = destination_directory / "remote-shard.part"
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = _validate_remote_url(str(response.geturl()))
            content_type = str(response.headers.get("content-type", ""))
            with part_path.open("wb") as destination:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    bytes_read += len(block)
                destination.flush()
                os.fsync(destination.fileno())
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        raise RuntimeError("remote dataset shard download failed: %s" % error) from error
    actual_sha256 = digest.hexdigest()
    declared = expected_sha256.strip().lower()
    if declared:
        if len(declared) != 64 or any(value not in "0123456789abcdef" for value in declared):
            raise ValueError("manifest shard sha256 must be 64 hexadecimal characters")
        if actual_sha256 != declared:
            raise ValueError(
                "remote dataset shard checksum mismatch: expected %s, received %s"
                % (declared, actual_sha256)
            )
    final_path = destination_directory / (
        "remote-" + actual_sha256[:16] + _remote_suffix(final_url, content_type)
    )
    part_path.replace(final_path)
    return _RemoteDownload(
        path=final_path,
        requested_url=raw_url,
        final_url=final_url,
        sha256=actual_sha256,
        bytes_read=bytes_read,
        content_type=content_type,
    )


def dataset_format(path: Path, requested: str = "") -> str:
    if requested and requested not in {"unknown", "dataset"}:
        normalized = requested.lower()
        if normalized in {"markdown", "code"}:
            return "text"
        return normalized
    lower_name = path.name.lower()
    suffix = path.suffix.lower()
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tar.bz2"):
        return "archive"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in {".csv", ".tsv"}:
        return suffix[1:]
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        if any(lower_name.endswith(marker) for marker in MANIFEST_SUFFIXES):
            return "huggingface"
        return "json"
    if suffix == ".sqlite" or suffix in {".sqlite3", ".db"}:
        return "sqlite"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".arrow", ".feather", ".ipc"}:
        return "arrow"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    if suffix == ".pdf":
        return "pdf"
    if _media_kind(path) is not None:
        return str(_media_kind(path))
    return "unknown"


def dataset_record_count_hint(path: Path, requested: str = "") -> Optional[int]:
    """Return a cheap exact record total when the container exposes one.

    Streaming text/JSONL/CSV totals deliberately remain unknown: pre-scanning a
    30+ GB source just to draw a denominator would double I/O and delay actual
    learning. Parquet stores its row count in the footer, so that value is both
    exact and inexpensive.
    """

    format_name = dataset_format(path, requested)
    if format_name == "parquet":
        require_parser_resources("columnar row-count decoder import", ram_bytes=32 * 1024 * 1024)
        admit_parquet_footer(path)
        _require_pyarrow()
        import pyarrow.parquet as parquet  # type: ignore

        return max(0, int(parquet.ParquetFile(path).metadata.num_rows))
    if format_name in {"image", "audio", "video"}:
        return 1
    return None


def _record(
    text: str,
    name: str,
    bytes_read: int,
    coverage: DatasetCoverage,
    kind: str = "text",
    provenance: Optional[Mapping[str, Any]] = None,
) -> Optional[DatasetRecord]:
    if len(text) > INLINE_TEXT_BYTES:
        builder = TextBuilder(clean=True)
        try:
            builder.write(text)
            return _payload_record(builder.finish(), name, bytes_read, coverage, kind, provenance)
        except BaseException:
            builder.close()
            raise
    clean = text.replace("\x00", "").strip()
    coverage.discovered_records += 1
    coverage.processed_bytes += max(0, int(bytes_read))
    if not clean:
        coverage.reject(
            name,
            "record contained no usable text",
            already_discovered=True,
        )
        return None
    coverage.processed_records += 1
    coverage.modality_counts[kind] = coverage.modality_counts.get(kind, 0) + 1
    return DatasetRecord(
        clean,
        name,
        max(0, int(bytes_read)),
        kind,
        provenance=dict(provenance or {}),
    )


def _payload_record(payload, name, bytes_read, coverage, kind="text", provenance=None, source_directory=None):
    descriptor = (provenance or {}).get("speechPair")
    if descriptor is not None:
        if (provenance or {}).get("selectedField") != "text":
            coverage.discovered_records += 1
            coverage.processed_bytes += max(0, int(bytes_read))
            coverage.reject(name, "speech pair requires its literal text field, not a fallback or another content column", already_discovered=True)
            return None
        return _speech_pair_record(descriptor, payload, name, bytes_read, coverage, source_directory)
    coverage.discovered_records += 1
    coverage.processed_bytes += max(0, int(bytes_read))
    if not payload.bytes:
        payload.close()
        coverage.reject(name, "record contained no usable text", already_discovered=True)
        return None
    coverage.processed_records += 1
    coverage.modality_counts[kind] = coverage.modality_counts.get(kind, 0) + 1
    return DatasetRecord(payload.text, name, max(0, int(bytes_read)), kind,
                         provenance=dict(provenance or {}), text_payload=payload)


def _row_provenance(
    value: Mapping[str, Any], selected_fields: Set[str]
) -> Dict[str, Any]:
    """Retain attributable scalar metadata without duplicating row content."""

    metadata: Dict[str, Any] = {}
    for raw_key, entry in value.items():
        key = str(raw_key)
        if key in selected_fields or key == "messages":
            continue
        if entry is None or isinstance(entry, (bool, int, float)):
            metadata[key] = entry
        elif isinstance(entry, ColumnarTextValue):
            metadata[key] = entry.metadata()
        elif isinstance(entry, str):
            # Identifiers and URLs are useful provenance. Very large strings
            # are represented by a hash so ingestion remains streaming.
            if len(entry) <= 4_096 and len(entry.encode("utf-8", errors="replace")) <= 4_096:
                metadata[key] = entry
            else:
                digest = hashlib.sha256()
                count = 0
                for offset in range(0, len(entry), TEXT_BLOCK_CHARS):
                    encoded = entry[offset:offset + TEXT_BLOCK_CHARS].encode("utf-8", errors="replace")
                    digest.update(encoded)
                    count += len(encoded)
                metadata[key] = {"sha256": digest.hexdigest(), "bytes": count}
    return metadata


def _conversation_training_text(
    messages: Any,
) -> Tuple[str, Dict[str, Any]]:
    """Flatten typed dialogue while recording assistant supervision spans.

    System/persona messages are deliberately excluded from the clean starter
    corpus. Human and brain boundary labels are data-format markers, not a
    hidden runtime instruction.
    """

    if not isinstance(messages, list):
        return "", {"invalidMessages": True}
    pieces: List[str] = []
    spans: List[Dict[str, Any]] = []
    dialogue_pairs: List[Dict[str, str]] = []
    excluded_roles: Dict[str, int] = {}
    cursor = 0
    latest_human = ""
    for message in messages:
        if not isinstance(message, Mapping):
            excluded_roles["invalid"] = excluded_roles.get("invalid", 0) + 1
            continue
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            excluded_roles[role or "invalid"] = excluded_roles.get(role or "invalid", 0) + 1
            continue
        if role in {"system", "developer"}:
            excluded_roles[role] = excluded_roles.get(role, 0) + 1
            continue
        if role in {"user", "human"}:
            boundary = "human: "
            normalized_role = "human"
            latest_human = content.strip()
        elif role in {"assistant", "brain"}:
            boundary = "brain: "
            normalized_role = "brain"
        else:
            excluded_roles[role or "unknown"] = excluded_roles.get(role or "unknown", 0) + 1
            continue
        line = boundary + content.strip() + "\n"
        start = cursor + len(boundary)
        end = cursor + len(line.rstrip("\n"))
        pieces.append(line)
        if normalized_role == "brain":
            spans.append({"start": start, "end": end, "role": "brain"})
            if latest_human:
                # Kept only on the streaming DatasetRecord. Ingestion consumes
                # these typed targets and never copies them into source
                # metadata, so Synapses Only still leaves no raw dialogue on
                # disk after the record has been encoded.
                dialogue_pairs.append(
                    {"human": latest_human, "brain": content.strip()}
                )
        cursor += len(line)
    provenance: Dict[str, Any] = {
        "format": "typed-dialogue",
        "assistantSpans": spans,
        "dialoguePairs": dialogue_pairs,
    }
    if excluded_roles:
        provenance["excludedRoles"] = excluded_roles
    return "".join(pieces).strip(), provenance


def _training_value(
    value: Any,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Resolve one structured row into training content and provenance."""

    if isinstance(value, str):
        return value, {}, None
    if isinstance(value, Mapping):
        if "messages" in value:
            messages = value.get("messages")
            # A native nested scalar may already have needed a whole-scalar
            # library allocation under explicit admission. Do not add a
            # second giant joined transcript/list of literal target copies,
            # and never lose supervision just because its text is spooled.
            size_hint = 0
            if isinstance(messages, ColumnarSequence):
                size_hint = INLINE_TEXT_BYTES + 1
            elif isinstance(messages, list):
                for message in messages:
                    content = message.get("content") if isinstance(message, Mapping) else None
                    size_hint += (len(content) * 4 if isinstance(content, str) else 0) + 64
                    if size_hint > INLINE_TEXT_BYTES:
                        break
            if size_hint > INLINE_TEXT_BYTES:
                lease = TypedDialogueLease()
                transferred = False
                try:
                    for message in messages:
                        content = message.get("content") if isinstance(message, Mapping) else None
                        role = str(message.get("role", "")) if isinstance(message, Mapping) else "invalid"
                        payload = None
                        if isinstance(content, (str, ColumnarTextValue)):
                            builder = TextBuilder(clean="strip")
                            try:
                                if isinstance(content, ColumnarTextValue):
                                    for piece in content.chunks(): builder.write(piece)
                                else: builder.write(content)
                                payload = builder.finish()
                            except BaseException:
                                builder.close()
                                raise
                        try:
                            lease.add(role, payload)
                        finally:
                            if payload is not None: payload.close()
                    text, dialogue = lease.finish()
                    if not text.bytes:
                        text.close()
                        return "", dialogue, "dialogue row has no trainable human or brain messages"
                    transferred = True
                    return text, {**_row_provenance(value, {"messages"}), **dialogue}, None
                finally:
                    if not transferred: lease.close()
            text, dialogue = _conversation_training_text(value.get("messages"))
            if text:
                return (
                    text,
                    {
                        **_row_provenance(value, {"messages"}),
                        **dialogue,
                        "selectedField": "messages",
                    },
                    None,
                )
            return (
                "",
                {
                    **_row_provenance(value, {"messages"}),
                    **dialogue,
                    "selectedField": "messages",
                },
                "dialogue row has no trainable human or brain messages",
            )
        for field_name in _TRAINING_TEXT_FIELDS:
            content = value.get(field_name)
            if isinstance(content, ColumnarTextValue) and content.usable():
                return content, {
                    **_row_provenance(value, {field_name}), "selectedField": field_name,
                }, None
            if isinstance(content, str) and content and not content.isspace():
                return (
                    content,
                    {
                        **_row_provenance(value, {field_name}),
                        "selectedField": field_name,
                    },
                    None,
                )
        normalized_keys = {str(key).strip().lower() for key in value}
        if normalized_keys and normalized_keys.issubset(_METADATA_ONLY_FIELDS):
            return "", _row_provenance(value, set()), "metadata-only row has no trainable content"
    try:
        builder = TextBuilder(clean=True)
        try:
            for piece in _bounded_json_encoding(value, compact=True):
                builder.write(piece)
            return builder.finish(), {"selectedField": "structured-row"}, None
        except BaseException:
            builder.close()
            raise
    except (TypeError, ValueError) as error:
        return "", {}, "structured row is not serializable: %s" % error


def _speech_pair_record(descriptor, transcript, name, bytes_read, coverage, source_directory=None):
    """Admit one explicit, hash-bound local transcript/audio pair, never infer a transcript."""
    try:
        if not isinstance(descriptor, Mapping) or descriptor.get("format") != "omni-speech-pair-1":
            raise ValueError("unsupported speech pairing format")
        audio_reference, declared = descriptor.get("audioPath"), descriptor.get("audioSha256")
        if not isinstance(audio_reference, str) or not audio_reference or "\x00" in audio_reference or \
                not isinstance(declared, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", declared):
            raise ValueError("speech pair needs a relative audioPath and full audioSha256")
        if source_directory is None or Path(audio_reference).is_absolute():
            raise ValueError("speech pair audioPath must be relative to its local source directory")
        source_root = Path(source_directory).resolve()
        audio_path = (source_root / audio_reference).resolve()
        if not audio_path.is_relative_to(source_root) or not audio_path.is_file() or _media_kind(audio_path) != "audio":
            raise ValueError("speech pair references no supported audio file inside its source directory")
        if _sha256_path(audio_path) != declared.lower():
            raise ValueError("speech pair audio checksum mismatch")
        if isinstance(transcript, TextPayload):
            # The present neural text-to-idea library needs one utterance. This
            # allocation is measured, not silently shortened at a text cap.
            require_parser_resources("speech utterance text conditioning", ram_bytes=transcript.bytes * 64 + TEXT_BLOCK_CHARS * 8)
            text = "".join(piece for piece, _ in transcript.windows())
        elif isinstance(transcript, ColumnarTextValue):
            require_parser_resources("speech utterance text conditioning", ram_bytes=len(transcript.buffer) * 64 + TEXT_BLOCK_CHARS * 8)
            text = "".join(transcript.chunks())
        else:
            text = transcript
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError("speech pair text must be the literal nonempty recorded utterance")
        require_parser_resources("speech utterance text conditioning", ram_bytes=len(text) * 64 + TEXT_BLOCK_CHARS * 8)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        pair_sha = hashlib.sha256(bytes.fromhex(declared) + bytes.fromhex(text_sha)).hexdigest()
        return _binary_record(name=name, local_path=audio_path, bytes_read=bytes_read,
            content_sha256=pair_sha, kind="audio", coverage=coverage, provenance={
                "format": "omni-speech-pair-1", "speech_text": text,
                "speech_text_sha256": text_sha, "audio_sha256": declared.lower(),
                "audio_bytes": audio_path.stat().st_size, "selectedField": "text",
            })
    except (ValueError, OSError) as error:
        coverage.discovered_records += 1
        coverage.processed_bytes += max(0, int(bytes_read))
        coverage.reject(name, str(error), already_discovered=True)
        return None


def _structured_record(
    value: Any,
    name: str,
    bytes_read: int,
    coverage: DatasetCoverage,
    source_directory: Optional[Path] = None,
) -> Optional[DatasetRecord]:
    if isinstance(value, Mapping):
        format_value = value.get("format")
        if isinstance(format_value, ColumnarTextValue) and len(format_value.buffer) <= 64:
            format_value = "".join(format_value.chunks())
        if format_value == "omni-speech-pair-1":
            descriptor = {"format": format_value}
            for field_name in ("audioPath", "audioSha256"):
                scalar = value.get(field_name)
                if isinstance(scalar, ColumnarTextValue) and len(scalar.buffer) <= 4096:
                    scalar = "".join(scalar.chunks())
                descriptor[field_name] = scalar
            return _speech_pair_record(descriptor, value.get("text"), name, bytes_read, coverage, source_directory)
    text, provenance, rejection = _training_value(value)
    if rejection is not None:
        coverage.discovered_records += 1
        coverage.processed_bytes += max(0, int(bytes_read))
        coverage.reject(name, rejection, already_discovered=True)
        return None
    if isinstance(text, TextPayload):
        return _payload_record(text, name, bytes_read, coverage, provenance=provenance)
    if isinstance(text, ColumnarTextValue):
        builder = TextBuilder(clean=True)
        try:
            for piece in text.chunks():
                builder.write(piece)
            return _payload_record(builder.finish(), name, bytes_read, coverage, provenance=provenance)
        except BaseException:
            builder.close()
            raise
    return _record(
        text,
        name,
        bytes_read,
        coverage,
        provenance=provenance,
    )


def _iter_text_stream(
    stream: io.TextIOBase,
    name: str,
    coverage: DatasetCoverage,
    chunk_chars: int = 32_768,
) -> Iterator[DatasetRecord]:
    pending = ""
    while True:
        block = stream.read(chunk_chars)
        if not block:
            break
        pending += block
        while len(pending) >= chunk_chars:
            split_at = pending.rfind("\n", 0, chunk_chars)
            if split_at <= 0:
                split_at = chunk_chars
            piece, pending = pending[:split_at], pending[split_at:]
            record = _record(
                piece,
                name,
                len(piece.encode("utf-8", errors="replace")),
                coverage,
            )
            if record is not None:
                yield record
    if pending:
        record = _record(
            pending,
            name,
            len(pending.encode("utf-8", errors="replace")),
            coverage,
        )
        if record is not None:
            yield record


def _iter_text_path(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        yield from _iter_text_stream(stream, path.name, coverage)


def _configure_csv_field_limit() -> None:
    # csv's small default field ceiling is not a dataset policy. Use the
    # largest positive integer accepted by this runtime instead. Some Python
    # builds expose a narrower C integer than sys.maxsize; find its actual
    # boundary without replacing the default with another arbitrary cap.
    try:
        csv.field_size_limit(sys.maxsize)
        return
    except OverflowError:
        lower, upper = 0, sys.maxsize
    while lower + 1 < upper:
        candidate = (lower + upper) // 2
        try:
            csv.field_size_limit(candidate)
        except OverflowError:
            upper = candidate
        else:
            lower = candidate
    csv.field_size_limit(lower)


def _iter_delimited(
    path: Path, delimiter: str, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        chars = BoundedCharacters(stream)
        with path.open("rb") as prefix:
            bom_bytes = 3 if prefix.read(3) == b"\xef\xbb\xbf" else 0
        index = 0
        while True:
            row_start_bytes = chars.byte_position
            builder = TextBuilder()
            builder.write("[")
            in_quotes = False
            at_field_start = True
            field_open = False
            row_started = False
            pending = []
            eof = False
            def start_field():
                nonlocal field_open, row_started
                if not field_open:
                    builder.write('"')
                    field_open = True
                    row_started = True
            def flush_field():
                if pending:
                    write_json_string_piece(builder, "".join(pending))
                    pending.clear()
            try:
                while True:
                    char = chars.get()
                    if not char:
                        eof = True
                        break
                    if in_quotes:
                        if char == '"':
                            following = chars.get()
                            if following == '"':
                                pending.append('"')
                            else:
                                in_quotes = False
                                chars.unread(following)
                        else:
                            pending.append(char)
                    elif char == delimiter:
                        start_field()
                        flush_field()
                        builder.write('",')
                        field_open = False
                        at_field_start = True
                        # A trailing delimiter means a final empty field.
                        row_started = True
                    elif char in {"\r", "\n"}:
                        if char == "\r":
                            following = chars.get()
                            if following != "\n":
                                chars.unread(following)
                        break
                    else:
                        start_field()
                        if at_field_start and char == '"':
                            in_quotes = True
                        else:
                            pending.append(char)
                        at_field_start = False
                    if len(pending) >= TEXT_BLOCK_CHARS:
                        flush_field()
                if eof and not row_started:
                    builder.close()
                    return
                if row_started:
                    start_field()
                    flush_field()
                    builder.write('"')
                builder.write("]")
                payload = builder.finish()
                index += 1
                record = _payload_record(payload, "%s#row-%d" % (path.name, index),
                                         chars.byte_position - row_start_bytes + (bom_bytes if index == 1 else 0), coverage)
                if record is not None:
                    yield record
                if eof:
                    return
            except BaseException:
                builder.close()
                raise


def _iter_jsonl(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    with path.open("rb") as stream:
        index = 0
        while True:
            line = stream.readline(INLINE_TEXT_BYTES + 1)
            if not line:
                break
            index += 1
            if len(line) > INLINE_TEXT_BYTES and not line.endswith(b"\n"):
                descriptor, raw_path = tempfile.mkstemp(prefix="omni-json-row-", suffix=".json")
                raw_bytes = 0
                payload = None
                try:
                    with os.fdopen(descriptor, "wb") as raw:
                        while True:
                            require_parser_resources("JSONL record spool", disk_bytes=len(line))
                            raw.write(line)
                            raw_bytes += len(line)
                            if line.endswith(b"\n"):
                                break
                            line = stream.readline(TEXT_BLOCK_CHARS)
                            if not line:
                                break
                    with open(raw_path, "r", encoding="utf-8-sig" if index == 1 else "utf-8", errors="replace", newline="") as raw:
                        payload, provenance, rejection = parse_spooled_json(
                            raw, _TRAINING_TEXT_FIELDS, _METADATA_ONLY_FIELDS
                        )
                    if rejection:
                        coverage.processed_bytes += raw_bytes
                        coverage.reject("%s#line-%d" % (path.name, index), rejection)
                    elif payload is not None:
                        record = _payload_record(payload, "%s#line-%d" % (path.name, index),
                                                 raw_bytes, coverage, provenance=provenance, source_directory=path.parent)
                        if record is not None:
                            yield record
                except ValueError as error:
                    coverage.processed_bytes += raw_bytes
                    coverage.reject("%s#line-%d" % (path.name, index), "invalid JSONL: %s" % error)
                finally:
                    if payload is not None:
                        payload.close()
                    os.unlink(raw_path)
                continue
            physical_bytes = len(line)
            line = line.decode("utf-8-sig" if index == 1 else "utf-8", errors="replace")
            stripped = line.strip()
            if not stripped:
                coverage.processed_bytes += physical_bytes
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                coverage.processed_bytes += physical_bytes
                coverage.reject(
                    "%s#line-%d" % (path.name, index),
                    "invalid JSONL: %s" % error,
                )
                continue
            record = _structured_record(
                value,
                "%s#line-%d" % (path.name, index),
                physical_bytes,
                coverage,
                source_directory=path.parent,
            )
            if record is not None:
                yield record


def _known_root_json_record(path: Path) -> bool:
    """Recognize explicit root records without materializing/skipping values.

    Other root maps retain their historical key/value-record semantics. This
    bounded lexical pass inspects only keys and the tiny format declaration;
    it is not a record-count pre-scan or a whole-value Python conversion.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
        chars = BoundedCharacters(source)
        if chars.nonspace() != "{":
            return False
        following = chars.nonspace()
        while following and following != "}":
            if following != '"':
                return False
            key = capture_json_value(chars, following)
            try:
                name = json.loads(key.text) if not key.path else None
            finally:
                key.close()
            if chars.nonspace() != ":":
                return False
            first = chars.nonspace()
            if name == "messages" and first == "[":
                return True
            if name == "format":
                declaration = capture_json_value(chars, first)
                try:
                    if not declaration.path and json.loads(declaration.text) == "omni-speech-pair-1":
                        return True
                finally:
                    declaration.close()
            else:
                capture_json_value(chars, first, store=False)
            following = chars.nonspace()
            if following != ",":
                return False
            following = chars.nonspace()
    return False


def _iter_json(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    try:
        root_record = _known_root_json_record(path)
    except ValueError:
        # The ordinary parser owns malformed-input reporting. A recognition
        # probe must not turn a malformed tail into a different record order.
        root_record = False
    if root_record:
        payload = None
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                payload, provenance, rejection = parse_spooled_json(stream, _TRAINING_TEXT_FIELDS, _METADATA_ONLY_FIELDS)
            physical_bytes = path.stat().st_size
            if rejection:
                coverage.processed_bytes += physical_bytes
                coverage.reject(path.name + "#record-1", rejection)
            else:
                record = _payload_record(payload, path.name + "#record-1", physical_bytes, coverage,
                    provenance=provenance, source_directory=path.parent)
                if record is not None:
                    yield record
        finally:
            if payload is not None: payload.close()
        return
    # Item-level decoders still materialize one giant scalar/list. Frame each
    # logical array item/map entry to a lease instead, preserving their order.
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        chars = BoundedCharacters(stream)
        root = chars.nonspace()
        if not root:
            raise ValueError("empty JSON source")
        array, mapping = root == "[", root == "{"
        first = chars.nonspace() if array or mapping else root
        index = 0
        accounted_bytes = 0
        while True:
            if (array and first == "]") or (mapping and first == "}"):
                break
            raw_value = raw_key = wrapped = selected = None
            try:
                if mapping:
                    if first != '"':
                        raise ValueError("JSON map key is not a string")
                    raw_key = capture_json_value(chars, first)
                    if chars.nonspace() != ":":
                        raise ValueError("JSON map entry is missing its colon")
                    first = chars.nonspace()
                raw_value = capture_json_value(chars, first)
                if not raw_value.path and (raw_key is None or not raw_key.path):
                    entry = json.loads(raw_value.text)
                    if raw_key is not None:
                        entry = {"key": json.loads(raw_key.text), "value": entry}
                    payload_source = None
                else:
                    if raw_key is not None:
                        builder = TextBuilder()
                        try:
                            builder.write('{"key":')
                            for piece, _ in raw_key.windows(): builder.write(piece)
                            builder.write(',"value":')
                            for piece, _ in raw_value.windows(): builder.write(piece)
                            builder.write("}")
                            wrapped = builder.finish()
                        except BaseException:
                            builder.close()
                            raise
                    payload_source = wrapped or raw_value
                following = chars.nonspace()
                if array or mapping:
                    end = "]" if array else "}"
                    if following not in {",", end}:
                        raise ValueError("JSON source is missing a comma or container end")
                elif following:
                    raise ValueError("JSON source contains trailing content")
                row_bytes = chars.byte_position - accounted_bytes
                accounted_bytes = chars.byte_position
                index += 1
                name = "%s#record-%d" % (path.name, index)
                if payload_source is None:
                    record = _structured_record(entry, name, row_bytes, coverage, source_directory=path.parent)
                else:
                    if payload_source.path:
                        with open(payload_source.path, "r", encoding="utf-8", newline="") as row:
                            selected, provenance, rejection = parse_spooled_json(row, _TRAINING_TEXT_FIELDS, _METADATA_ONLY_FIELDS)
                    else:
                        selected, provenance, rejection = parse_spooled_json(io.StringIO(payload_source.text), _TRAINING_TEXT_FIELDS, _METADATA_ONLY_FIELDS)
                    if rejection:
                        coverage.processed_bytes += row_bytes
                        coverage.reject(name, rejection)
                        record = None
                    else:
                        record = _payload_record(selected, name, row_bytes, coverage, provenance=provenance, source_directory=path.parent)
                if record is not None:
                    yield record
                if not (array or mapping) or following != ",":
                    break
                first = chars.nonspace()
                if first in {"}", "]", ""}:
                    raise ValueError("JSON source has a trailing comma")
            finally:
                for lease in (raw_value, raw_key, wrapped, selected):
                    if lease is not None:
                        lease.close()
        if chars.nonspace():
            raise ValueError("JSON source contains trailing content")
        coverage.processed_bytes += max(0, path.stat().st_size - accounted_bytes)


def _sqlite_identifier(value: str) -> str:
    return '"%s"' % value.replace('"', '""')


@contextmanager
def sqlite_consistent_snapshot(
    path: Path, committed_sha256: str = ""
) -> Iterator[SQLiteSnapshot]:
    """Lease a single-file SQLite backup containing the committed WAL state.

    Reading the main database file directly is not a snapshot when WAL mode is
    active: recently committed rows can live only in ``-wal``. SQLite's backup
    API takes one transactionally consistent view and folds those pages into a
    standalone database. The digest therefore identifies the bytes actually
    traversed, not merely the stale main file beside an ignored WAL.
    """

    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    declared = committed_sha256.strip().lower()
    if declared:
        if len(declared) != 64 or any(
            value not in "0123456789abcdef" for value in declared
        ):
            raise ValueError("committed SQLite snapshot sha256 is invalid")
        actual = _sha256_path(source_path)
        if actual != declared:
            raise ValueError("committed SQLite snapshot checksum mismatch")
        # The desktop manifest owns and protects this immutable snapshot. Read
        # those exact bytes so its committed identity remains authoritative.
        yield SQLiteSnapshot(path=source_path, sha256=actual)
        return
    with tempfile.TemporaryDirectory(prefix="omni-sqlite-snapshot-") as root:
        snapshot_path = Path(root) / "snapshot.sqlite3"
        source_uri = source_path.as_uri() + "?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, isolation_level=None)
        destination = sqlite3.connect(str(snapshot_path), isolation_level=None)
        try:
            source.execute("PRAGMA query_only = ON")
            source.execute("PRAGMA busy_timeout = 30000")
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        digest = _sha256_path(snapshot_path)
        yield SQLiteSnapshot(path=snapshot_path, sha256=digest)


def sqlite_consistent_snapshot_sha256(path: Path) -> str:
    """Return the identity used by deterministic SQLite traversal."""

    with sqlite_consistent_snapshot(path) as snapshot:
        return snapshot.sha256


def _sqlite_row_order(
    connection: sqlite3.Connection, table: str
) -> Tuple[str, List[str]]:
    quoted_table = _sqlite_identifier(table)
    columns = list(connection.execute("PRAGMA table_info(%s)" % quoted_table))
    primary_key = sorted(
        (
            (int(row[5]), str(row[1]))
            for row in columns
            if len(row) > 5 and int(row[5]) > 0
        ),
        key=lambda value: value[0],
    )
    if primary_key:
        names = [name for _, name in primary_key]
        return ", ".join(_sqlite_identifier(name) for name in names), names

    # Ordinary SQLite tables have a hidden integer row id. A user column may
    # shadow one alias, so choose an unshadowed spelling explicitly.
    declared = {str(row[1]).casefold() for row in columns}
    for alias in ("rowid", "_rowid_", "oid"):
        if alias.casefold() not in declared:
            return alias, ["rowid"]
    raise ValueError(
        "SQLite table %s has no declared primary key and shadows every rowid alias"
        % table
    )


def _iter_sqlite(
    path: Path,
    coverage: DatasetCoverage,
    committed_snapshot_sha256: str = "",
) -> Iterator[DatasetRecord]:
    with sqlite_consistent_snapshot(path, committed_snapshot_sha256) as snapshot:
        snapshot_uri = snapshot.path.resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(snapshot_uri, uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type='table' AND name NOT GLOB 'sqlite_*' "
                    "ORDER BY name COLLATE BINARY"
                )
            ]
            for table in tables:
                quoted_table = _sqlite_identifier(table)
                order_expression, order_columns = _sqlite_row_order(
                    connection, table
                )
                cursor = connection.execute(
                    "SELECT * FROM %s ORDER BY %s"
                    % (quoted_table, order_expression)
                )
                columns = [str(value[0]) for value in (cursor.description or [])]
                row_number = 0
                while True:
                    rows = cursor.fetchmany(256)
                    if not rows:
                        break
                    for row in rows:
                        row_number += 1
                        value = dict(zip(columns, row))
                        encoded_size = len(
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                default=str,
                            ).encode("utf-8")
                        )
                        record = _structured_record(
                            value,
                            "%s#%s-%d" % (path.name, table, row_number),
                            encoded_size,
                            coverage,
                            source_directory=path.parent,
                        )
                        if record is not None:
                            record.provenance = {
                                "sqlite_snapshot_sha256": snapshot.sha256,
                                "sqlite_table": table,
                                "sqlite_order": list(order_columns),
                                **record.provenance,
                            }
                            yield record
        finally:
            connection.close()


def _require_pyarrow() -> Any:
    try:
        import pyarrow  # type: ignore

        return pyarrow
    except ImportError as error:
        raise RuntimeError(
            "Parquet and Arrow ingestion require pyarrow from engine/requirements.txt"
        ) from error


def _metadata_only_columnar_schema(column_names: Iterable[Any]) -> bool:
    normalized = {str(name).strip().lower() for name in column_names}
    return bool(normalized) and normalized.issubset(_METADATA_ONLY_FIELDS)


def _iter_columnar_rows(batch: Any) -> Iterator[Dict[str, Any]]:
    # Keep Arrow's native batch, but never duplicate every row as Python
    # dictionaries/strings at once. A single scalar (including a nested value)
    # still needs to fit memory; this is not a bound on native batch decoding.
    names = batch.schema.names
    columns = [batch.column(index) for index in range(batch.num_columns)]
    for row_index in range(batch.num_rows):
        yield {
            name: column[row_index].as_py()
            for name, column in zip(names, columns)
        }


def _serialized_row_bytes(value: Any) -> int:
    # Match the old json.dumps(...).encode('utf-8') byte accounting without
    # joining a row-wide serialization or allocating a complete escaped scalar
    # and its whole UTF-8 copy. Every string is escaped in bounded pieces.
    return sum(
        len(piece[offset : offset + 32_768].encode("utf-8"))
        for piece in _bounded_json_encoding(value)
        for offset in range(0, len(piece), 32_768)
    )


def _bounded_json_encoding(value, compact=False):
    comma, colon = (",", ":") if compact else (", ", ": ")
    if isinstance(value, ColumnarBinaryValue):
        yield '"'
        for piece in value.literal_chunks():
            yield json.dumps(piece, ensure_ascii=False)[1:-1]
        yield '"'
    elif isinstance(value, (str, ColumnarTextValue)):
        yield '"'
        pieces = value.chunks() if isinstance(value, ColumnarTextValue) else (
            value[offset:offset + TEXT_BLOCK_CHARS] for offset in range(0, len(value), TEXT_BLOCK_CHARS)
        )
        for piece in pieces:
            yield json.dumps(piece, ensure_ascii=False)[1:-1]
        yield '"'
    elif isinstance(value, Mapping):
        yield "{"
        for index, (key, entry) in enumerate(value.items()):
            if index:
                yield comma
            yield from _bounded_json_encoding(str(key), compact)
            yield colon
            yield from _bounded_json_encoding(entry, compact)
        yield "}"
    elif isinstance(value, (list, tuple, ColumnarSequence)):
        yield "["
        for index, entry in enumerate(value):
            if index:
                yield comma
            yield from _bounded_json_encoding(entry, compact)
        yield "]"
    else:
        yield json.dumps(value, ensure_ascii=False, default=str)


def _bounded_columnar_rows(batch, arrow):
    names = batch.schema.names
    columns = [batch.column(index) for index in range(batch.num_columns)]
    for row_index in range(batch.num_rows):
        values = {}
        for name, column in zip(names, columns):
            scalar = column[row_index]
            values[name] = _columnar_value(scalar, arrow)
        yield values


def _iter_columnar(
    path: Path, format_name: str, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    require_parser_resources("columnar decoder import", ram_bytes=32 * 1024 * 1024)
    arrow = _require_pyarrow()
    if format_name == "parquet":
        import pyarrow.parquet as parquet  # type: ignore

        admit_parquet_footer(path)
        parquet_file = parquet.ParquetFile(path)
        if _metadata_only_columnar_schema(parquet_file.schema_arrow.names):
            row_count = int(parquet_file.metadata.num_rows)
            coverage.reject_many(
                "%s#all-rows" % path.name,
                "metadata-only Parquet schema has no trainable content; "
                "the referenced source/blob payload is required",
                row_count,
                processed_bytes=path.stat().st_size,
            )
            return
        expected_rows = int(parquet_file.metadata.num_rows)
        def admitted_batches():
            for group_index in range(parquet_file.metadata.num_row_groups):
                admit_parquet_row_group(parquet_file.metadata, group_index)
                yield from parquet_file.iter_batches(
                    batch_size=256, row_groups=[group_index], use_threads=False
                )
        batches = admitted_batches()
    else:
        import pyarrow.ipc as ipc  # type: ignore

        source = arrow.memory_map(str(path), "r")
        frame_source = arrow.memory_map(str(path), "r")
        try:
            file_size = path.stat().st_size
            dictionary_allocation = 0
            for frame_kind, estimate in ipc_allocation_frames(frame_source, ipc, file_size):
                if frame_kind == 2:
                    dictionary_allocation += estimate
            require_parser_resources("Arrow IPC dictionary decode", ram_bytes=dictionary_allocation)
            source.seek(0)
            try:
                reader = ipc.open_file(source)
                file_reader = True
            except Exception as error:
                if _parser_resource_failure(error):
                    raise
                source.seek(0)
                reader = ipc.open_stream(source)
                file_reader = False
            batch_index = 0
            for frame_kind, estimate in ipc_allocation_frames(frame_source, ipc, file_size):
                if frame_kind != 3:
                    continue
                require_parser_resources("Arrow IPC native batch decode", ram_bytes=estimate + dictionary_allocation)
                batch = reader.get_batch(batch_index) if file_reader else reader.read_next_batch()
                for row_index, value in enumerate(_bounded_columnar_rows(batch, arrow)):
                    encoded_size = _serialized_row_bytes(value)
                    record = _structured_record(
                        value,
                        "%s#batch-%d-row-%d"
                        % (path.name, batch_index + 1, row_index + 1),
                        encoded_size,
                        coverage,
                        source_directory=path.parent,
                    )
                    if record is not None:
                        yield record
                batch_index += 1
            if file_reader and batch_index != reader.num_record_batches:
                raise DatasetTraversalIncomplete("Arrow IPC allocation frames do not exhaust its record batches")
            return
        finally:
            source.close()
            frame_source.close()
    discovered_before = coverage.discovered_records
    try:
        logical_row = 0
        for batch in batches:
            for value in _bounded_columnar_rows(batch, arrow):
                encoded_size = _serialized_row_bytes(value)
                record = _structured_record(
                    value,
                    "%s#batch-%d-row-%d" % (path.name, logical_row // 256 + 1, logical_row % 256 + 1),
                    encoded_size,
                    coverage,
                    source_directory=path.parent,
                )
                if record is not None:
                    yield record
                logical_row += 1
    except Exception as error:
        if isinstance(error, DatasetResourcePause) or _parser_resource_failure(error):
            raise
        raise DatasetTraversalIncomplete(
            "Parquet traversal failed before its footer-declared rows were visited"
        ) from error
    observed_rows = coverage.discovered_records - discovered_before
    if observed_rows != expected_rows:
        raise DatasetTraversalIncomplete(
            "Parquet traversal visited %d of %d footer-declared rows"
            % (observed_rows, expected_rows)
        )


def _xml_local_name(name: str) -> str:
    return str(name).rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _natural_member_key(name: str) -> List[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _iter_streamed_xml(
    source: BinaryIO,
    name: str,
    coverage: DatasetCoverage,
    *,
    capture_tags: Optional[Set[str]] = None,
    line_tags: Optional[Set[str]] = None,
    space_tags: Optional[Set[str]] = None,
    attribute_names: Optional[Set[str]] = None,
    chunk_chars: int = 32_768,
) -> Iterator[DatasetRecord]:
    """Extract XML character data incrementally without materializing a member.

    Office/OpenDocument containers can contain very large XML parts. Expat is
    fed bounded byte blocks and emitted text is drained into ordinary dataset
    records as it arrives. DTDs and external entities are rejected.
    """

    wanted = set(capture_tags or ())
    lines = set(line_tags or ())
    spaces = set(space_tags or ())
    attributes = set(attribute_names or ())
    capture_all = capture_tags is None
    active_depth = 1 if capture_all else 0
    fragments: List[str] = []
    pending = ""

    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    def reject_doctype(*_args: Any) -> None:
        raise ValueError("XML document type declarations are not accepted")

    def start_element(raw_name: str, raw_attributes: Dict[str, str]) -> None:
        nonlocal active_depth
        local = _xml_local_name(raw_name)
        if not capture_all and local in wanted:
            active_depth += 1
        if active_depth > 0 and local in spaces:
            fragments.append(" ")
        if active_depth > 0 and attributes:
            for key, value in raw_attributes.items():
                if _xml_local_name(key) in attributes and value:
                    fragments.extend((str(value), " "))

    def characters(value: str) -> None:
        if active_depth > 0 and value:
            fragments.append(value)

    def end_element(raw_name: str) -> None:
        nonlocal active_depth
        local = _xml_local_name(raw_name)
        if not capture_all and local in wanted:
            active_depth = max(0, active_depth - 1)
        if local in lines:
            fragments.append("\n")

    parser.StartDoctypeDeclHandler = reject_doctype
    parser.ExternalEntityRefHandler = lambda *_args: 0
    parser.StartElementHandler = start_element
    parser.CharacterDataHandler = characters
    parser.EndElementHandler = end_element

    while True:
        block = source.read(256 * 1024)
        if not block:
            break
        parser.Parse(block, False)
        if fragments:
            pending += "".join(fragments)
            fragments.clear()
        while len(pending) >= chunk_chars:
            split_at = pending.rfind("\n", 0, chunk_chars)
            if split_at <= 0:
                split_at = chunk_chars
            piece, pending = pending[:split_at], pending[split_at:]
            record = _record(
                piece,
                name,
                len(piece.encode("utf-8", errors="replace")),
                coverage,
            )
            if record is not None:
                yield record
    parser.Parse(b"", True)
    if fragments:
        pending += "".join(fragments)
    if pending:
        record = _record(
            pending,
            name,
            len(pending.encode("utf-8", errors="replace")),
            coverage,
        )
        if record is not None:
            yield record


def _office_members(
    archive: zipfile.ZipFile, suffix: str
) -> List[zipfile.ZipInfo]:
    members = [member for member in archive.infolist() if not member.is_dir()]

    def selected(member: zipfile.ZipInfo) -> bool:
        name = member.filename.replace("\\", "/").lower()
        if suffix == ".docx":
            return bool(
                re.fullmatch(
                    r"word/(?:document|footnotes|endnotes|comments|header\d+|footer\d+)\.xml",
                    name,
                )
            )
        if suffix == ".pptx":
            return bool(
                re.fullmatch(
                    r"ppt/(?:slides/slide\d+|notesslides/notesslide\d+)\.xml",
                    name,
                )
            )
        if suffix == ".xlsx":
            return name == "xl/sharedstrings.xml" or bool(
                re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
            )
        return name == "content.xml"

    chosen = [member for member in members if selected(member)]
    if suffix == ".xlsx":
        chosen.sort(
            key=lambda member: (
                0 if member.filename.lower() == "xl/sharedstrings.xml" else 1,
                _natural_member_key(member.filename),
            )
        )
    else:
        chosen.sort(key=lambda member: _natural_member_key(member.filename))
    return chosen


def _office_xml_options(
    suffix: str, member_name: str
) -> Dict[str, Set[str]]:
    lower_name = member_name.replace("\\", "/").lower()
    if suffix == ".docx":
        return {
            "capture_tags": {"t", "instrText"},
            "line_tags": {"p", "tr"},
            "space_tags": {"tab", "br"},
        }
    if suffix == ".pptx":
        return {
            "capture_tags": {"t"},
            "line_tags": {"p"},
            "space_tags": {"br"},
        }
    if suffix == ".xlsx":
        if lower_name == "xl/sharedstrings.xml":
            return {
                "capture_tags": {"t"},
                "line_tags": {"si"},
                "space_tags": set(),
            }
        return {
            "capture_tags": {"v", "t"},
            "line_tags": {"c", "row"},
            "space_tags": set(),
        }
    if suffix == ".ods":
        return {
            "capture_tags": {"table-cell"},
            "line_tags": {"table-cell", "table-row"},
            "space_tags": {"s", "tab", "line-break"},
            "attribute_names": {
                "value",
                "date-value",
                "time-value",
                "boolean-value",
                "string-value",
            },
        }
    return {
        "capture_tags": {"p", "h"},
        "line_tags": {"p", "h"},
        "space_tags": {"s", "tab", "line-break"},
    }


def _iter_office_archive(
    path: Path, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    if not zipfile.is_zipfile(path):
        raise ValueError("%s is not a valid Office/OpenDocument container" % path.name)
    coverage.shards += 1
    with zipfile.ZipFile(path) as archive:
        members = _office_members(archive, path.suffix.lower())
        if not members:
            raise ValueError("office document contains no supported content XML")
        for member in members:
            coverage.discovered_files += 1
            display_name = "%s!%s" % (path.name, member.filename)
            discovered_before = coverage.discovered_records
            try:
                with archive.open(member) as source:
                    yield from _iter_streamed_xml(
                        source,
                        display_name,
                        coverage,
                        **_office_xml_options(path.suffix.lower(), member.filename),
                    )
                coverage.completed_files += 1
            except Exception as error:
                if _parser_resource_failure(error):
                    raise
                if coverage.discovered_records > discovered_before:
                    coverage.traversal_incomplete = True
                coverage.reject_file(
                    display_name,
                    str(error),
                    already_discovered=True,
                )


def _iter_standalone_compressed(
    path: Path, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    suffix = path.suffix.lower()
    opener = {
        ".gz": gzip.open,
        ".bz2": bz2.open,
        ".xz": lzma.open,
    }.get(suffix)
    if opener is None:
        raise ValueError("unsupported standalone compression format")
    inner_name = path.name[: -len(suffix)] or (path.name + ".txt")
    coverage.discovered_files += 1
    discovered_before = coverage.discovered_records
    try:
        with opener(
            path, mode="rt", encoding="utf-8", errors="replace", newline=""
        ) as stream:
            yield from _iter_text_stream(
                stream,
                "%s!%s" % (path.name, inner_name),
                coverage,
            )
        coverage.completed_files += 1
    except Exception as error:
        if _parser_resource_failure(error):
            raise
        if coverage.discovered_records > discovered_before:
            coverage.traversal_incomplete = True
        coverage.reject_file(
            "%s!%s" % (path.name, inner_name),
            str(error),
            already_discovered=True,
        )


def _iter_binary_member(
    source: BinaryIO,
    *,
    archive_name: str,
    member_name: str,
    kind: str,
    coverage: DatasetCoverage,
) -> Iterator[DatasetRecord]:
    """Spool one media member and lease its path to the record consumer."""

    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="omni-dataset-member-",
        suffix=Path(member_name).suffix.lower(),
        delete=False,
    )
    temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with temporary:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                temporary.write(block)
                digest.update(block)
                bytes_read += len(block)
            temporary.flush()
            os.fsync(temporary.fileno())
        checksum = digest.hexdigest()
        yield _binary_record(
            name="%s!%s" % (archive_name, member_name),
            local_path=temporary_path,
            bytes_read=bytes_read,
            content_sha256=checksum,
            kind=kind,
            coverage=coverage,
            provenance={
                "archive": archive_name,
                "member": member_name,
                "sha256": checksum,
            },
        )
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _iter_archive(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    suffix = path.suffix.lower()
    if suffix in OFFICE_EXTENSIONS:
        yield from _iter_office_archive(path, coverage)
        return
    coverage.shards += 1
    if suffix in {".gz", ".bz2", ".xz"} and not tarfile.is_tarfile(path):
        yield from _iter_standalone_compressed(path, coverage)
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                coverage.discovered_files += 1
                member_path = Path(member.filename)
                discovered_before = coverage.discovered_records
                try:
                    with archive.open(member) as source:
                        suffix = member_path.suffix.lower()
                        if suffix == ".xml":
                            yield from _iter_streamed_xml(
                                source,
                                "%s!%s" % (path.name, member.filename),
                                coverage,
                                line_tags={"p", "h", "row", "tr"},
                            )
                        elif suffix in {".xhtml", ".html", ".htm"}:
                            with io.TextIOWrapper(
                                source,
                                encoding="utf-8",
                                errors="replace",
                                newline="",
                            ) as text_stream:
                                yield from _iter_text_stream(
                                    text_stream,
                                    "%s!%s" % (path.name, member.filename),
                                    coverage,
                                )
                        elif suffix in TEXT_EXTENSIONS or suffix in {
                            ".csv",
                            ".tsv",
                            ".json",
                            ".jsonl",
                            ".ndjson",
                        }:
                            with io.TextIOWrapper(
                                source, encoding="utf-8", errors="replace", newline=""
                            ) as text_stream:
                                yield from _iter_text_stream(
                                    text_stream,
                                    "%s!%s" % (path.name, member.filename),
                                    coverage,
                                )
                        elif _media_kind(member_path) is not None:
                            yield from _iter_binary_member(
                                source,
                                archive_name=path.name,
                                member_name=member.filename,
                                kind=str(_media_kind(member_path)),
                                coverage=coverage,
                            )
                        else:
                            coverage.reject_file(
                                "%s!%s" % (path.name, member.filename),
                                "unsupported binary archive member",
                                already_discovered=True,
                            )
                            continue
                    coverage.completed_files += 1
                except Exception as error:
                    if _parser_resource_failure(error):
                        raise
                    if coverage.discovered_records > discovered_before:
                        coverage.traversal_incomplete = True
                    coverage.reject_file(
                        "%s!%s" % (path.name, member.filename),
                        str(error),
                        already_discovered=True,
                    )
        return

    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            coverage.discovered_files += 1
            discovered_before = coverage.discovered_records
            source = archive.extractfile(member)
            if source is None:
                coverage.reject_file(
                    "%s!%s" % (path.name, member.name),
                    "member could not be opened",
                    already_discovered=True,
                )
                continue
            try:
                suffix = Path(member.name).suffix.lower()
                if suffix in TEXT_EXTENSIONS or suffix in {
                    ".csv",
                    ".tsv",
                    ".json",
                    ".jsonl",
                    ".ndjson",
                }:
                    with io.TextIOWrapper(
                        source, encoding="utf-8", errors="replace", newline=""
                    ) as text_stream:
                        yield from _iter_text_stream(
                            text_stream,
                            "%s!%s" % (path.name, member.name),
                            coverage,
                        )
                elif _media_kind(Path(member.name)) is not None:
                    yield from _iter_binary_member(
                        source,
                        archive_name=path.name,
                        member_name=member.name,
                        kind=str(_media_kind(Path(member.name))),
                        coverage=coverage,
                    )
                else:
                    coverage.reject_file(
                        "%s!%s" % (path.name, member.name),
                        "unsupported binary archive member",
                        already_discovered=True,
                    )
                    continue
                coverage.completed_files += 1
            except Exception as error:
                if _parser_resource_failure(error):
                    raise
                if coverage.discovered_records > discovered_before:
                    coverage.traversal_incomplete = True
                coverage.reject_file(
                    "%s!%s" % (path.name, member.name),
                    str(error),
                    already_discovered=True,
                )


def _manifest_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _manifest_metadata(value: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, entry in value.items():
        normalized = _manifest_key(key)
        if normalized in _MANIFEST_METADATA_KEYS and isinstance(
            entry, (str, int, float, bool, dict)
        ):
            metadata[normalized] = entry
    return metadata


def _manifest_references(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
    elif isinstance(value, list):
        for entry in value:
            yield from _manifest_references(entry)


def _declared_sha256(metadata: Dict[str, Any]) -> str:
    candidate: Any = metadata.get("sha256")
    if candidate is None:
        checksum = metadata.get("checksum", metadata.get("hash"))
        if isinstance(checksum, dict):
            candidate = checksum.get("sha256") or checksum.get("SHA256")
        else:
            candidate = checksum
    if not isinstance(candidate, str):
        return ""
    normalized = candidate.strip().lower()
    for prefix in ("sha256:", "sha256="):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    return normalized


def _manifest_shards(
    value: Any,
    *,
    within_container: bool = False,
    inherited_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[_ManifestShard]:
    inherited = dict(inherited_metadata or {})
    if isinstance(value, str):
        if within_container and value.strip():
            yield _ManifestShard(
                reference=value.strip(),
                sha256=_declared_sha256(inherited),
                metadata=inherited,
            )
        return
    if isinstance(value, list):
        for entry in value:
            yield from _manifest_shards(
                entry,
                within_container=within_container,
                inherited_metadata=inherited,
            )
        return
    if not isinstance(value, dict):
        return

    metadata = {**inherited, **_manifest_metadata(value)}
    direct_references: List[str] = []
    for key, entry in value.items():
        if _manifest_key(key) in _MANIFEST_REFERENCE_KEYS:
            direct_references.extend(_manifest_references(entry))
    if direct_references:
        for reference in direct_references:
            yield _ManifestShard(
                reference=reference,
                sha256=_declared_sha256(metadata),
                metadata=dict(metadata),
            )

    traversed_container = False
    for key, entry in value.items():
        normalized = _manifest_key(key)
        if normalized not in _MANIFEST_CONTAINER_KEYS:
            continue
        traversed_container = True
        yield from _manifest_shards(
            entry,
            within_container=True,
            inherited_metadata=metadata,
        )

    # Split maps commonly use arbitrary keys such as "train" and "test".
    # Once inside a recognized manifest container, descend these maps while
    # deliberately excluding checksum/license metadata strings.
    if within_container and not direct_references and not traversed_container:
        for key, entry in value.items():
            normalized = _manifest_key(key)
            if normalized in _MANIFEST_METADATA_KEYS:
                continue
            child_metadata = dict(metadata)
            child_metadata.setdefault("split", str(key))
            yield from _manifest_shards(
                entry,
                within_container=True,
                inherited_metadata=child_metadata,
            )


def _manifest_paths(value: Any) -> Iterator[str]:
    """Compatibility iterator used by callers that only need references."""

    for shard in _manifest_shards(value):
        yield shard.reference


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_shard_provenance(
    records: Iterable[DatasetRecord],
    *,
    local_name: str,
    display_name: str,
    provenance: Dict[str, Any],
) -> Iterator[DatasetRecord]:
    for record in records:
        if record.name.startswith(local_name):
            record.name = display_name + record.name[len(local_name) :]
        record.provenance = {**provenance, **record.provenance}
        yield record


def _iter_huggingface_manifest(
    path: Path, coverage: DatasetCoverage, seen: Set[Path]
) -> Iterator[DatasetRecord]:
    require_parser_resources("Hugging Face manifest JSON metadata decode", ram_bytes=path.stat().st_size * 8)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    shards: List[_ManifestShard] = []
    seen_references: Set[str] = set()
    for shard in _manifest_shards(value):
        if shard.reference in seen_references:
            continue
        seen_references.add(shard.reference)
        shards.append(shard)
    if not shards:
        coverage.reject(path.name, "manifest contains no data_files or shards")
        return

    manifest_provenance = str(path.resolve())
    for shard in shards:
        raw = shard.reference
        if raw.startswith(("https://", "http://", "hf://")):
            if not shard.sha256:
                # A manifest file can remain byte-identical while an unpinned
                # remote object changes. Record-level resume and completion
                # receipts therefore require a declared content identity; the
                # downloader still verifies it against the resolved bytes.
                coverage.reject_file(
                    raw,
                    "remote manifest shard requires a declared sha256 for "
                    "restart-safe training",
                )
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix="omni-remote-dataset-"
                ) as temporary_directory:
                    downloaded = _download_remote_shard(
                        raw,
                        Path(temporary_directory),
                        expected_sha256=shard.sha256,
                    )
                    records = iter_dataset_records(
                        downloaded.path,
                        coverage=coverage,
                        _seen=seen,
                    )
                    yield from _with_shard_provenance(
                        records,
                        local_name=downloaded.path.name,
                        display_name=downloaded.final_url,
                        provenance={
                            "manifest": manifest_provenance,
                            "requested_url": downloaded.requested_url,
                            "final_url": downloaded.final_url,
                            "shard_sha256": downloaded.sha256,
                            "downloaded_bytes": downloaded.bytes_read,
                            "content_type": downloaded.content_type,
                            "declared": dict(shard.metadata),
                        },
                    )
            except Exception as error:
                if _parser_resource_failure(error):
                    raise
                coverage.reject_file(raw, str(error))
            continue

        candidate = (path.parent / raw).resolve()
        try:
            candidate.relative_to(path.parent.resolve())
        except ValueError:
            coverage.reject_file(
                raw,
                "manifest path escapes its dataset directory",
            )
            continue
        if not candidate.exists():
            coverage.reject_file(raw, "manifest shard was not found")
            continue
        if shard.sha256:
            declared = shard.sha256.strip().lower()
            if len(declared) != 64 or any(
                value not in "0123456789abcdef" for value in declared
            ):
                coverage.reject_file(raw, "manifest shard sha256 is invalid")
                continue
            actual = _sha256_path(candidate)
            if actual != declared:
                coverage.reject_file(raw, "manifest shard checksum mismatch")
                continue
        else:
            actual = _sha256_path(candidate)
        yield from _with_shard_provenance(
            iter_dataset_records(candidate, coverage=coverage, _seen=seen),
            local_name=candidate.name,
            display_name=candidate.name,
            provenance={
                "manifest": manifest_provenance,
                "shard_path": str(candidate),
                "shard_sha256": actual,
                "declared": dict(shard.metadata),
            },
        )


def _iter_pdf(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF ingestion requires pypdf from engine/requirements.txt"
        ) from error
    reader = PdfReader(str(path))
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        record = _record(
            text,
            "%s#page-%d" % (path.name, index + 1),
            len(text.encode("utf-8", errors="replace")),
            coverage,
        )
        if record is not None:
            yield record


def iter_dataset_records(
    path: Path,
    requested_kind: str = "",
    coverage: Optional[DatasetCoverage] = None,
    _seen: Optional[Set[Path]] = None,
    _committed_sqlite_snapshot_sha256: str = "",
    _resource_admission=None,
) -> Iterator[DatasetRecord]:
    """Visit complete logical records with leases and shared resource admission."""
    with parser_admission(_resource_admission):
        for record in _iter_dataset_records_impl(
            path, requested_kind, coverage, _seen, _committed_sqlite_snapshot_sha256
        ):
            leased_payload = record.text_payload
            try:
                yield record
            finally:
                if leased_payload is not None:
                    leased_payload.close()


def _iter_dataset_records_impl(
    path: Path,
    requested_kind: str = "",
    coverage: Optional[DatasetCoverage] = None,
    _seen: Optional[Set[Path]] = None,
    _committed_sqlite_snapshot_sha256: str = "",
) -> Iterator[DatasetRecord]:
    """Visit every readable record in ``path`` exactly once for this traversal."""

    target = path.resolve()
    state = coverage if coverage is not None else DatasetCoverage()
    seen = _seen if _seen is not None else set()
    if target in seen:
        return
    seen.add(target)
    if target.is_dir():
        for child in sorted(target.iterdir(), key=lambda value: value.name.lower()):
            if child.is_symlink():
                state.reject_file(str(child), "symbolic links are not followed")
                continue
            if child.is_dir() and (
                child.name.lower() in _INTERNAL_DATASET_DIRECTORIES
                or child.name.startswith(".")
            ):
                continue
            yield from iter_dataset_records(child, coverage=state, _seen=seen)
        return
    if not target.is_file():
        state.reject_file(str(target), "dataset path is not a regular file")
        return

    auxiliary_rejection = _auxiliary_dataset_file_rejection(target)
    if auxiliary_rejection is not None:
        state.reject_file(str(target), auxiliary_rejection)
        return

    state.discovered_files += 1
    format_name = dataset_format(target, requested_kind)
    file_rejected = False
    discovered_before = state.discovered_records
    try:
        if format_name == "text":
            yield from _iter_text_path(target, state)
        elif format_name == "csv":
            yield from _iter_delimited(target, ",", state)
        elif format_name == "tsv":
            yield from _iter_delimited(target, "\t", state)
        elif format_name == "jsonl":
            yield from _iter_jsonl(target, state)
        elif format_name == "json":
            yield from _iter_json(target, state)
        elif format_name == "sqlite":
            yield from _iter_sqlite(
                target,
                state,
                committed_snapshot_sha256=(
                    _committed_sqlite_snapshot_sha256
                ),
            )
        elif format_name in {"parquet", "arrow"}:
            yield from _iter_columnar(target, format_name, state)
        elif format_name == "archive":
            yield from _iter_archive(target, state)
        elif format_name == "huggingface":
            yield from _iter_huggingface_manifest(target, state, seen)
        elif format_name == "pdf":
            yield from _iter_pdf(target, state)
        elif format_name in {"image", "audio", "video"}:
            checksum = _sha256_path(target)
            yield _binary_record(
                name=target.name,
                local_path=target,
                bytes_read=target.stat().st_size,
                content_sha256=checksum,
                kind=format_name,
                coverage=state,
                provenance={
                    "source_path": str(target),
                    "sha256": checksum,
                },
            )
        else:
            with target.open("rb") as stream:
                sample = stream.read(256 * 1024)
            decoded = sample.decode("utf-8", errors="replace")
            if decoded and decoded.count("\ufffd") / len(decoded) < 0.02:
                yield from _iter_text_path(target, state)
            else:
                state.reject_file(
                    str(target),
                    "unsupported binary dataset format",
                    already_discovered=True,
                )
                file_rejected = True
    except Exception as error:
        if isinstance(error, DatasetResourcePause) or _parser_resource_failure(error):
            state.traversal_incomplete = True
            if isinstance(error, DatasetResourcePause):
                raise
            raise DatasetResourcePause(
                "dataset decoding reached a physical resource boundary; resume its uncommitted suffix"
            ) from error
        # A desktop-provided SQLite snapshot is already content-addressed by
        # the authoritative ingestion manifest. Treat a bad declaration as a
        # transaction-integrity failure, not as an ordinary invalid dataset
        # row that coverage reporting may skip. Otherwise training could be
        # recorded under bytes different from the committed generation.
        if format_name == "sqlite" and _committed_sqlite_snapshot_sha256:
            raise
        if (
            isinstance(error, DatasetTraversalIncomplete)
            or _parser_resource_failure(error)
            or state.discovered_records > discovered_before
        ):
            state.traversal_incomplete = True
        if not file_rejected:
            state.reject_file(
                str(target),
                (
                    "dataset parsing ran out of memory; traversal must be resumed"
                    if isinstance(error, MemoryError)
                    else str(error)
                ),
                already_discovered=True,
            )
            file_rejected = True
    if not file_rejected:
        state.completed_files += 1
