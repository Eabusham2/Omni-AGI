"""Streaming dataset readers used by OmniCortex ingestion.

The readers deliberately expose records instead of loading a complete corpus
into memory.  Optional columnar formats use PyArrow, while archives, SQLite,
CSV, JSONL, Hugging Face-style manifests, and ordinary text use the standard
library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import os
import socket
import sqlite3
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Optional, Set
from xml.etree import ElementTree


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
    ".odt",
}
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
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".mpeg",
    ".mpg",
}
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


@dataclass
class DatasetCoverage:
    discovered_files: int = 0
    completed_files: int = 0
    discovered_records: int = 0
    processed_records: int = 0
    rejected_records: int = 0
    processed_bytes: int = 0
    shards: int = 0
    modality_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def reject(self, source: str, message: str) -> None:
        self.rejected_records += 1
        self.errors.append({"source": source, "message": message})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "discoveredFiles": self.discovered_files,
            "completedFiles": self.completed_files,
            "discoveredRecords": self.discovered_records,
            "processedRecords": self.processed_records,
            "rejectedRecords": self.rejected_records,
            "processedBytes": self.processed_bytes,
            "shards": self.shards,
            "modalityCounts": dict(self.modality_counts),
            "errors": list(self.errors),
        }


def _media_kind(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
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


def _record(
    text: str,
    name: str,
    bytes_read: int,
    coverage: DatasetCoverage,
    kind: str = "text",
) -> Optional[DatasetRecord]:
    clean = text.replace("\x00", "").strip()
    coverage.discovered_records += 1
    coverage.processed_bytes += max(0, int(bytes_read))
    if not clean:
        coverage.reject(name, "record contained no usable text")
        return None
    coverage.processed_records += 1
    coverage.modality_counts[kind] = coverage.modality_counts.get(kind, 0) + 1
    return DatasetRecord(clean, name, max(0, int(bytes_read)), kind)


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


def _iter_delimited(
    path: Path, delimiter: str, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        for index, row in enumerate(reader):
            text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            record = _record(
                text,
                "%s#row-%d" % (path.name, index + 1),
                len(text.encode("utf-8")),
                coverage,
            )
            if record is not None:
                yield record


def _iter_jsonl(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for index, line in enumerate(stream):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except json.JSONDecodeError as error:
                coverage.reject(
                    "%s#line-%d" % (path.name, index + 1),
                    "invalid JSONL: %s" % error,
                )
                continue
            record = _record(
                text,
                "%s#line-%d" % (path.name, index + 1),
                len(line.encode("utf-8", errors="replace")),
                coverage,
            )
            if record is not None:
                yield record


def _iter_json(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    # The packaged runtime includes ijson so large arrays/maps are visited
    # incrementally. Keep a standard-library fallback for minimal developer
    # environments.
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    values: Iterable[Any]
    stream: Any = None
    if ijson is not None:
        stream = path.open("rb")
        prefix = stream.read(4_096).lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
        stream.seek(0)
        if prefix == b"[":
            values = ijson.items(stream, "item")
        elif prefix == b"{":
            values = (
                {"key": key, "value": entry}
                for key, entry in ijson.kvitems(stream, "")
            )
        else:
            values = ijson.items(stream, "")
    else:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            values = (
                [{"key": key, "value": entry} for key, entry in value.items()]
                if value
                else [value]
            )
        else:
            values = [value]
    try:
        for index, entry in enumerate(values):
            text = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
            record = _record(
                text,
                "%s#record-%d" % (path.name, index + 1),
                len(text.encode("utf-8")),
                coverage,
            )
            if record is not None:
                yield record
    finally:
        if stream is not None:
            stream.close()


def _iter_sqlite(path: Path, coverage: DatasetCoverage) -> Iterator[DatasetRecord]:
    connection = sqlite3.connect("file:%s?mode=ro" % path.as_posix(), uri=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            escaped = table.replace('"', '""')
            cursor = connection.execute('SELECT * FROM "%s"' % escaped)
            columns = [str(value[0]) for value in (cursor.description or [])]
            row_number = 0
            while True:
                rows = cursor.fetchmany(256)
                if not rows:
                    break
                for row in rows:
                    row_number += 1
                    value = dict(zip(columns, row))
                    text = json.dumps(value, ensure_ascii=False, default=str)
                    record = _record(
                        text,
                        "%s#%s-%d" % (path.name, table, row_number),
                        len(text.encode("utf-8")),
                        coverage,
                    )
                    if record is not None:
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


def _iter_columnar(
    path: Path, format_name: str, coverage: DatasetCoverage
) -> Iterator[DatasetRecord]:
    _require_pyarrow()
    if format_name == "parquet":
        import pyarrow.parquet as parquet  # type: ignore

        batches = parquet.ParquetFile(path).iter_batches(batch_size=256)
    else:
        import pyarrow.ipc as ipc  # type: ignore

        source = path.open("rb")
        try:
            try:
                reader = ipc.open_file(source)
                batches = (reader.get_batch(index) for index in range(reader.num_record_batches))
            except Exception:
                source.seek(0)
                batches = ipc.open_stream(source)
            for batch_index, batch in enumerate(batches):
                for row_index, value in enumerate(batch.to_pylist()):
                    text = json.dumps(value, ensure_ascii=False, default=str)
                    record = _record(
                        text,
                        "%s#batch-%d-row-%d"
                        % (path.name, batch_index + 1, row_index + 1),
                        len(text.encode("utf-8")),
                        coverage,
                    )
                    if record is not None:
                        yield record
            return
        finally:
            source.close()
    for batch_index, batch in enumerate(batches):
        for row_index, value in enumerate(batch.to_pylist()):
            text = json.dumps(value, ensure_ascii=False, default=str)
            record = _record(
                text,
                "%s#batch-%d-row-%d" % (path.name, batch_index + 1, row_index + 1),
                len(text.encode("utf-8")),
                coverage,
            )
            if record is not None:
                yield record


def _xml_text(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    return " ".join(part.strip() for part in root.itertext() if part.strip())


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
    coverage.shards += 1
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                coverage.discovered_files += 1
                member_path = Path(member.filename)
                try:
                    with archive.open(member) as source:
                        suffix = member_path.suffix.lower()
                        if suffix in {".xml", ".xhtml", ".html", ".htm"}:
                            data = source.read()
                            text = _xml_text(data) if suffix == ".xml" else data.decode(
                                "utf-8", errors="replace"
                            )
                            record = _record(
                                text,
                                "%s!%s" % (path.name, member.filename),
                                len(data),
                                coverage,
                            )
                            if record is not None:
                                yield record
                        elif suffix in TEXT_EXTENSIONS or suffix in {
                            ".csv",
                            ".tsv",
                            ".json",
                            ".jsonl",
                            ".ndjson",
                        }:
                            text_stream = io.TextIOWrapper(
                                source, encoding="utf-8", errors="replace", newline=""
                            )
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
                            coverage.reject(
                                "%s!%s" % (path.name, member.filename),
                                "unsupported binary archive member",
                            )
                    coverage.completed_files += 1
                except Exception as error:
                    coverage.reject(
                        "%s!%s" % (path.name, member.filename), str(error)
                    )
        return

    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            coverage.discovered_files += 1
            source = archive.extractfile(member)
            if source is None:
                coverage.reject(
                    "%s!%s" % (path.name, member.name), "member could not be opened"
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
                    coverage.reject(
                        "%s!%s" % (path.name, member.name),
                        "unsupported binary archive member",
                    )
                coverage.completed_files += 1
            except Exception as error:
                coverage.reject("%s!%s" % (path.name, member.name), str(error))


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
                coverage.reject(raw, str(error))
            continue

        candidate = (path.parent / raw).resolve()
        try:
            candidate.relative_to(path.parent.resolve())
        except ValueError:
            coverage.reject(raw, "manifest path escapes its dataset directory")
            continue
        if not candidate.exists():
            coverage.reject(raw, "manifest shard was not found")
            continue
        if shard.sha256:
            declared = shard.sha256.strip().lower()
            if len(declared) != 64 or any(
                value not in "0123456789abcdef" for value in declared
            ):
                coverage.reject(raw, "manifest shard sha256 is invalid")
                continue
            actual = _sha256_path(candidate)
            if actual != declared:
                coverage.reject(raw, "manifest shard checksum mismatch")
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
                state.reject(str(child), "symbolic links are not followed")
                continue
            yield from iter_dataset_records(child, coverage=state, _seen=seen)
        return
    if not target.is_file():
        state.reject(str(target), "dataset path is not a regular file")
        return

    state.discovered_files += 1
    format_name = dataset_format(target, requested_kind)
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
            yield from _iter_sqlite(target, state)
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
                state.reject(str(target), "unsupported binary dataset format")
    except Exception as error:
        state.reject(str(target), str(error))
    finally:
        state.completed_files += 1
