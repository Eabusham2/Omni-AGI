"""Explicit, attributable selected web pixels; never URLs as seen images.

Main rechecks current tool/scope authority before offering or dispatching the
owned descriptor. Native independently checks exact task custody and bytes.
Read-only perception does not claim learning; ordinary image ingestion owns
lasting updates and its replay-safe transaction receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from torch.nn import functional as F

from .mcp_sampling_media import IMAGE_MIMES, _image_windows
from .parameter_diagnostics import native_parameter_read_only

_SHA = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CHUNK = 65536


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value)


def _id(value):
    return isinstance(value, str) and _ID.fullmatch(value)


def _url(value):
    if not isinstance(value, str) or "\0" in value:
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password and not parsed.fragment


def validate_selection(value, *, brain_id=None, turn_id=None):
    fields = {"format", "version", "brainId", "turnId", "reference", "download", "attachment", "selectionSha256", "observation"}
    if not isinstance(value, dict) or set(value) != fields or value["format"] != "omni-web-image-selection" or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("selected web image requires its exact typed source descriptor")
    reference, downloaded, attachment = value["reference"], value["download"], value["attachment"]
    if not isinstance(reference, dict) or set(reference) - {"format", "version", "id", "brainId", "turnId", "url", "pageUrl", "label", "fetched", "source"}:
        raise ValueError("selected image reference is not attributable")
    source = reference.get("source")
    if (not isinstance(source, dict) or set(source) != {"kind", "url", "sha256", "fetchedAt"}
            or source["kind"] not in {"search-response", "html-response", "image-response"} or not _sha(source["sha256"]) or not _url(source["url"])
            or reference.get("format") != "omni-web-image-reference" or reference.get("version") != 1 or reference.get("fetched") is not False
            or not all(_id(reference.get(key)) for key in ("brainId", "turnId"))
            or not all(_url(reference.get(key)) for key in ("url", "pageUrl"))
            or "label" in reference and not isinstance(reference["label"], str)):
        raise ValueError("selected image reference lost its observed source fields")
    if not _sha(reference.get("id")) or digest({key: child for key, child in reference.items() if key != "id"}) != reference["id"]:
        raise ValueError("selected image reference source checksum changed")
    required = {"reference", "requestId", "ownerTurnId", "scopeTurnId", "requestedUrl", "finalUrl", "redirects", "fetchedAt", "permissionRevision", "mimeType", "sha256", "bytes"}
    if (not isinstance(downloaded, dict) or not required <= set(downloaded) or set(downloaded) - required - {"learningPolicySha256"}
            or downloaded["reference"] != reference or downloaded["ownerTurnId"] != value["turnId"] or not _id(downloaded["requestId"])
            or downloaded["mimeType"] not in IMAGE_MIMES or not _sha(downloaded["sha256"]) or not _sha(downloaded["permissionRevision"])
            or "learningPolicySha256" in downloaded and not _sha(downloaded["learningPolicySha256"])
            or type(downloaded["bytes"]) is not int or not 0 < downloaded["bytes"] <= (1 << 53) - 1):
        raise ValueError("selected web image lost its exact download provenance")
    redirects = downloaded["redirects"]
    if (not isinstance(redirects, list) or not redirects or not all(_url(url) for url in redirects)
            or redirects[0] != reference["url"] or downloaded["requestedUrl"] != reference["url"] or redirects[-1] != downloaded["finalUrl"]
            or downloaded["scopeTurnId"] != "webimg-" + digest([value["brainId"], value["turnId"], downloaded["requestId"], reference["id"]])):
        raise ValueError("selected image no longer belongs to its checked redirect/task scope")
    if source["kind"] == "image-response" and downloaded["sha256"] != source["sha256"]:
        raise ValueError("selected image differs from the previously fetched exact pixels")
    required_attachment = {"brainId", "turnId", "attachmentId", "displayName", "kind", "mimeType", "contentSha256", "manifestSha256", "bytes", "relativePath"}
    if (not isinstance(attachment, dict) or set(attachment) != required_attachment or not _id(attachment["attachmentId"])
            or attachment["brainId"] != value["brainId"] or attachment["turnId"] != downloaded["scopeTurnId"] or attachment["kind"] != "image"
            or attachment["mimeType"] != downloaded["mimeType"] or attachment["bytes"] != downloaded["bytes"]
            or attachment["contentSha256"] != downloaded["sha256"] or not _sha(attachment["manifestSha256"])
            or attachment["relativePath"] != "attachments/blobs/" + downloaded["sha256"] + ".data"
            or not isinstance(attachment["displayName"], str)):
        raise ValueError("selected web image lost its exact owned attachment descriptor")
    if (reference["brainId"] != value["brainId"] or reference["turnId"] != value["turnId"]
            or brain_id is not None and value["brainId"] != brain_id or turn_id is not None and value["turnId"] != turn_id
            or value["observation"] != {"state": "pending-native-perception", "pixelsRead": False, "parametersChanged": False, "rawSourceTextInjected": False}
            or digest({key: child for key, child in value.items() if key not in {"selectionSha256", "observation"}}) != value["selectionSha256"]):
        raise ValueError("selected web image has another current native action/input owner")
    return value


def _check(brain, cancelled):
    if cancelled():
        raise InterruptedError("selected web image reading cancelled before admission")
    pressure = getattr(brain.resource_policy, "check_active_pressure", None)
    if callable(pressure):
        pressure("selected native web image")


def verify_registry(brain, selected):
    root = Path(brain.storage_path)
    registry = root / "attachments" / "registry.sqlite3"
    for path in (root, registry.parent, registry):
        if path.is_symlink() or (not path.is_file() if path == registry else not path.is_dir()):
            raise ValueError("selected image registry requires real owned custody")
    db = sqlite3.connect(registry.absolute().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA trusted_schema=OFF"); db.execute("PRAGMA query_only=ON"); db.execute("PRAGMA cache_size=-64"); db.execute("PRAGMA mmap_size=0")
        db.execute("BEGIN")
        if db.execute("SELECT value FROM meta WHERE key='brainId'").fetchone()[0] != brain.brain_id:
            raise ValueError("selected image registry has another current owner")
        attachment, downloaded, reference = selected["attachment"], selected["download"], selected["reference"]
        size = db.execute("SELECT length(CAST(source_json AS BLOB)) + length(CAST(display_name AS BLOB)) AS bytes FROM attachments WHERE id=?", (attachment["attachmentId"],)).fetchone()
        if size is None:
            raise ValueError("selected image attachment is no longer registered")
        with brain.resource_policy.reserve_ram(int(size["bytes"]) * 16 + 131072, "selected image registry source proof"):
            row = db.execute("SELECT * FROM attachments WHERE id=?", (attachment["attachmentId"],)).fetchone()
            if (row["brain_id"] != brain.brain_id or row["request_id"] != downloaded["requestId"] or row["state"] != "bound"
                    or row["turn_id"] != downloaded["scopeTurnId"] or row["content_sha256"] != downloaded["sha256"]
                    or row["bytes"] != downloaded["bytes"] or row["mime_type"] != downloaded["mimeType"] or row["kind"] != "image"
                    or row["manifest_sha256"] != attachment["manifestSha256"] or row["display_name"] != attachment["displayName"]
                    or hashlib.sha256(row["source_json"].encode()).hexdigest() != row["source_sha256"]):
                raise ValueError("selected image attachment/source row changed")
            source = json.loads(row["source_json"])
            if source.get("kind") != "web-image" or source.get("webImage") != downloaded or source.get("selectedAt") != downloaded["fetchedAt"]:
                raise ValueError("selected image no longer matches its immutable download source")
            manifest = {"format": "omni-chat-task-attachment", "version": 1, "id": row["id"], "brainId": row["brain_id"],
                "displayName": row["display_name"], "kind": row["kind"], "mimeType": row["mime_type"], "contentSha256": row["content_sha256"],
                "bytes": row["bytes"], "state": row["state"], "turnId": row["turn_id"], "createdAt": row["created_at"], "updatedAt": row["updated_at"]}
            if digest(manifest) != row["manifest_sha256"]:
                raise ValueError("selected image public task manifest checksum changed")
        size = db.execute("SELECT length(CAST(body AS BLOB)) FROM web_image_references WHERE id=?", (reference["id"],)).fetchone()
        if size is None:
            raise ValueError("selected image has no observed reference")
        with brain.resource_policy.reserve_ram(int(size[0]) * 16 + 131072, "selected image observed reference proof"):
            row = db.execute("SELECT * FROM web_image_references WHERE id=?", (reference["id"],)).fetchone()
            if (row["active"] != 1 or row["brain_id"] != brain.brain_id or row["turn_id"] != selected["turnId"]
                    or hashlib.sha256(row["body"].encode()).hexdigest() != row["sha256"] or json.loads(row["body"]) != reference):
                raise ValueError("selected web image is dormant or belongs to another original task")
        db.rollback()
    finally:
        db.close()


class _BorrowedReader:
    """A decoder may close its view, never the verified owner descriptor."""
    def __init__(self, handle):
        self._handle = handle
    def __getattr__(self, name):
        return getattr(self._handle, name)
    def close(self):
        pass


class OwnedWebImageSource:
    """Hash the opened file before any pixel decode; never copy/unlink raw data."""
    def __init__(self, brain, selected, cancelled):
        self.handle = self._file = None
        self.path = Path(brain.storage_path) / selected["attachment"]["relativePath"]
        self.sha256 = selected["download"]["sha256"]
        self.block = SimpleNamespace(mime_type=selected["download"]["mimeType"], decoded_bytes=selected["download"]["bytes"])
        try:
            for path in (Path(brain.storage_path), self.path.parent.parent, self.path.parent, self.path):
                if path.is_symlink():
                    raise ValueError("selected image custody contains a symbolic link")
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            self._file = os.fdopen(descriptor, "rb")
            self.handle = _BorrowedReader(self._file)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != self.block.decoded_bytes:
                raise ValueError("selected image file has another size/type")
            self.identity = self._identity(opened)
            checksum = hashlib.sha256()
            with brain.resource_policy.reserve_ram(_CHUNK * 3, "selected image exact opened-file hash"):
                while True:
                    _check(brain, cancelled)
                    piece = self.handle.read(_CHUNK)
                    if not piece:
                        break
                    checksum.update(piece)
                    self.verify_path()
            if checksum.hexdigest() != self.sha256:
                raise ValueError("selected image bytes differ from the approved source hash")
            self.handle.seek(0); self.verify_path()
        except BaseException:
            self.close(); raise

    @staticmethod
    def _identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def verify_path(self):
        opened, named = os.fstat(self.handle.fileno()), self.path.lstat()
        if (not stat.S_ISREG(named.st_mode) or self._identity(opened) != self.identity or self._identity(named) != self.identity):
            raise ValueError("selected image source was replaced or changed during inspection")
        return self.path

    def close(self):
        if self._file is not None:
            self._file.close(); self._file = self.handle = None


@contextmanager
def encode_web_image(brain, selected, *, turn_id=None, cancelled=lambda: False):
    selected = validate_selection(selected, brain_id=brain.brain_id, turn_id=turn_id)
    if not (brain.config.image_enabled or brain.config.vision_enabled):
        raise ValueError("this native brain has no enabled image perception region")
    verify_registry(brain, selected)
    source = OwnedWebImageSource(brain, selected, cancelled)
    dimensions, modes = int(brain.config.vsa_dim), [(module, module.training) for module in brain.modalities.modules()]
    pager = getattr(brain, "working_attention_pager", None)
    receipt = {"format": "omni-web-image-perception", "version": 1, "selectionSha256": selected["selectionSha256"],
        "referenceId": selected["reference"]["id"], "sourceSha256": source.sha256, "sourceBytes": source.block.decoded_bytes,
        "sourceFrames": 0, "decodedWindows": 0, "wholeInputTraversed": False, "nativeSensory": False,
        "rawSourceTextInjected": False, "parametersChanged": False}
    try:
        # Pillow header/metadata parsing itself may allocate before its pixel
        # expansion reservation. Admit that separately; raw data stays on disk.
        with brain.resource_policy.reserve_ram(source.block.decoded_bytes * 3 + dimensions * 48 + 131072, "selected image header and shared sensory activity"):
            total, correction = torch.zeros(dimensions), torch.zeros(dimensions)
            brain.modalities.eval()
            with torch.no_grad(), native_parameter_read_only(), pager.activate(cancelled) if pager is not None else nullcontext():
                windows = _image_windows(brain, source, receipt, cancelled)
                try:
                    for index, (tensor, elapsed) in enumerate(windows):
                        _check(brain, cancelled); source.verify_path()
                        with brain.resource_policy.reserve_ram(brain.config.image_size ** 2 * 128 + brain.config.idea_dim * 64 + dimensions * 64 + 131072, "same-brain selected image forward activity"):
                            embedding = brain.modalities.perception_embedding("image", tensor)
                            if embedding.shape != (1, brain.config.idea_dim) or not bool(torch.isfinite(embedding).all()):
                                raise ValueError("selected image returned invalid same-brain sensory activity")
                            vector = brain._sensory_substrate_vector(embedding, "image", source.sha256)
                            position = brain.memory.space.symbol("web-image-frame:%d" % index)
                            bound = brain.memory.space.permute(brain.memory.space.bind(vector, position), index + 1)
                            increment = bound * max(1e-6, float(elapsed)) - correction
                            updated = total + increment; correction = (updated - total) - increment; total = updated
                finally:
                    windows.close()
                _check(brain, cancelled); source.verify_path(); verify_registry(brain, selected)
                if receipt["decodedWindows"] < 1 or receipt["sourceFrames"] < 1 or not bool(torch.isfinite(total).all()) or float(total.norm()) <= 1e-8:
                    raise ValueError("selected image has no fully traversed same-brain pixel observation")
                receipt.update(wholeInputTraversed=True, nativeSensory=True)
                cue = F.normalize(total.reshape(1, -1), dim=-1)[0]
            for module, mode in modes:
                module.training = mode
            yield cue, receipt
    finally:
        for module, mode in modes:
            module.training = mode
        source.close()


def selected_tool_output(observation):
    if observation.get("toolId") != "web.fetch" or observation.get("action") != "inspect-image":
        return None
    payload = json.loads(observation["payloadJson"])
    if payload.get("executionState") == "failed" or payload.get("outputPresent") is not True:
        return None
    selected = validate_selection(payload.get("output"), brain_id=observation["brainId"], turn_id=observation["turnId"])
    if selected["download"]["requestId"] != digest(["web-image", observation["brainId"], observation["turnId"], observation["actionEventId"], selected["reference"]["id"]]):
        raise ValueError("selected image does not bind the actual native action event")
    return selected


def selection_from_evidence(brain, path, expected_sha256, evidence_id, execution_id):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("selected image action evidence must be a real owned file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(descriptor)
        if OwnedWebImageSource._identity(opened) != OwnedWebImageSource._identity(before):
            raise ValueError("selected image action evidence was replaced")
        checksum = hashlib.sha256()
        with brain.resource_policy.reserve_ram(_CHUNK * 3, "selected image exact action evidence hash"):
            for piece in iter(lambda: handle.read(_CHUNK), b""):
                checksum.update(piece)
        if checksum.hexdigest() != expected_sha256:
            raise ValueError("selected image action evidence checksum changed")
        with brain.resource_policy.reserve_ram(opened.st_size * 16 + 131072, "selected image action descriptor parse"):
            handle.seek(0); body = json.load(handle)
        if OwnedWebImageSource._identity(os.fstat(descriptor)) != OwnedWebImageSource._identity(opened) or OwnedWebImageSource._identity(path.lstat()) != OwnedWebImageSource._identity(opened):
            raise ValueError("selected image action evidence changed during reading")
    if (body.get("format") != "omni-completed-action-result-v1" or body.get("originBrainId") != brain.brain_id
            or body.get("evidenceId") != evidence_id or body.get("executionId") != execution_id
            or body.get("toolId") != "web.fetch" or body.get("action") != "inspect-image" or body.get("executionState") != "complete"
            or body.get("outputPresent") is not True or not _id(body.get("chatTurnId"))):
        raise ValueError("selected image has no exact completed action/source evidence")
    selected = validate_selection(body.get("output"), brain_id=brain.brain_id, turn_id=body["chatTurnId"])
    if (not isinstance(body.get("arguments"), dict) or body["arguments"].get("referenceId") != selected["reference"]["id"]
            or selected["download"]["requestId"] != digest(["web-image", brain.brain_id, body["chatTurnId"], body["actionEventId"], selected["reference"]["id"]])):
        raise ValueError("selected image evidence changed the actual action operands")
    return selected


@contextmanager
def image_learning_source(brain, selected):
    """Hold exact source/pixel-expansion admission over ordinary atomic ingest."""
    from PIL import Image
    selected = validate_selection(selected, brain_id=brain.brain_id)
    verify_registry(brain, selected)
    source = OwnedWebImageSource(brain, selected, lambda: False)
    try:
        with brain.resource_policy.reserve_ram(source.block.decoded_bytes * 3 + _CHUNK, "selected image training container metadata"):
            with Image.open(source.handle, formats=[IMAGE_MIMES[source.block.mime_type]]) as image:
                width, height = image.size
                if image.format != IMAGE_MIMES[source.block.mime_type] or width < 1 or height < 1:
                    raise ValueError("selected image container differs from its exact MIME")
                frames = int(getattr(image, "n_frames", 1))
            source.handle.seek(0)
            with brain.resource_policy.reserve_ram(width * height * 64 + brain.config.image_size ** 2 * 128, "selected image training decoder expansion"):
                if frames > 1 and not brain.config.video_enabled:
                    raise ValueError("animated selected image needs this same brain's enabled temporal region")
                yield source.verify_path(), "video" if frames > 1 else "image"
                source.verify_path(); verify_registry(brain, selected)
    finally:
        source.close()


def is_pillow_image_source(path):
    with open(path, "rb") as handle:
        header = handle.read(16)
    return header.startswith((b"GIF87a", b"GIF89a", b"\x89PNG\r\n\x1a\n")) or header[:4] == b"RIFF" and header[8:12] == b"WEBP"


def learn_selected_web_image_result(brain, *, evidence_id, execution_id, evidence_path, evidence_sha256, progress=None, merge_continuation=None):
    if merge_continuation is not None:
        raise ValueError("inherited selected-image authority is dormant; explicitly select the image under this identity")
    selected = selection_from_evidence(brain, evidence_path, evidence_sha256, evidence_id, execution_id)
    with image_learning_source(brain, selected) as (path, kind):
        result = brain.ingest(path=str(path), name="web-image-" + evidence_id, kind=kind, policy="pretrain",
            expected_hash=selected["download"]["sha256"], allow_replay=False, epoch=0,
            transaction_key=evidence_id, progress=progress)
    source = result.get("source", {})
    committed, coverage = source.get("completionReceipt", {}), source.get("media_coverage", {})
    if (not isinstance(committed, dict) or not _sha(committed.get("transactionId")) or not _sha(committed.get("sourceIdentity"))
            or committed.get("contentHash") != selected["download"]["sha256"] or source.get("modality_trained") is not True
            or not isinstance(coverage, dict) or coverage.get("complete") is not True or int(coverage.get("processedUnits", 0)) < 1):
        raise RuntimeError("selected image has no actual complete ordinary neural media learning receipt")
    duplicate = bool(result.get("duplicate"))
    receipt = {"format": "omni-web-image-learning", "version": 1, "selectionSha256": selected["selectionSha256"],
        "referenceId": selected["reference"]["id"], "sourceSha256": selected["download"]["sha256"], "sourceBytes": selected["download"]["bytes"],
        "committed": True, "duplicate": duplicate, "sourceIdentity": committed["sourceIdentity"], "transactionId": committed["transactionId"]}
    return {"brainId": brain.brain_id, "evidenceId": evidence_id, "executionId": execution_id, "evidenceSha256": evidence_sha256,
        "processed": not duplicate, "duplicate": duplicate, "committed": True, "sourceIdentity": committed["sourceIdentity"],
        "commitSequence": source.get("record_batch_commits", committed.get("batchCommits", 0)), "webImage": receipt,
        "source": source, "metrics": result["metrics"] if "metrics" in result else brain.metrics()}
