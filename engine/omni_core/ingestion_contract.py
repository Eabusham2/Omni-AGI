"""Source-free transaction identities and attributable media input failures."""

import hashlib
import re


class MediaDecodeError(ValueError):
    """Invalid source bytes, not a learner, allocator, or runtime failure."""


class NoAudioTrackError(MediaDecodeError):
    """The decoder explicitly verified that a video has no audio stream."""


def ingestion_transaction_id(
    *,
    content_hash: str,
    epoch: int,
    policy: str,
    resolved_kind: str,
    source_bytes: int,
    transaction_key: str = "",
) -> str:
    if not isinstance(transaction_key, str):
        raise ValueError("ingestion transaction key must be a lowercase sha256")
    if transaction_key:
        if not re.fullmatch(r"[a-f0-9]{64}", transaction_key):
            raise ValueError("ingestion transaction key must be a lowercase sha256")
        # Main's key binds manifest, entry, epoch, content and policy. A new
        # requested run must not alias an older content-only completion receipt.
        return transaction_key
    return hashlib.sha256(
        ("%s\0%d\0%s\0%s\0%d" % (
            content_hash, max(0, int(epoch)), policy, resolved_kind, source_bytes
        )).encode("utf-8")
    ).hexdigest()
