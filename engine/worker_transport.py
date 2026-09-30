"""Serial neural stdio dispatch with one narrow, flags-only control lane."""
import json
import os
import queue
import sys
import threading
import traceback
from collections import deque
from stdio_request_admission import (AdmittedRequest, RequestAllocationReceipt, RequestMemoryPressure,
    decoded_request_bound, request_id_hint, reserve_request_bytes)


class ProtocolLineReader:
    """Bounded OS reads; a daemon never holds CPython's buffered-stdin lock."""
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.chunks, self.buffered = deque(), 0
        self.reserve_memory = None
        self.line_lease = None

    def _line_length(self):
        total = 0
        for chunk, start in self.chunks:
            position = chunk.find(b"\n", start)
            if position >= 0: return total + position - start + 1
            total += len(chunk) - start
        return 0

    def _take(self, length):
        lease = reserve_request_bytes(self.reserve_memory, length * 2 + 256, "stdio complete raw frame")
        parts, remaining = [], length
        try:
            for chunk, start in self.chunks:
                count = min(remaining, len(chunk) - start)
                parts.append(chunk[start:start + count])
                remaining -= count
                if not remaining: break
            line = b"".join(parts)
            remaining = length
            while remaining:
                chunk, start = self.chunks.popleft()
                count = min(remaining, len(chunk) - start)
                if start + count < len(chunk): self.chunks.appendleft((chunk, start + count))
                remaining -= count
            self.buffered -= length
            self.line_lease = lease
            return line
        except BaseException:
            if lease is not None: lease.release()
            raise

    def readline(self, size=-1):
        while True:
            length = self._line_length()
            if size > 0 and (not length or length > size) and self.buffered >= size:
                length = size
            if length:
                return self._take(length)
            remaining = max(1, size - self.buffered) if size > 0 else 65536
            count = min(65536, remaining)
            lease = reserve_request_bytes(self.reserve_memory, count * 2 + 256, "stdio OS read chunk")
            try:
                chunk = os.read(self.descriptor, count)
                if chunk:
                    self.chunks.append((chunk, 0)); self.buffered += len(chunk)
                    if lease is not None: lease.mark_allocated(len(chunk) * 2 + 256)
            finally:
                if lease is not None: lease.release()
            if not chunk:
                return self._take(self.buffered) if self.buffered else b""

    def discard_line(self, prefix=b""):
        """Discard only the rejected frame; preserve bytes after its newline."""
        captured = bytearray(prefix[:4096])
        while True:
            while self.chunks:
                chunk, start = self.chunks.popleft(); self.buffered -= len(chunk) - start
                newline = chunk.find(b"\n", start)
                end = newline + 1 if newline >= 0 else len(chunk)
                if len(captured) < 4096: captured.extend(chunk[start:min(end, start + 4096 - len(captured))])
                if newline >= 0:
                    if newline + 1 < len(chunk):
                        self.chunks.appendleft((chunk, newline + 1)); self.buffered += len(chunk) - newline - 1
                    return bytes(captured)
            chunk = os.read(self.descriptor, 4096)  # covered by retained drain reserve
            if not chunk: return bytes(captured)
            self.chunks.append((chunk, 0)); self.buffered += len(chunk)

    def request_prefix(self):
        parts, remaining = [], 4096
        for chunk, start in self.chunks:
            count = min(remaining, len(chunk) - start)
            parts.append(chunk[start:start + count]); remaining -= count
            if not remaining: break
        return b"".join(parts)


def serve_worker_stdio(worker, stream, send, available_memory, rpc_fault_type,
                       reserve_request_memory=None, request_memory_headroom=None):
    # The supervisor writes at most one normal RPC at a time. Bound ordinary
    # prefetch too; only the explicit control method bypasses neural dispatch.
    requests = queue.Queue(maxsize=1)
    eof = object()

    def release_queued():
        while True:
            try: item = requests.get_nowait()
            except queue.Empty: return
            if item is not eof:
                _request, receipt = item
                receipt.release()
    if reserve_request_memory is None: reserve_request_memory = getattr(worker, "reserve_stdio_memory", None)
    if request_memory_headroom is None: request_memory_headroom = getattr(worker, "stdio_memory_headroom", None)
    if isinstance(stream, ProtocolLineReader): stream.reserve_memory = reserve_request_memory

    def dispatch(request, receipt, control=False):
        request_id = request.get("id") if isinstance(request, dict) else None
        try:
            response = (worker.dispatch_control(request) if control else worker.dispatch(request))
            if response is not None:
                send(response)
        except rpc_fault_type as error:
            payload = {"code": error.code, "message": error.message}
            if error.data is not None:
                payload["data"] = error.data
            send({"jsonrpc": "2.0", "id": request_id, "error": payload})
        except Exception as error:
            diagnostic = traceback.format_exc()
            sys.stderr.write(diagnostic)
            sys.stderr.flush()
            send({"jsonrpc": "2.0", "id": request_id, "error": {
                "code": -32000, "message": "%s: %s" % (type(error).__name__, error),
                "data": {"traceback": diagnostic[-8000:]}}})
        finally:
            receipt.release()

    def reject_frame(prefix, error):
        send({"jsonrpc": "2.0", "id": request_id_hint(prefix), "error": {
            "code": -32010, "message": str(error), "data": {"resourcePaused": True,
                "recoverable": True, "frameRejected": True, "inputTruncated": False,
                "workerRestartRequired": False, "status": getattr(error, "status", {})}}})

    def drain_rejected(prefix=b""):
        discard = getattr(stream, "discard_line", None)
        if callable(discard): return discard(prefix)
        captured = bytearray(prefix[:4096])
        while True:
            part = stream.readline(4096)
            if len(captured) < 4096: captured.extend(part[:4096 - len(captured)])
            if not part or part.endswith(b"\n"): return bytes(captured)

    def read_requests():
        drain_lease = None
        try:
            # Small fixed control/drain working set is reserved before normal
            # work starts, so a rejected large frame never disables its suffix.
            while worker.running and drain_lease is None and reserve_request_memory is not None:
                try:
                    drain_lease = reserve_request_bytes(reserve_request_memory, 65536, "stdio control/drain safety working set")
                except RequestMemoryPressure as error:
                    reject_frame(b"", error)
                    threading.Event().wait(0.25)
            while worker.running:
                available = available_memory()
                limit = max(1, available // 8) if available is not None else None
                if callable(request_memory_headroom):
                    selected = max(0, int(request_memory_headroom())) // 4
                    limit = min(limit, selected) if limit is not None else selected
                if limit is not None: limit = max(1, limit)
                try:
                    raw = stream.readline(limit + 1) if limit is not None else stream.readline()
                except (RequestMemoryPressure, MemoryError, OverflowError) as error:
                    prefix = drain_rejected()
                    reject_frame(prefix, error); continue
                if not raw:
                    break
                raw_lease = getattr(stream, "line_lease", None)
                if isinstance(stream, ProtocolLineReader): stream.line_lease = None
                if limit is not None and len(raw) > limit:
                    prefix = raw[:4096] if raw.endswith(b"\n") else drain_rejected(raw[:4096])
                    reject_frame(prefix, RequestMemoryPressure("request frame exceeds the selected-budget protocol envelope"))
                    if raw_lease is not None: RequestAllocationReceipt([raw_lease]).release()
                    continue
                try:
                    decode_lease = None
                    decode_lease = reserve_request_bytes(reserve_request_memory, decoded_request_bound(raw), "stdio decoded request")
                    request = json.loads(raw.decode("utf-8"))
                except RequestMemoryPressure as error:
                    reject_frame(raw[:4096], error)
                    if raw_lease is not None: RequestAllocationReceipt([raw_lease]).release()
                    continue
                except (ValueError, UnicodeError, RecursionError, MemoryError) as error:
                    RequestAllocationReceipt([raw_lease, decode_lease]).release()
                    send({"jsonrpc": "2.0", "id": None, "error": {
                        "code": -32700, "message": "parse error: %s" % error}})
                    continue
                receipt = RequestAllocationReceipt([raw_lease, decode_lease])
                if isinstance(request, dict): request = AdmittedRequest(request, receipt)
                if isinstance(request, dict) and request.get("method") in {"cancel_inline_generation", "steer_chat", "resolve_codec_runtime", "cancel_artifact_request", "observe_chat_action"}:
                    # dispatch_control validates the exact ownership tuple and
                    # cannot execute/load/save a brain or alter chat cancellation.
                    dispatch(request, receipt, control=True)
                else:
                    reserve = getattr(worker, "reserve_chat_steering", None)
                    if isinstance(request, dict) and request.get("method") == "chat" and callable(reserve):
                        try:
                            reserve(request)
                        except rpc_fault_type as error:
                            send({"jsonrpc": "2.0", "id": request.get("id"), "error": {
                                "code": error.code, "message": error.message}})
                            receipt.release()
                            continue
                    requests.put((request, receipt))
        finally:
            if drain_lease is not None: drain_lease.release()
            if worker.running: requests.put(eof)
            else: release_queued()

    reader = threading.Thread(target=read_requests, name="omni-stdio-control", daemon=True)
    reader.start()
    try:
        while worker.running:
            item = requests.get()
            if item is eof:
                break
            request, receipt = item
            dispatch(request, receipt)
    finally:
        release_queued()
