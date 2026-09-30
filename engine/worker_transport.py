"""Serial neural stdio dispatch with one narrow, flags-only control lane."""
import json
import os
import queue
import sys
import threading
import traceback


class ProtocolLineReader:
    """Bounded OS reads; a daemon never holds CPython's buffered-stdin lock."""
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.buffer = bytearray()

    def readline(self, size=-1):
        while True:
            newline = self.buffer.find(b"\n")
            length = newline + 1 if newline >= 0 else 0
            if size > 0 and (not length or length > size) and len(self.buffer) >= size:
                length = size
            if length:
                line = bytes(self.buffer[:length])
                del self.buffer[:length]
                return line
            remaining = max(1, size - len(self.buffer)) if size > 0 else 65536
            chunk = os.read(self.descriptor, min(65536, remaining))
            if not chunk:
                line = bytes(self.buffer)
                self.buffer.clear()
                return line
            self.buffer.extend(chunk)


def serve_worker_stdio(worker, stream, send, available_memory, rpc_fault_type):
    # The supervisor writes at most one normal RPC at a time. Bound ordinary
    # prefetch too; only the explicit control method bypasses neural dispatch.
    requests = queue.Queue(maxsize=1)
    eof = object()

    def dispatch(request, control=False):
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

    def read_requests():
        try:
            while worker.running:
                available = available_memory()
                limit = max(1, available // 8) if available is not None else None
                raw = stream.readline(limit + 1) if limit is not None else stream.readline()
                if not raw:
                    break
                if limit is not None and len(raw) > limit and not raw.endswith(b"\n"):
                    send({"jsonrpc": "2.0", "id": None, "error": {
                        "code": -32600, "message": "request line exceeds the current resource-derived protocol envelope"}})
                    break
                try:
                    request = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError) as error:
                    send({"jsonrpc": "2.0", "id": None, "error": {
                        "code": -32700, "message": "parse error: %s" % error}})
                    continue
                if isinstance(request, dict) and request.get("method") in {"cancel_inline_generation", "steer_chat", "resolve_codec_runtime", "cancel_artifact_request"}:
                    # dispatch_control validates the exact ownership tuple and
                    # cannot execute/load/save a brain or alter chat cancellation.
                    dispatch(request, control=True)
                else:
                    reserve = getattr(worker, "reserve_chat_steering", None)
                    if isinstance(request, dict) and request.get("method") == "chat" and callable(reserve):
                        try:
                            reserve(request)
                        except rpc_fault_type as error:
                            send({"jsonrpc": "2.0", "id": request.get("id"), "error": {
                                "code": error.code, "message": error.message}})
                            continue
                    requests.put(request)
        finally:
            requests.put(eof)

    reader = threading.Thread(target=read_requests, name="omni-stdio-control", daemon=True)
    reader.start()
    while worker.running:
        request = requests.get()
        if request is eof:
            break
        dispatch(request)
