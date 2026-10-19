"""Frozen Windows transport classifier; stdlib only, never an AI/brain worker."""
import json
import sys

from worker_transport import ProtocolLineReader, serve_worker_stdio


class ProbeFault(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message, self.data = code, message, None


class ProbeWorker:
    def __init__(self):
        self.running = True

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("params"), dict):
            raise ProbeFault(-32600, "invalid stdio probe request")
        method = request.get("method")
        if method == "health":
            result = {"ready": True, "worker": "windows-stdio-probe", "protocolVersion": 1, "loadedBrains": 0}
        elif method == "shutdown":
            self.running = False
            result = {"stopping": True}
        else:
            raise ProbeFault(-32601, "stdio probe only supports health and shutdown")
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


def _send(message):
    sys.stdout.buffer.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main():
    serve_worker_stdio(ProbeWorker(), ProtocolLineReader(sys.stdin.buffer.fileno()), _send, lambda: None, ProbeFault)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
