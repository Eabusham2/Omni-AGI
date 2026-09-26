#!/usr/bin/env python3
"""Deterministic host gateway used by Android and iOS companion CI smoke tests."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOKEN = "A" * 43


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write((format % args) + "\n")
        sys.stdout.flush()

    def json_response(self, status: int, value: object) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        return self.headers.get("authorization") == f"Bearer {TOKEN}"

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/health":
            self.json_response(200, {"schemaVersion": 1, "status": "ready"})
            return
        if not self.authorized():
            self.json_response(401, {"error": "Pair this device first."})
            return
        if self.path == "/v1/brains":
            self.json_response(
                200,
                {
                    "schemaVersion": 1,
                    "brains": [
                        {
                            "id": "brain-1",
                            "name": "Emulator persistent mind",
                            "updatedAt": "2026-08-23T00:00:00.000Z",
                            "preset": "whole-brain",
                            "generation": 1,
                            "readiness": "ready",
                        }
                    ],
                },
            )
            return
        if self.path == "/v1/brains/brain-1/messages":
            self.json_response(
                200,
                {
                    "schemaVersion": 1,
                    "messages": [
                        {
                            "id": "existing-1",
                            "role": "brain",
                            "content": "Existing persisted conversation",
                            "createdAt": "2026-08-23T00:00:00.000Z",
                        }
                    ],
                },
            )
            return
        self.json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/pair":
            body = self.read_json()
            if body.get("code") != "123456":
                self.json_response(401, {"error": "The pairing code is invalid or expired."})
                return
            self.json_response(
                201,
                {
                    "schemaVersion": 1,
                    "protocolVersion": 1,
                    "token": TOKEN,
                    "device": {
                        "id": str(uuid.uuid4()),
                        "name": body.get("deviceName", "emulator"),
                    },
                },
            )
            return
        if not self.authorized():
            self.json_response(401, {"error": "Pair this device first."})
            return
        if self.path == "/v1/brains/brain-1/chat":
            body = self.read_json()
            turn_id = str(body.get("turnId", "turn-1"))
            frames = [
                {
                    "type": "stream",
                    "event": {
                        "id": "state-1",
                        "brainId": "brain-1",
                        "turnId": turn_id,
                        "sequence": 0,
                        "createdAt": "2026-08-23T00:00:01.000Z",
                        "type": "chat-state",
                        "state": "started",
                    },
                },
                {
                    "type": "stream",
                    "event": {
                        "id": "token-1",
                        "brainId": "brain-1",
                        "turnId": turn_id,
                        "sequence": 1,
                        "createdAt": "2026-08-23T00:00:02.000Z",
                        "type": "chat-token",
                        "delta": "Same-brain emulator reply",
                    },
                },
                {
                    "type": "result",
                    "turnId": turn_id,
                    "brainMessage": {
                        "id": "reply-1",
                        "role": "brain",
                        "content": "Same-brain emulator reply",
                        "createdAt": "2026-08-23T00:00:02.000Z",
                    },
                },
            ]
            payload = "".join(json.dumps(frame) + "\n" for frame in frames).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/v1/brains/brain-1/experience":
            length = int(self.headers.get("content-length", "0"))
            digest = hashlib.sha256()
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            received = length - remaining
            self.json_response(
                201,
                {
                    "schemaVersion": 1,
                    "fileName": "emulator-audio.raw",
                    "bytes": received,
                    "sha256": digest.hexdigest(),
                    "results": [{"sourceId": "mobile-emulator-source"}],
                },
            )
            return
        if self.path.endswith("/cancel"):
            self.read_json()
            self.json_response(200, {"schemaVersion": 1, "cancelled": 1})
            return
        self.json_response(404, {"error": "Not found"})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 41837
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"fake Omni gateway listening on {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
