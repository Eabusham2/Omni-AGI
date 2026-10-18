"""Harmless, task-owned 16-byte process-memory fixture. Not product code."""
import ctypes
import hashlib
import json
import os
import sys

buffer = ctypes.create_string_buffer(b"omni-fixture-old")
print(json.dumps({"pid": os.getpid(), "address": hex(ctypes.addressof(buffer)), "length": 16,
                  "sha256": hashlib.sha256(buffer.raw[:16]).hexdigest()}), flush=True)
if sys.stdin.readline().strip() == "inspect":
    print(json.dumps({"sha256": hashlib.sha256(buffer.raw[:16]).hexdigest(), "bytes": buffer.raw[:16].decode("ascii")}), flush=True)
