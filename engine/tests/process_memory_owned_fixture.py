"""One bounded real OS access attempt against our own child, never other PIDs.

OS denial is reported separately; it is not a successful read/write proof.
The caller supplies an outer timeout. No privilege/protection changes occur.
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("process_memory_helper", Path(__file__).parents[1] / "process_memory_helper.py")
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)
start = time.monotonic()
child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("process_memory_owned_child.py"))],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
target = None
try:
    source = json.loads(child.stdout.readline())
    if source["pid"] != child.pid or source["length"] != 16:
        raise RuntimeError("Owned child did not report the exact expected tiny buffer.")
    target = memory.open_target(child.pid, "read-write")
    identity = target.identity()
    req = {"format": memory.FORMAT, "operationId": "actual-owned-read", "operation": "read", "pid": child.pid,
           "access": "read-write", "expectedIdentity": identity, "address": source["address"], "length": source["length"]}
    read = memory.receipt_for(req); read["identity"] = identity
    memory.perform(target, memory.request(req), identity, read)
    if read["state"] != "complete" or read["dataSha256"] != source["sha256"]:
        raise RuntimeError("Actual owned read did not match the child source hash.")
    write_request = {**req, "operationId": "actual-owned-write", "operation": "write",
                     "dataBase64": memory.base64.b64encode(b"omni-fixture-new").decode(), "expectedBeforeSha256": source["sha256"]}
    write = memory.receipt_for(write_request); write["identity"] = identity
    memory.perform(target, memory.request(write_request), identity, write)
    child.stdin.write("inspect\n"); child.stdin.flush()
    observed = json.loads(child.stdout.readline())
    expected = memory.sha(b"omni-fixture-new")
    if write["state"] != "complete" or write["transferredBytes"] != 16 or observed["sha256"] != expected or write.get("afterSha256") != expected:
        raise RuntimeError("Actual bounded write/readback did not match the child observation.")
    print(json.dumps({"status": "actual-owned-read-write-passed", "platform": sys.platform, "pid": child.pid,
                      "bytes": 16, "beforeSha256": source["sha256"], "afterSha256": expected,
                      "wallSeconds": round(time.monotonic() - start, 6), "neuralRuntimeImported": "torch" in sys.modules,
                      "otherProcessesInspected": 0, "privilegeOrProtectionChanges": False}))
except memory.MemoryFailure as error:
    print(json.dumps({"status": "actual-owned-access-not-proven", "state": error.state, "osCode": error.code,
                      "detail": str(error), "wallSeconds": round(time.monotonic() - start, 6),
                      "platform": sys.platform, "pid": child.pid, "otherProcessesInspected": 0,
                      "privilegeOrProtectionChanges": False}))
finally:
    if target is not None:
        target.close()
    if child.poll() is None:
        child.stdin.close()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.terminate(); child.wait(timeout=1)
