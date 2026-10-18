"""Bounded, two-phase OS-permitted process memory adapter. Standard library only.

No debugger privilege, task signing, ptrace attach, page-protection change,
remote thread, target signal, process enumeration or physical-memory access.
Public OS ABI declarations below are independently wrapped, not OS code.
"""
from __future__ import annotations

import base64
import ctypes as c
import errno
import hashlib
import json
import os
import re
import secrets
import sys
from typing import Any, Iterator

FORMAT = "omni-process-memory-1"
MAX_BYTES = 1024 * 1024
MAX_LINE = 2 * 1024 * 1024
MAX_REGIONS = 200


class MemoryFailure(Exception):
    def __init__(self, state: str, message: str, code: int | None = None):
        super().__init__(message)
        self.state, self.code = state, code


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def whole(value: Any, maximum: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MemoryFailure("failed", "Invalid bounded integer operand.")
    return value


def address(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]{0,15})", value):
        raise MemoryFailure("failed", "Use an exact canonical hexadecimal address.")
    return int(value, 16)


def request(value: Any) -> dict[str, Any]:
    keys = {"format", "operationId", "operation", "pid", "access", "expectedIdentity", "address", "length",
            "dataBase64", "expectedBeforeSha256", "afterAddress", "regionLimit"}
    if not isinstance(value, dict) or set(value) - keys or value.get("format") != FORMAT:
        raise MemoryFailure("failed", "Invalid memory protocol request.")
    if not isinstance(value.get("operationId"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value["operationId"]):
        raise MemoryFailure("failed", "Invalid operation identity.")
    op = value.get("operation")
    if op not in {"attach", "regions", "read", "write"} or value.get("access") not in {"read", "read-write"}:
        raise MemoryFailure("unsupported", "Unsupported process-memory operation/access.")
    whole(value.get("pid"), 0x7fffffff, 2)
    if value["pid"] == os.getpid():
        raise MemoryFailure("protected", "The memory adapter cannot target its own authority process.")
    if op != "attach" and not isinstance(value.get("expectedIdentity"), dict):
        raise MemoryFailure("failed", "A process-creation identity is required.")
    expected = value.get("expectedIdentity")
    if expected is not None and (set(expected) != {"platform", "pid", "creation", "executable", "executableIdentity"}
                                or expected.get("pid") != value["pid"]):
        raise MemoryFailure("failed", "Invalid expected process identity.")
    if op in {"read", "write"}:
        start, count = address(value.get("address")), whole(value.get("length"), MAX_BYTES)
        if start + count > 1 << 64:
            raise MemoryFailure("failed", "Process address range overflows.")
    if op == "write":
        if value["access"] != "read-write" or not isinstance(value.get("expectedBeforeSha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", value["expectedBeforeSha256"]):
            raise MemoryFailure("failed", "Writes require read-write access and the exact preimage hash.")
        try:
            data = base64.b64decode(value.get("dataBase64", ""), validate=True)
        except (ValueError, TypeError) as error:
            raise MemoryFailure("failed", "Write data is not canonical base64.") from error
        if len(data) != value["length"] or base64.b64encode(data).decode() != value["dataBase64"]:
            raise MemoryFailure("failed", "Write data count does not match its exact request.")
    if op == "regions":
        address(value.get("afterAddress", "0x0"))
        whole(value.get("regionLimit", 50), MAX_REGIONS)
    allowed = {"format", "operationId", "operation", "pid", "access", "expectedIdentity"}
    allowed |= {"address", "length"} if op in {"read", "write"} else set()
    allowed |= {"dataBase64", "expectedBeforeSha256"} if op == "write" else set()
    allowed |= {"afterAddress", "regionLimit"} if op == "regions" else set()
    if set(value) - allowed:
        raise MemoryFailure("failed", "Extraneous memory operands are not permitted.")
    return value


def os_failure(error: OSError) -> MemoryFailure:
    code = error.errno
    state = "denied" if code in {errno.EPERM, errno.EACCES} else "identity-changed" if code in {errno.ESRCH, errno.ENOENT} else "failed"
    return MemoryFailure(state, str(error), code)


def region(start: int, end: int, read: bool, write: bool, execute: bool) -> dict[str, Any]:
    return {"start": hex(start), "end": hex(end), "readable": bool(read), "writable": bool(write), "executable": bool(execute)}


class Target:
    def __init__(self, pid: int, access: str):
        self.pid, self.access = pid, access

    def close(self) -> None:
        pass

    def ensure_range(self, start: int, length: int, write: bool = False) -> None:
        cursor, end = start, start + length
        for item in self.regions(start):
            a, b = address(item["start"]), address(item["end"])
            if b <= cursor:
                continue
            if a > cursor or not item["readable"] or (write and not item["writable"]):
                raise MemoryFailure("protected", "The exact range is not currently readable/writable; protections were not changed.")
            cursor = b
            if cursor >= end:
                return
        raise MemoryFailure("protected", "The requested range contains unmapped memory.")

    def verify(self, expected: dict[str, Any]) -> None:
        if self.identity() != expected:
            raise MemoryFailure("identity-changed", "The selected process incarnation or executable changed.")


class LinuxTarget(Target):
    def __init__(self, pid: int, access: str):
        super().__init__(pid, access)
        self.directory = self.memory = -1
        try:
            self.directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            self.initial = self.identity()
            self.memory = os.open("mem", (os.O_RDWR if access == "read-write" else os.O_RDONLY) | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=self.directory)
            self.verify(self.initial)
        except Exception:
            self.close()
            raise

    def text(self, name: str, limit: int) -> str:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
        try:
            data = os.read(fd, limit + 1)
            if len(data) > limit:
                raise MemoryFailure("failed", "Selected-process metadata exceeded its bounded envelope.")
            return data.decode("utf-8", "strict")
        finally:
            os.close(fd)

    def identity(self) -> dict[str, Any]:
        stat = self.text("stat", 65536)
        split = stat.rfind(")")
        fields = stat[split + 2:].split()
        if split < 0 or len(fields) < 20 or fields[0] in {"Z", "X", "x"}:
            raise MemoryFailure("identity-changed", "Selected process has exited or has invalid identity metadata.")
        executable = os.readlink("exe", dir_fd=self.directory)
        info = os.stat("exe", dir_fd=self.directory)
        # A pinned proc directory and mem FD retain the original object; PID reuse
        # cannot redirect a write. No global process directory is enumerated.
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="ascii") as boot:
            boot_id = boot.read(64).strip()
        return {"platform": "linux", "pid": self.pid, "creation": f"{boot_id}:{fields[19]}",
                "executable": executable, "executableIdentity": f"{info.st_dev}:{info.st_ino}"}

    def regions(self, after: int) -> Iterator[dict[str, Any]]:
        fd = os.open("maps", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
        try:
            with os.fdopen(fd, "r", encoding="utf-8", errors="strict", closefd=False) as stream:
                for line in stream:
                    if len(line) > 65536:
                        raise MemoryFailure("failed", "A selected-process region entry exceeds its envelope.")
                    fields = line.split(None, 2)
                    a, b = (int(part, 16) for part in fields[0].split("-"))
                    if b > after:
                        flags = fields[1]
                        yield region(a, b, "r" in flags, "w" in flags, "x" in flags)
        finally:
            os.close(fd)

    def read(self, start: int, length: int) -> tuple[bytes, int | None]:
        if start + length > 1 << 63:
            raise MemoryFailure("unsupported", "This proc-mem adapter requires a signed 64-bit offset.")
        return os.pread(self.memory, length, start), None

    def write(self, start: int, data: bytes) -> tuple[int, int | None]:
        if start + len(data) > 1 << 63:
            raise MemoryFailure("unsupported", "This proc-mem adapter requires a signed 64-bit offset.")
        return os.pwrite(self.memory, data, start), None

    def close(self) -> None:
        for key in ("memory", "directory"):
            fd = getattr(self, key, -1)
            if fd >= 0:
                os.close(fd)
                setattr(self, key, -1)


class DarwinTarget(Target):
    def __init__(self, pid: int, access: str):
        super().__init__(pid, access)
        self.port = c.c_uint32(0)
        self.lib = c.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.proc = c.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.mach_task_self.restype = c.c_uint32
        self.self_port = self.lib.mach_task_self()
        self.lib.task_for_pid.argtypes = [c.c_uint32, c.c_int32, c.POINTER(c.c_uint32)]
        self.lib.pid_for_task.argtypes = [c.c_uint32, c.POINTER(c.c_int32)]
        self.lib.mach_port_deallocate.argtypes = [c.c_uint32, c.c_uint32]
        self.proc.proc_pidinfo.argtypes = [c.c_int, c.c_int, c.c_uint64, c.c_void_p, c.c_int]
        self.proc.proc_pidpath.argtypes = [c.c_int, c.c_void_p, c.c_uint32]
        self.lib.mach_vm_read_overwrite.argtypes = [c.c_uint32, c.c_uint64, c.c_uint64, c.c_uint64, c.POINTER(c.c_uint64)]
        self.lib.mach_vm_write.argtypes = [c.c_uint32, c.c_uint64, c.c_void_p, c.c_uint32]
        self.lib.mach_vm_region.argtypes = [c.c_uint32, c.POINTER(c.c_uint64), c.POINTER(c.c_uint64), c.c_int,
                                          c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32)]
        initial = self.identity()
        code = self.lib.task_for_pid(self.self_port, pid, c.byref(self.port))
        if code:
            raise MemoryFailure("denied", "macOS did not grant a task port. No signing, entitlement or OS-protection changes were attempted.", code)
        try:
            current = c.c_int32()
            code = self.lib.pid_for_task(self.port, c.byref(current))
            if code or current.value != pid:
                raise MemoryFailure("identity-changed", "Acquired Mach task port does not identify the requested process.", code)
            self.verify(initial)
        except Exception:
            self.close()
            raise

    def identity(self) -> dict[str, Any]:
        class BsdInfo(c.Structure):
            _fields_ = [("prefix", c.c_uint32 * 12), ("comm", c.c_char * 16), ("name", c.c_char * 32),
                        ("middle", c.c_uint32 * 5), ("nice", c.c_int32), ("seconds", c.c_uint64), ("micros", c.c_uint64)]
        info, path = BsdInfo(), c.create_string_buffer(4096)
        if self.proc.proc_pidinfo(self.pid, 3, 0, c.byref(info), c.sizeof(info)) != c.sizeof(info):
            raise os_failure(OSError(c.get_errno(), "Cannot read the exact selected-process identity."))
        if info.prefix[3] != self.pid or info.prefix[0] & 5:
            raise MemoryFailure("protected", "System/exiting processes are not supported targets.")
        if self.proc.proc_pidpath(self.pid, path, len(path)) <= 0:
            raise os_failure(OSError(c.get_errno(), "Cannot identify the selected executable."))
        executable = os.fsdecode(path.value)
        item = os.stat(executable)
        return {"platform": "darwin", "pid": self.pid, "creation": f"{info.seconds}:{info.micros}",
                "executable": executable, "executableIdentity": f"{item.st_dev}:{item.st_ino}"}

    def regions(self, after: int) -> Iterator[dict[str, Any]]:
        class BasicInfo(c.Structure):
            _pack_ = 4
            _fields_ = [("protection", c.c_int32), ("maximum", c.c_int32), ("inheritance", c.c_int32),
                        ("shared", c.c_uint32), ("reserved", c.c_uint32), ("offset", c.c_uint64),
                        ("behavior", c.c_int32), ("wired", c.c_uint16)]
        cursor = after
        while cursor < (1 << 64) - 1:
            start, size, info = c.c_uint64(cursor), c.c_uint64(), BasicInfo()
            count, object_port = c.c_uint32(c.sizeof(info) // 4), c.c_uint32()
            code = self.lib.mach_vm_region(self.port, c.byref(start), c.byref(size), 9, c.byref(info), c.byref(count), c.byref(object_port))
            if object_port.value:
                self.lib.mach_port_deallocate(self.self_port, object_port)
            if code == 1:  # KERN_INVALID_ADDRESS: no following mapped region.
                return
            if code:
                raise MemoryFailure("protected" if code == 2 else "denied", "Mach region inspection was rejected.", code)
            end = start.value + size.value
            if not size.value or end <= cursor or end >= 1 << 64:
                raise MemoryFailure("failed", "Mach returned a nonadvancing region.")
            yield region(start.value, end, bool(info.protection & 1), bool(info.protection & 2), bool(info.protection & 4))
            cursor = end

    def read(self, start: int, length: int) -> tuple[bytes, int | None]:
        data, count = c.create_string_buffer(length), c.c_uint64()
        code = self.lib.mach_vm_read_overwrite(self.port, start, length, c.addressof(data), c.byref(count))
        if count.value > length:
            raise MemoryFailure("failed", "Mach returned an invalid byte count.")
        return data.raw[:count.value], code or None

    def write(self, start: int, data: bytes) -> tuple[int | None, int | None]:
        source = c.create_string_buffer(data)
        code = self.lib.mach_vm_write(self.port, start, source, len(data))
        # mach_vm_write has no partial byte-count output on failure.
        return (len(data), None) if not code else (None, code)

    def close(self) -> None:
        if self.port.value:
            self.lib.mach_port_deallocate(self.self_port, self.port)
            self.port.value = 0


class WindowsTarget(Target):
    def __init__(self, pid: int, access: str):
        super().__init__(pid, access)
        if c.sizeof(c.c_void_p) != 8:
            raise MemoryFailure("unsupported", "The Windows memory adapter requires a 64-bit host helper.")
        self.lib = c.WinDLL("kernel32", use_last_error=True)
        self.lib.OpenProcess.argtypes, self.lib.OpenProcess.restype = [c.c_uint32, c.c_int, c.c_uint32], c.c_void_p
        self.lib.CloseHandle.argtypes = [c.c_void_p]
        self.lib.GetProcessTimes.argtypes = [c.c_void_p] + [c.c_void_p] * 4
        self.lib.QueryFullProcessImageNameW.argtypes = [c.c_void_p, c.c_uint32, c.c_wchar_p, c.POINTER(c.c_uint32)]
        self.lib.GetExitCodeProcess.argtypes = [c.c_void_p, c.POINTER(c.c_uint32)]
        self.lib.ReadProcessMemory.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t)]
        self.lib.WriteProcessMemory.argtypes = self.lib.ReadProcessMemory.argtypes
        self.lib.VirtualQueryEx.argtypes, self.lib.VirtualQueryEx.restype = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t], c.c_size_t
        rights = 0x0400 | 0x1000 | 0x0010  # query, limited query, VM read
        if access == "read-write":
            rights |= 0x0020 | 0x0008  # VM write and operation only
        self.handle = self.lib.OpenProcess(rights, 0, pid)
        if not self.handle:
            raise MemoryFailure("denied", "Windows did not grant the requested process handle. No privilege was enabled.", c.get_last_error())
        try:
            self.identity()
        except Exception:
            self.close()
            raise

    def identity(self) -> dict[str, Any]:
        created, exited, kernel, user = (c.c_uint64() for _ in range(4))
        state, capacity, path = c.c_uint32(), c.c_uint32(32768), c.create_unicode_buffer(32768)
        if not self.lib.GetExitCodeProcess(self.handle, c.byref(state)) or state.value != 259:
            raise MemoryFailure("identity-changed", "The selected Windows process has exited.", c.get_last_error())
        if not self.lib.GetProcessTimes(self.handle, c.byref(created), c.byref(exited), c.byref(kernel), c.byref(user)) or not self.lib.QueryFullProcessImageNameW(self.handle, 0, path, c.byref(capacity)):
            raise MemoryFailure("denied", "Windows rejected process-identity inspection.", c.get_last_error())
        item = os.stat(path.value)
        return {"platform": "win32", "pid": self.pid, "creation": str(created.value), "executable": path.value,
                "executableIdentity": f"{item.st_dev}:{item.st_ino}"}

    def regions(self, after: int) -> Iterator[dict[str, Any]]:
        class BasicInfo(c.Structure):
            _fields_ = [("base", c.c_void_p), ("allocation", c.c_void_p), ("allocation_protection", c.c_uint32),
                        ("partition", c.c_uint16), ("size", c.c_size_t), ("state", c.c_uint32),
                        ("protection", c.c_uint32), ("kind", c.c_uint32)]
        cursor = after
        while cursor < (1 << 63):
            info = BasicInfo()
            if not self.lib.VirtualQueryEx(self.handle, c.c_void_p(cursor), c.byref(info), c.sizeof(info)):
                code = c.get_last_error()
                if code == 87:
                    return  # beyond supported user address space
                raise MemoryFailure("denied", "Windows rejected region inspection.", code)
            start, end = info.base or 0, (info.base or 0) + info.size
            if end <= cursor:
                raise MemoryFailure("failed", "Windows returned a nonadvancing region.")
            p, committed = info.protection & 0xff, info.state == 0x1000 and not (info.protection & 0x100)
            yield region(start, end, committed and p in {2, 4, 8, 32, 64, 128}, committed and p in {4, 8, 64, 128}, committed and p in {16, 32, 64, 128})
            cursor = end

    def read(self, start: int, length: int) -> tuple[bytes, int | None]:
        data, count = c.create_string_buffer(length), c.c_size_t()
        okay = self.lib.ReadProcessMemory(self.handle, c.c_void_p(start), data, length, c.byref(count))
        if count.value > length:
            raise MemoryFailure("failed", "Windows returned an invalid read byte count.")
        return data.raw[:count.value], None if okay else c.get_last_error()

    def write(self, start: int, data: bytes) -> tuple[int, int | None]:
        source, count = c.create_string_buffer(data), c.c_size_t()
        okay = self.lib.WriteProcessMemory(self.handle, c.c_void_p(start), source, len(data), c.byref(count))
        return count.value, None if okay else c.get_last_error()

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.CloseHandle(self.handle)
            self.handle = None


def open_target(pid: int, access: str) -> Target:
    kind = {"linux": LinuxTarget, "darwin": DarwinTarget, "win32": WindowsTarget}.get(sys.platform)
    if not kind:
        raise MemoryFailure("unsupported", "This operating system has no process-memory adapter.")
    return kind(pid, access)


def receipt_for(req: dict[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "operationId": req["operationId"], "operation": req["operation"],
            "requestSha256": sha(canonical(req).encode("utf-8")), "state": "failed", "requestedBytes": req.get("length", 0),
            "transferredBytes": 0, "writeMayHaveOccurred": False, "retrySafe": True, "compareAtomic": False}


def perform(target: Target, req: dict[str, Any], identity: dict[str, Any], result: dict[str, Any]) -> None:
    target.verify(identity)
    op = req["operation"]
    if op == "attach":
        result["state"] = "complete"
    elif op == "regions":
        rows, limit = [], req.get("regionLimit", 50)
        iterator = target.regions(address(req.get("afterAddress", "0x0")))
        for row in iterator:
            if len(rows) == limit:
                result["endOfRegions"] = False
                break
            # Paging starts at a previous end; never silently skip a mapping.
            rows.append(row)
        else:
            result["endOfRegions"] = True
        result.update(state="complete", regions=rows, nextAddress=rows[-1]["end"] if rows else req.get("afterAddress", "0x0"))
    else:
        start, length = address(req["address"]), req["length"]
        target.ensure_range(start, length, op == "write")
        target.verify(identity)
        before, code = target.read(start, length)
        if op == "read":
            result.update(state="complete" if len(before) == length and code is None else "partial",
                          transferredBytes=len(before), dataBase64=base64.b64encode(before).decode(), dataSha256=sha(before))
            if code is not None:
                result.update(osCode=code, error="The OS did not complete the entire read.")
        else:
            result["beforeSha256"] = sha(before)
            if code is not None or len(before) != length:
                raise MemoryFailure("protected", "The complete write preimage could not be read.", code)
            if result["beforeSha256"] != req["expectedBeforeSha256"]:
                raise MemoryFailure("preimage-mismatch", "Memory differs from the exact approved preimage; no write was attempted.")
            data = base64.b64decode(req["dataBase64"], validate=True)
            target.ensure_range(start, length, True)
            target.verify(identity)
            # This is a bounded debugger write, NOT an atomic remote compare-and-swap.
            result.update(writeMayHaveOccurred=True, retrySafe=False, transferredBytes=None, state="write-uncertain")
            count, code = target.write(start, data)
            result.update(transferredBytes=count, state="complete" if count == length and code is None else "partial" if count is not None else "write-uncertain")
            if code is not None:
                result.update(osCode=code, error="The OS did not confirm a complete write; do not blindly retry.")
            observed, observed_code = target.read(start, length)
            if len(observed) == length and observed_code is None:
                result["afterSha256"] = sha(observed)
                if count == length and observed != data:
                    result.update(state="partial", error="Write completed, but concurrent target activity changed the readback.")
    target.verify(identity)


def read_line(stream: Any) -> dict[str, Any]:
    raw = stream.readline(MAX_LINE + 1)
    if not raw or len(raw) > MAX_LINE or not raw.endswith(b"\n"):
        raise MemoryFailure("cancelled", "No complete bounded host authorization was received.")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise MemoryFailure("failed", "Protocol messages must be objects.")
    return value


def emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical(value) + "\n").encode("utf-8"))
    stream.flush()


def serve(stream_in: Any, stream_out: Any, target_factory: Any = open_target) -> int:
    target, result = None, None
    try:
        req = request(read_line(stream_in))
        result = receipt_for(req)
        target = target_factory(req["pid"], req["access"])
        identity = target.identity()
        if req.get("expectedIdentity") is not None:
            target.verify(req["expectedIdentity"])
        result["identity"] = identity
        nonce = secrets.token_hex(24)
        emit(stream_out, {"format": FORMAT, "type": "ready", "operationId": req["operationId"],
                          "requestSha256": result["requestSha256"], "nonce": nonce, "identity": identity})
        approval = read_line(stream_in)
        if approval != {"commit": nonce, "requestSha256": result["requestSha256"]}:
            raise MemoryFailure("cancelled", "The host did not authorize this exact prepared operation.")
        perform(target, req, identity, result)
    except Exception as error:
        failure = error if isinstance(error, MemoryFailure) else os_failure(error) if isinstance(error, OSError) else MemoryFailure("failed", str(error))
        if result is None:
            emit(stream_out, {"format": FORMAT, "type": "protocol-error", "state": failure.state, "error": str(failure)[:4096]})
            return 2
        # Once the write boundary has started, loss of certainty is never
        # relabeled as cancellation/no effect, including an identity race.
        result.update(state="write-uncertain" if result["writeMayHaveOccurred"] and result["transferredBytes"] is None else failure.state,
                      error=str(failure)[:4096])
        if req["operation"] == "read" and failure.state == "identity-changed":
            result.pop("dataBase64", None)
        if failure.code is not None:
            result["osCode"] = failure.code
    finally:
        if target is not None:
            target.close()
    emit(stream_out, result)
    return 0


def main() -> int:
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
