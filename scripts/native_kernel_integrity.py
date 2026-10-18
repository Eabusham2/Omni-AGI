"""Build-only Mach-O custody across relocation and code signing; stdlib only.

Runtime verification still hashes the entire final file. This fingerprint is
only a build proof that re-signing/rebasing the dylib's own ID and loader rpath
did not replace its machine payload, dependencies or functional load commands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct


def macho_payload(path, *, raw=None):
    raw = Path(path).read_bytes() if raw is None else raw
    if len(raw) < 32:
        raise ValueError("native kernel Mach-O header is truncated")
    magic, cpu, subtype, filetype, commands, command_bytes, flags, reserved = struct.unpack_from("<IiiIIIII", raw)
    if magic != 0xFEEDFACF or cpu not in (0x1000007, 0x100000C) or filetype != 6:
        raise ValueError("native kernel must be a thin supported 64-bit Mach-O dylib")
    end, offset, records, data_offsets, signature, rpaths = 32 + command_bytes, 32, [], [], None, []
    if not commands or commands > command_bytes // 8 or end > len(raw):
        raise ValueError("native kernel Mach-O commands are invalid")
    def text(command, start):
        if not 0 <= start < len(command) or b"\0" not in command[start:]:
            raise ValueError("native kernel Mach-O string is invalid")
        return command[start:].split(b"\0", 1)[0].decode("utf-8")
    for _ in range(commands):
        if offset + 8 > end:
            raise ValueError("native kernel Mach-O command is truncated")
        kind, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or size % 8 or offset + size > end:
            raise ValueError("native kernel Mach-O command size is invalid")
        command = bytearray(raw[offset:offset + size]); offset += size
        if kind == 0x1D:  # LC_CODE_SIGNATURE: external cryptographic envelope.
            if signature is not None or size != 16:
                raise ValueError("native kernel Mach-O signature command is invalid")
            signature = struct.unpack_from("<II", command, 8)
        elif kind == 0x8000001C:  # LC_RPATH; only the known PyInstaller relative path.
            if size < 16:
                raise ValueError("native kernel Mach-O rpath is invalid")
            value = text(command, struct.unpack_from("<I", command, 8)[0])
            if value != "@loader_path/../.." or value in rpaths:
                raise ValueError("native kernel Mach-O contains an unexpected rpath")
            rpaths.append(value)
        elif kind == 0xD:  # LC_ID_DYLIB: normalize only this library's own path.
            if size < 24:
                raise ValueError("native kernel Mach-O ID is invalid")
            name = text(command, struct.unpack_from("<I", command, 8)[0])
            if Path(name).name != "omni_packed_kernel.dylib":
                raise ValueError("native kernel Mach-O ID names a foreign library")
            records.append([kind, command[12:24].hex(), "omni_packed_kernel.dylib"])
        else:
            if kind == 0x19:  # LC_SEGMENT_64.
                if size < 72:
                    raise ValueError("native kernel Mach-O segment is truncated")
                segment = bytes(command[8:24]).split(b"\0", 1)[0]
                sections = struct.unpack_from("<I", command, 64)[0]
                if 72 + sections * 80 != size:
                    raise ValueError("native kernel Mach-O section table is invalid")
                if segment == b"__LINKEDIT":
                    # Only these two sizes include a changing signature blob.
                    command[32:40] = b"\0" * 8
                    command[48:56] = b"\0" * 8
                for index in range(sections):
                    section = 72 + index * 80
                    position = struct.unpack_from("<I", command, section + 48)[0]
                    section_flags = struct.unpack_from("<I", command, section + 64)[0]
                    section_size = struct.unpack_from("<Q", command, section + 40)[0]
                    if section_flags & 0xFF not in (1, 0xC, 0x12) and section_size:
                        if position < end or position + section_size > len(raw):
                            raise ValueError("native kernel Mach-O section is out of bounds")
                        data_offsets.append(position)
            records.append([kind, command.hex()])
    if offset != end or not data_offsets:
        raise ValueError("native kernel Mach-O has no closed machine payload")
    payload_start = min(data_offsets)
    payload_end = len(raw) if signature is None else signature[0]
    if payload_start >= payload_end or signature is not None and (
            signature[1] <= 0 or signature[0] + signature[1] != len(raw)):
        raise ValueError("native kernel Mach-O signature/payload bounds are invalid")
    header = [magic, cpu, subtype, filetype, flags, reserved]
    hasher = hashlib.sha256(json.dumps([header, records], separators=(",", ":")).encode())
    hasher.update(raw[payload_start:payload_end])
    return {"version": 1, "architecture": "arm64" if cpu == 0x100000C else "x86_64",
        "sha256": hasher.hexdigest(), "rpaths": rpaths}


def finalize_manifest(module_root, package_root, *, compiled_manifest):
    module_root, package_root = Path(module_root).resolve(), Path(package_root).resolve()
    module_root.relative_to(package_root)
    native = module_root / "native"
    manifest_path = native / "manifest.json"
    library = native / "omni_packed_kernel.dylib"
    source = module_root / "kernels/weighted_ternary.cpp"
    for path in (manifest_path, library, source):
        path.resolve(strict=True).relative_to(package_root)
        if not path.is_file() or path == manifest_path and path.is_symlink():
            raise ValueError("native kernel package custody requires regular contained files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (type(manifest.get("abi")) is not int or manifest["abi"] != 1 or manifest.get("platform") != "darwin"
            or manifest.get("file") != library.name or manifest.get("machoPayloadVersion") != 1
            or manifest.get("sourceSha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise ValueError("native kernel manifest ABI/source/payload custody is invalid")
    if (set(manifest) != set(compiled_manifest) or any(type(manifest[key]) is not type(value) or manifest[key] != value
            for key, value in compiled_manifest.items() if key != "sha256")):
        raise ValueError("native kernel package differs from its exact compiler-issued manifest")
    raw = library.read_bytes()
    payload = macho_payload(library, raw=raw)
    if (manifest.get("architecture") != payload["architecture"]
            or manifest.get("machoPayloadSha256") != payload["sha256"]):
        raise ValueError("native kernel machine payload or architecture differs from its compiled source custody")
    digest = hashlib.sha256(raw).hexdigest()
    updated = {**manifest, "sha256": digest}
    if library.read_bytes() != raw:
        raise ValueError("native kernel changed during final payload verification")
    if updated != manifest:
        temporary = manifest_path.with_name("manifest.json.next")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(updated, stream, indent=2); stream.write("\n")
        temporary.replace(manifest_path)
    return {"architecture": payload["architecture"], "sha256": digest,
        "sourceSha256": manifest["sourceSha256"], "compiledPayloadUnchanged": True}


def checked_compiled_manifest(module_root):
    module_root = Path(module_root)
    library = module_root / "native/omni_packed_kernel.dylib"
    source = module_root / "kernels/weighted_ternary.cpp"
    manifest = json.loads((module_root / "native/manifest.json").read_text(encoding="utf-8"))
    raw = library.read_bytes()
    payload = macho_payload(library, raw=raw)
    if (type(manifest.get("abi")) is not int or manifest["abi"] != 1 or manifest.get("platform") != "darwin"
            or manifest.get("file") != library.name or manifest.get("machoPayloadVersion") != 1
            or manifest.get("architecture") != payload["architecture"]
            or manifest.get("machoPayloadSha256") != payload["sha256"]
            or manifest.get("sourceSha256") != hashlib.sha256(source.read_bytes()).hexdigest()
            or manifest.get("sha256") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("compiler-issued native kernel source/payload/full-file custody is invalid")
    return manifest
