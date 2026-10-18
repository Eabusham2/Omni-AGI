import { createHash } from "node:crypto";
import {
  PLUGIN_WASM_PARSER_ABI,
  type PluginTrainingModuleDefinition
} from "../shared/plugins";

const SHA256 = /^[a-f0-9]{64}$/;
const WASM_PAGE_BYTES = 65_536;

function readU32(bytes: Uint8Array, cursor: { offset: number }, end = bytes.length): number {
  let value = 0, shift = 0;
  for (let index = 0; index < 5; index += 1) {
    if (cursor.offset >= end) throw new Error("WASM parser module ended inside an unsigned integer.");
    const byte = bytes[cursor.offset++]!;
    value |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return value >>> 0;
    shift += 7;
  }
  throw new Error("WASM parser module contains an oversized unsigned integer.");
}

export interface PluginWasmModuleInspection {
  bytes: Buffer;
  initialPages: number;
  maximumPages: number;
  initialBytes: number;
  maximumBytes: number;
}

/** Structural admission only. Compilation/instantiation occurs in the
 * killable worker so untrusted module work never runs on Studio's main loop. */
export function inspectPluginWasmModule(module: PluginTrainingModuleDefinition): PluginWasmModuleInspection {
  if (module.kind !== "wasm-stream" || module.abi !== PLUGIN_WASM_PARSER_ABI ||
      typeof module.wasmBase64 !== "string" || !module.wasmBase64.length ||
      module.wasmBase64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(module.wasmBase64) ||
      typeof module.wasmSha256 !== "string" || !SHA256.test(module.wasmSha256)) {
    throw new Error("Executable plugin parser module declaration is invalid.");
  }
  const bytes = Buffer.from(module.wasmBase64, "base64");
  if (!bytes.length || bytes.toString("base64") !== module.wasmBase64 ||
      createHash("sha256").update(bytes).digest("hex") !== module.wasmSha256) {
    throw new Error("Executable plugin parser module bytes failed canonical checksum validation.");
  }
  if (bytes.length < 8 || bytes.subarray(0, 8).toString("hex") !== "0061736d01000000") {
    throw new Error("Executable plugin parser is not a WebAssembly v1 module.");
  }
  const cursor = { offset: 8 };
  let memory: { initial: number; maximum: number } | undefined;
  while (cursor.offset < bytes.length) {
    const sectionId = bytes[cursor.offset++]!;
    const size = readU32(bytes, cursor), end = cursor.offset + size;
    if (!Number.isSafeInteger(end) || end > bytes.length) throw new Error("WASM parser section exceeds its module bytes.");
    if (sectionId === 2) {
      const nested = { offset: cursor.offset };
      if (readU32(bytes, nested, end) !== 0) {
        throw new Error("Executable plugin parsers may not import functions, memory, globals, tables, or host capabilities.");
      }
    } else if (sectionId === 5) {
      if (memory) throw new Error("Executable plugin parser declares multiple memory sections.");
      const nested = { offset: cursor.offset }, count = readU32(bytes, nested, end);
      if (count !== 1) throw new Error("Executable plugin parser must declare exactly one bounded memory.");
      const flags = readU32(bytes, nested, end);
      if (flags !== 1) throw new Error("Executable plugin parser memory must be unshared wasm32 with an explicit maximum.");
      const initial = readU32(bytes, nested, end), maximum = readU32(bytes, nested, end);
      if (nested.offset !== end || initial > maximum || maximum > 65_536) {
        throw new Error("Executable plugin parser memory limits are invalid.");
      }
      memory = { initial, maximum };
    } else if (sectionId === 8) {
      throw new Error("Executable plugin parsers may not declare a start function.");
    }
    cursor.offset = end;
  }
  if (!memory) throw new Error("Executable plugin parser has no bounded linear memory.");
  return { bytes, initialPages: memory.initial, maximumPages: memory.maximum,
    initialBytes: memory.initial * WASM_PAGE_BYTES, maximumBytes: memory.maximum * WASM_PAGE_BYTES };
}
