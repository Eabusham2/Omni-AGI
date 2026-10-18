import { createHash } from "node:crypto";
import { PLUGIN_WASM_PARSER_ABI, type PluginTrainingModuleDefinition } from "../../src/shared/plugins";

const u32 = (value: number): number[] => {
  const bytes: number[] = []; let remaining = value >>> 0;
  do { let byte = remaining & 0x7f; remaining >>>= 7; if (remaining) byte |= 0x80; bytes.push(byte); } while (remaining);
  return bytes;
};
const name = (value: string): number[] => { const bytes = [...Buffer.from(value)]; return [...u32(bytes.length), ...bytes]; };
const section = (id: number, payload: number[]): number[] => [id, ...u32(payload.length), ...payload];

export function parserWasm(options: { maximum?: number; importedMemory?: boolean; infinite?: boolean } = {}): Buffer {
  const types = [2, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 3, 0x7f, 0x7f, 0x7f, 1, 0x7e];
  const imports = options.importedMemory
    ? section(2, [1, ...name("env"), ...name("memory"), 2, 1, 1, 1]) : [];
  const maximum = options.maximum;
  const memory = section(5, maximum === undefined ? [1, 0, 1] : [1, 1, 1, ...u32(maximum)]);
  const exports = section(7, [3,
    ...name("memory"), 2, 0,
    ...name("omni_alloc"), 0, 0,
    ...name("omni_parse"), 0, 1]);
  const alloc = [0, 0x41, 0, 0x0b];
  const parse = options.infinite
    ? [0, 0x03, 0x40, 0x0c, 0, 0x0b, 0x42, 0, 0x0b]
    : [0, 0x20, 0, 0xad, 0x42, 0x20, 0x86, 0x20, 1, 0xad, 0x84, 0x0b];
  const code = section(10, [2, ...u32(alloc.length), ...alloc, ...u32(parse.length), ...parse]);
  return Buffer.from([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...section(1, types), ...imports, ...section(3, [2, 0, 1]), ...memory, ...exports, ...code]);
}

export function parserModule(bytes = parserWasm({ maximum: 4 }), id = "lines"): PluginTrainingModuleDefinition {
  return { id, kind: "wasm-stream", abi: PLUGIN_WASM_PARSER_ABI,
    wasmBase64: bytes.toString("base64"), wasmSha256: createHash("sha256").update(bytes).digest("hex") };
}
