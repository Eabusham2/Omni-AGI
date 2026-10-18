import { createHash } from "node:crypto";
import type { ProcessMemoryIdentity, ProcessMemoryNativeRequest, ProcessMemoryReceipt } from "../shared/processMemory";
import { PROCESS_MEMORY_FORMAT, PROCESS_MEMORY_MAX_BYTES, PROCESS_MEMORY_MAX_REGIONS } from "../shared/processMemory";

export function processMemoryCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(processMemoryCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${processMemoryCanonical(child)}`).join(",")}}`;
  const text = JSON.stringify(value); if (text === undefined) throw new Error("Process-memory records require explicit JSON."); return text;
}
export const processMemoryHash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
export const memoryObject = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Process-memory operands must be an object.");
  return value as Record<string, unknown>;
};
export function memoryKeys(value: Record<string, unknown>, keys: readonly string[]): void {
  if (Object.keys(value).some(key => !keys.includes(key))) throw new Error("Unsupported process-memory operands.");
}
export function memoryId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid process-memory owner/request identity.");
  return value;
}
export function memorySha(value: unknown): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Process-memory SHA-256 must be exact lowercase hexadecimal.");
  return value;
}
export function memoryWhole(value: unknown, maximum: number, minimum = 1): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum || Number(value) > maximum) throw new Error("Invalid process-memory integer operand.");
  return Number(value);
}
export function memoryAddress(value: unknown): string {
  if (typeof value !== "string" || !/^0x(?:0|[1-9a-f][0-9a-f]{0,15})$/.test(value)) throw new Error("Use an exact canonical hexadecimal process address, not a rounded number.");
  return value;
}
export function memoryBytes(value: unknown): Buffer {
  if (typeof value !== "string" || value.length > Math.ceil(PROCESS_MEMORY_MAX_BYTES / 3) * 4 || !value.length ||
      !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) throw new Error("Memory data must be bounded canonical base64.");
  const bytes = Buffer.from(value, "base64");
  if (bytes.length < 1 || bytes.length > PROCESS_MEMORY_MAX_BYTES || bytes.toString("base64") !== value) throw new Error("Memory data is outside the transfer envelope.");
  return bytes;
}
export function memoryIdentity(value: unknown): ProcessMemoryIdentity {
  const row = memoryObject(value); memoryKeys(row, ["platform", "pid", "creation", "executable", "executableIdentity"]);
  if (!["darwin", "linux", "win32"].includes(String(row.platform)) || typeof row.creation !== "string" ||
      !row.creation.length || row.creation.length > 256 || typeof row.executable !== "string" || !row.executable.length ||
      row.executable.length > 32768 || typeof row.executableIdentity !== "string" || !row.executableIdentity.length ||
      row.executableIdentity.length > 256) throw new Error("The operating-system process identity is invalid.");
  return { platform: row.platform as ProcessMemoryIdentity["platform"], pid: memoryWhole(row.pid, 0x7fffffff, 2),
    creation: row.creation, executable: row.executable, executableIdentity: row.executableIdentity };
}
export function memoryReceipt(value: unknown, request: ProcessMemoryNativeRequest): ProcessMemoryReceipt {
  const row = memoryObject(value), hash = processMemoryHash(processMemoryCanonical(request));
  const states = ["complete", "partial", "denied", "protected", "unsupported", "identity-changed", "preimage-mismatch", "cancelled", "failed", "write-uncertain"];
  const length = request.length ?? 0;
  if (row.format !== PROCESS_MEMORY_FORMAT || row.operationId !== request.operationId || row.operation !== request.operation ||
      row.requestSha256 !== hash || !states.includes(String(row.state)) || row.requestedBytes !== length ||
      row.compareAtomic !== false || typeof row.writeMayHaveOccurred !== "boolean" || typeof row.retrySafe !== "boolean" ||
      (row.transferredBytes !== null && (!Number.isSafeInteger(row.transferredBytes) || Number(row.transferredBytes) < 0 || Number(row.transferredBytes) > length)) ||
      (request.operation !== "write" && row.writeMayHaveOccurred !== false) ||
      (row.writeMayHaveOccurred && row.retrySafe !== false)) throw new Error("The native process-memory receipt does not match its exact request.");
  const identity = row.identity === undefined ? undefined : memoryIdentity(row.identity);
  if (identity && (identity.pid !== request.pid || (request.expectedIdentity && processMemoryCanonical(identity) !== processMemoryCanonical(request.expectedIdentity)))) throw new Error("Process-memory receipt changed its target identity.");
  if (row.state === "complete" && !identity) throw new Error("A successful memory operation requires an actual target identity.");
  if (row.state === "complete" && ["read", "write"].includes(request.operation) && row.transferredBytes !== length) throw new Error("A complete memory transfer must cover its exact requested range.");
  if (row.transferredBytes === null && (request.operation !== "write" || row.writeMayHaveOccurred !== true)) throw new Error("Unknown transfer counts require an honestly uncertain attempted write.");
  if (request.operation === "read" && row.state === "complete" && row.dataBase64 === undefined) throw new Error("A successful read must include its exact bytes.");
  if (row.dataBase64 !== undefined) {
    if (request.operation !== "read" || typeof row.dataBase64 !== "string") throw new Error("Unexpected process-memory byte disclosure.");
    const bytes = row.dataBase64 === "" ? Buffer.alloc(0) : memoryBytes(row.dataBase64);
    if (bytes.length !== row.transferredBytes || processMemoryHash(bytes) !== row.dataSha256) throw new Error("Process-memory returned bytes failed their exact hash/count.");
  }
  if (row.regions !== undefined) {
    if (request.operation !== "regions" || !Array.isArray(row.regions) || row.regions.length > (request.regionLimit ?? PROCESS_MEMORY_MAX_REGIONS)) throw new Error("Process-memory region page exceeded its request.");
    let after = BigInt(request.afterAddress ?? "0x0"), first = true;
    for (const region of row.regions) {
      const item = memoryObject(region), start = memoryAddress(item.start), end = memoryAddress(item.end);
      if ((!first && BigInt(start) < after) || BigInt(end) <= after || BigInt(end) <= BigInt(start) || [item.readable, item.writable, item.executable].some(flag => typeof flag !== "boolean")) throw new Error("Process-memory regions are unordered or malformed.");
      after = BigInt(end); first = false;
    }
    if (row.nextAddress !== undefined && BigInt(memoryAddress(row.nextAddress)) < after) throw new Error("Process-memory page cursor moved backward.");
  }
  return { ...row, ...(identity ? { identity } : {}) } as unknown as ProcessMemoryReceipt;
}
