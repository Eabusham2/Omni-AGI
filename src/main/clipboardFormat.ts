import { createHash } from "node:crypto";
import { CLIPBOARD_FORMAT, type ClipboardMutationReceipt, type ClipboardProvenance, type ClipboardReadReceipt, type ClipboardScope } from "../shared/clipboard";

export function clipboardCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(clipboardCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value as Record<string, unknown>)
    .filter(([, child]) => child !== undefined).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${clipboardCanonical(child)}`).join(",")}}`;
  return JSON.stringify(value);
}
export const clipboardHash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
export function clipboardObject(value: unknown, label = "clipboard value"): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}
export function clipboardKeys(value: Record<string, unknown>, allowed: readonly string[]): void {
  const extra = Object.keys(value).filter(key => !allowed.includes(key)); if (extra.length) throw new Error(`Unsupported clipboard operands: ${extra.join(", ")}.`);
}
export function clipboardId(value: unknown, label = "clipboard identity"): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) throw new Error(`Invalid ${label}.`); return value;
}
export function clipboardSha(value: unknown, label = "clipboard checksum"): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error(`Invalid ${label}.`); return value;
}
export function clipboardScope(value: unknown): ClipboardScope {
  if (value !== "instance" && value !== "host") throw new Error("Clipboard scope must be exactly instance or host."); return value;
}
export function clipboardWhole(value: unknown, label: string, maximum: number, fallback?: number): number {
  if (value === undefined && fallback !== undefined) return fallback;
  if (!Number.isSafeInteger(value) || Number(value) < 0 || Number(value) > maximum) throw new Error(`${label} must be an integer from 0 to ${maximum}.`); return Number(value);
}
export function clipboardText(value: unknown): string {
  if (typeof value !== "string") throw new Error("Clipboard text must be UTF-8 text."); return value;
}
export function clipboardTimestamp(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error(`Invalid ${label}.`); return value;
}
export function clipboardProvenance(value: unknown): ClipboardProvenance {
  const row = clipboardObject(value, "clipboard provenance") as unknown as ClipboardProvenance;
  if (!["instance-owned", "host-os-clipboard", "initial", "inherited"].includes(row.kind)) throw new Error("Invalid clipboard provenance kind.");
  clipboardTimestamp(row.capturedAt, "clipboard provenance timestamp");
  if (row.brainId !== undefined) clipboardId(row.brainId, "clipboard provenance owner");
  if (row.parentBrainId !== undefined) clipboardId(row.parentBrainId, "clipboard provenance parent");
  if (row.parentVersionId !== undefined) clipboardId(row.parentVersionId, "clipboard provenance version");
  if (row.adapter !== undefined && (typeof row.adapter !== "string" || !/^[A-Za-z0-9._:-]{1,128}$/.test(row.adapter))) throw new Error("Invalid clipboard provenance adapter.");
  return row;
}
export function clipboardMutationReceipt(value: unknown): ClipboardMutationReceipt {
  const row = clipboardObject(value, "clipboard mutation receipt");
  if (row.format !== CLIPBOARD_FORMAT || row.version !== 1 || !["write", "clear"].includes(String(row.action)) ||
      !["instance", "host"].includes(String(row.scope)) || !["complete", "cancelled", "failed", "write-uncertain"].includes(String(row.state)) ||
      row.contentType !== "text/plain" || !Number.isSafeInteger(row.contentBytes) || Number(row.contentBytes) < 0 ||
      typeof row.writeMayHaveOccurred !== "boolean" || typeof row.retrySafe !== "boolean") throw new Error("Invalid clipboard mutation receipt.");
  clipboardId(row.requestId, "clipboard request ID"); clipboardSha(row.requestSha256, "clipboard request checksum"); clipboardSha(row.contentSha256, "clipboard content checksum");
  if (row.versionId !== undefined) clipboardId(row.versionId, "clipboard version ID");
  if (row.revision !== undefined && (!Number.isSafeInteger(row.revision) || Number(row.revision) < 0)) throw new Error("Invalid clipboard revision.");
  if (row.error !== undefined && (typeof row.error !== "string" || row.error.length > 4000 || row.error.includes("\0"))) throw new Error("Invalid clipboard receipt error.");
  clipboardProvenance(row.provenance);
  if (row.repeated !== undefined && typeof row.repeated !== "boolean") throw new Error("Invalid clipboard repeated-effect marker.");
  if (row.scope === "instance" && (row.state !== "complete" || row.versionId === undefined || row.revision === undefined || row.writeMayHaveOccurred !== true || row.retrySafe !== false)) throw new Error("Instance clipboard receipt is not an exact committed version.");
  if (row.scope === "host" && (row.versionId !== undefined || row.revision !== undefined || row.retrySafe !== false ||
      (row.state === "complete" && row.writeMayHaveOccurred !== true) || (row.state === "write-uncertain" && row.writeMayHaveOccurred !== true) ||
      (row.state === "cancelled" && row.writeMayHaveOccurred !== false))) throw new Error("Host clipboard receipt has inconsistent effect semantics.");
  return row as unknown as ClipboardMutationReceipt;
}
export function clipboardReadReceipt(value: unknown): ClipboardReadReceipt {
  const row = clipboardObject(value, "clipboard read receipt");
  if (row.format !== CLIPBOARD_FORMAT || row.version !== 1 || row.action !== "read" || !["instance", "host"].includes(String(row.scope)) ||
      !["complete", "partial"].includes(String(row.state)) || row.contentType !== "text/plain" || typeof row.text !== "string" ||
      !Number.isSafeInteger(row.chunkBytes) || Number(row.chunkBytes) < 0 || typeof row.complete !== "boolean") throw new Error("Invalid clipboard read receipt.");
  clipboardId(row.requestId, "clipboard request ID"); clipboardSha(row.chunkSha256, "clipboard chunk checksum");
  clipboardProvenance(row.provenance);
  if (Buffer.byteLength(row.text, "utf8") !== row.chunkBytes || clipboardHash(row.text) !== row.chunkSha256) throw new Error("Clipboard read bytes differ from their receipt.");
  return row as unknown as ClipboardReadReceipt;
}
