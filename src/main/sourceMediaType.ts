import { constants } from "node:fs";
import { open } from "node:fs/promises";
import type { DatasetFormat } from "../shared/types";
import type { SourceMediaType } from "../shared/sourceMediaType";
export { normalizeSourceMediaType, validSourceMediaType } from "../shared/sourceMediaType";

/** Only concrete signatures in a forty-byte prefix; no complete-file sniff. */
export async function observeLocalSourceMediaType(path: string, signal?: AbortSignal): Promise<SourceMediaType | undefined> {
  signal?.throwIfAborted();
  const file = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    if (!(await file.stat()).isFile()) throw new Error("MIME observation needs a regular selected file.");
    const prefix = Buffer.alloc(40), result = await file.read(prefix, 0, prefix.length, 0);
    signal?.throwIfAborted();
    const data = prefix.subarray(0, result.bytesRead), ascii = (start: number, end: number): string => data.subarray(start, end).toString("ascii");
    let value: string | undefined;
    if (data.length >= 8 && data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) value = "image/png";
    else if (data.length >= 3 && data[0] === 255 && data[1] === 216 && data[2] === 255) value = "image/jpeg";
    else if (["GIF87a", "GIF89a"].includes(ascii(0, 6))) value = "image/gif";
    else if (ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") value = "image/webp";
    else if (ascii(0, 4) === "RIFF" && ascii(8, 12) === "WAVE") value = "audio/wav";
    else if (ascii(0, 4) === "fLaC") value = "audio/flac";
    else if (ascii(0, 4) === "OggS") value = "application/ogg";
    else if (ascii(4, 8) === "ftyp") value = "video/mp4";
    else if (ascii(0, 5) === "%PDF-") value = "application/pdf";
    else if (ascii(0, 16) === "SQLite format 3\0") value = "application/vnd.sqlite3";
    else if (ascii(0, 4) === "PAR1") value = "application/vnd.apache.parquet";
    else if (data.length >= 4 && data[0] === 80 && data[1] === 75 && [[3, 4], [5, 6], [7, 8]].some(pair => data[2] === pair[0] && data[3] === pair[1])) value = "application/zip";
    return value ? { value, source: "file-signature" } : undefined;
  } finally { await file.close(); }
}

export function datasetFormatForMediaType(value: string | undefined): DatasetFormat {
  if (!value) return "unknown";
  if (value === "application/pdf") return "pdf";
  if (value.startsWith("image/")) return "image";
  if (value.startsWith("audio/")) return "audio";
  if (value.startsWith("video/")) return "video";
  if (value === "application/vnd.sqlite3" || value === "application/x-sqlite3") return "sqlite";
  if (value === "application/vnd.apache.parquet" || value === "application/x-parquet") return "parquet";
  if (value === "application/vnd.apache.arrow.file" || value === "application/vnd.apache.arrow.stream") return "arrow";
  if (["application/zip", "application/x-tar", "application/gzip", "application/x-gzip"].includes(value)) return "archive";
  if (["application/x-ndjson", "application/jsonl"].includes(value)) return "jsonl";
  if (value === "application/json" || value.endsWith("+json")) return "json";
  if (value === "text/csv") return "csv";
  if (value === "text/tab-separated-values") return "tsv";
  if (value.startsWith("text/") || value === "application/xml" || value.endsWith("+xml")) return "text";
  return "unknown";
}
