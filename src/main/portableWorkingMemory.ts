import { createHash } from "node:crypto";

const EMPTY_PAGE_DIGEST = createHash("sha256").digest("hex");

/** A sanitized .omni omits temporary cold working-memory pages. */
export function emptyPortableWorkingMemoryCheckpoint(): Record<string, unknown> {
  return {
    format: "omni-working-memory-pages",
    formatVersion: 1,
    count: 0,
    highWaterId: 0,
    contentSha256: EMPTY_PAGE_DIGEST,
    temporary: true,
    runtimeReadable: true,
    learningReadable: true,
    pageInSupported: true
  };
}

/** Reject metadata that claims cold pages absent from the portable archive. */
export function assertPortableWorkingMemoryCheckpoint(state: unknown, label: string): void {
  if (typeof state !== "object" || state === null || Array.isArray(state)) {
    throw new Error(`${label} is not a valid neural state document.`);
  }
  const record = state as Record<string, unknown>;
  if (record.format !== "omni-cortex-engine") return;
  const ingestion = record.ingestion_checkpoints;
  if (
    ingestion !== undefined &&
    (typeof ingestion !== "object" || ingestion === null ||
      Array.isArray(ingestion) || Object.keys(ingestion).length !== 0)
  ) {
    throw new Error(`${label} claims source-bound ingestion cursors omitted from the bundle.`);
  }
  const checkpoint = record.paged_working_memory;
  // Older sanitized bundles may omit this temporary checkpoint entirely.
  // They cannot claim cold pages, so omission is safe to load as empty.
  if (checkpoint === undefined) return;
  if (typeof checkpoint !== "object" || checkpoint === null || Array.isArray(checkpoint)) {
    throw new Error(`${label} has an invalid portable working-memory checkpoint.`);
  }
  const actual = checkpoint as Record<string, unknown>;
  const expected = emptyPortableWorkingMemoryCheckpoint();
  if (
    Object.keys(actual).length !== Object.keys(expected).length ||
    Object.entries(expected).some(([key, value]) => actual[key] !== value)
  ) {
    throw new Error(`${label} claims working-memory pages omitted from the bundle.`);
  }
}
