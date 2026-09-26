import { createHash } from "node:crypto";
import { expect, test } from "vitest";
import {
  assertPortableWorkingMemoryCheckpoint,
  emptyPortableWorkingMemoryCheckpoint
} from "../src/main/portableWorkingMemory";

test("sanitized working-memory checkpoint matches the Python empty page digest", () => {
  const checkpoint = emptyPortableWorkingMemoryCheckpoint();
  expect(checkpoint).toMatchObject({
    format: "omni-working-memory-pages",
    formatVersion: 1,
    count: 0,
    highWaterId: 0,
    contentSha256: createHash("sha256").digest("hex")
  });
  expect(() => assertPortableWorkingMemoryCheckpoint({
    format: "omni-cortex-engine",
    paged_working_memory: checkpoint
  }, "Current .omni state")).not.toThrow();
});

test("import rejects a checkpoint claiming missing cold pages", () => {
  const checkpoint = emptyPortableWorkingMemoryCheckpoint();
  expect(() => assertPortableWorkingMemoryCheckpoint({
    format: "omni-cortex-engine",
    paged_working_memory: { ...checkpoint, count: 7, highWaterId: 11 }
  }, "Current .omni state")).toThrow(/claims working-memory pages/);
});

test("older sanitized metadata without a cold page checkpoint is safe", () => {
  expect(() => assertPortableWorkingMemoryCheckpoint({
    format: "omni-cortex-engine"
  }, "Current .omni state")).not.toThrow();
});

test("import rejects source-bound active ingestion cursors", () => {
  expect(() => assertPortableWorkingMemoryCheckpoint({
    format: "omni-cortex-engine",
    paged_working_memory: emptyPortableWorkingMemoryCheckpoint(),
    ingestion_checkpoints: { pending: { source: "/private/source" } }
  }, "Current .omni state")).toThrow(/source-bound ingestion cursors/);
});
