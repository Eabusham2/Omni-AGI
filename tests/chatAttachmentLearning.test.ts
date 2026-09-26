import { describe, expect, it, vi } from "vitest";
import type { DatasetManifest, RuntimeJob } from "../src/shared/types";
import {
  ChatAttachmentOperationGate,
  queueChatAttachmentLearning
} from "../src/renderer/src/chatAttachmentLearning";

const manifest: DatasetManifest = {
  schemaVersion: 1,
  id: "manifest-1",
  brainId: "brain-1",
  createdAt: "2026-09-19T00:00:00.000Z",
  updatedAt: "2026-09-19T00:00:00.000Z",
  roots: ["fixture.txt"],
  entryFile: "entries.jsonl",
  discoveredFiles: 1,
  discoveredBytes: 12,
  manifestHash: "a".repeat(64)
};

const job: RuntimeJob = {
  id: "job-1",
  brainId: "brain-1",
  kind: "ingestion",
  state: "queued",
  progress: 0,
  label: "Queued attachment learning",
  createdAt: "2026-09-19T00:00:01.000Z",
  updatedAt: "2026-09-19T00:00:01.000Z"
};

describe("chat attachment learning", () => {
  it("commits one selected manifest and starts exactly one resumable job", async () => {
    const calls: string[] = [];
    const preview = vi.fn(async () => {
      calls.push("preview");
      return manifest;
    });
    const start = vi.fn(async () => {
      calls.push("start");
      return job;
    });

    await expect(queueChatAttachmentLearning(
      { preview, start },
      {
        brainId: "brain-1",
        policy: "pretrain",
        requestId: "preview-1",
        selection: "files"
      }
    )).resolves.toEqual({ manifest, job });

    expect(calls).toEqual(["preview", "start"]);
    expect(preview).toHaveBeenCalledTimes(1);
    expect(start).toHaveBeenCalledTimes(1);
    expect(start).toHaveBeenCalledWith({
      brainId: "brain-1",
      manifestId: "manifest-1",
      policy: "pretrain",
      epochs: 1,
      resume: true
    });
  });

  it("does not create a job when the native picker is cancelled", async () => {
    const start = vi.fn(async () => job);
    await expect(queueChatAttachmentLearning(
      { preview: vi.fn(async () => null), start },
      {
        brainId: "brain-1",
        policy: "pretrain",
        requestId: "preview-cancelled",
        selection: "audio"
      }
    )).resolves.toBeNull();
    expect(start).not.toHaveBeenCalled();
  });

  it("does not schedule a stale selection after the workspace changes", async () => {
    const start = vi.fn(async () => job);
    await expect(queueChatAttachmentLearning(
      { preview: vi.fn(async () => manifest), start },
      {
        brainId: "brain-1",
        policy: "pretrain",
        requestId: "preview-stale",
        selection: "files"
      },
      () => false
    )).resolves.toBeNull();
    expect(start).not.toHaveBeenCalled();
  });

  it("holds one synchronous picker lease and always permits a later retry", () => {
    const gate = new ChatAttachmentOperationGate();
    const first = gate.begin();
    expect(first).toBeTypeOf("symbol");
    expect(gate.begin()).toBeNull();
    expect(gate.finish(Symbol("stale"))).toBe(false);
    expect(gate.finish(first!)).toBe(true);
    const retry = gate.begin();
    expect(retry).toBeTypeOf("symbol");
    gate.reset();
    expect(gate.isCurrent(retry!)).toBe(false);
    expect(gate.begin()).toBeTypeOf("symbol");
  });
});
