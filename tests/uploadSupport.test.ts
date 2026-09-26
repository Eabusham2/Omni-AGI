import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  AUDIO_UPLOAD_EXTENSIONS,
  EXPERIENCE_UPLOADS,
  IMAGE_UPLOAD_EXTENSIONS,
  VIDEO_UPLOAD_EXTENSIONS,
  isExperienceUploadKind
} from "../src/shared/uploadSupport";

const root = resolve(import.meta.dirname, "..");

function read(relativePath: string): string {
  return readFileSync(resolve(root, relativePath), "utf8");
}

describe("experience upload support", () => {
  it("covers document, dataset, image, audio, and video formats in native selectors", () => {
    expect(EXPERIENCE_UPLOADS.files.extensions).toEqual(
      expect.arrayContaining([
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "odt",
        "ods",
        "odp",
        "parquet",
        "arrow",
        "sqlite",
        "zip",
        "gz",
        "bz2",
        "xz",
        "py",
        "png",
        "wav",
        "mp4"
      ])
    );
    expect(IMAGE_UPLOAD_EXTENSIONS).toEqual(
      expect.arrayContaining(["png", "jpeg", "gif", "tiff", "avif", "heic"])
    );
    expect(AUDIO_UPLOAD_EXTENSIONS).toEqual(
      expect.arrayContaining(["wav", "mp3", "flac", "ogg", "opus", "aiff"])
    );
    expect(VIDEO_UPLOAD_EXTENSIONS).toEqual(
      expect.arrayContaining(["mp4", "webm", "mov", "mkv", "avi", "mpeg"])
    );
    expect(new Set(EXPERIENCE_UPLOADS.files.extensions).size).toBe(
      EXPERIENCE_UPLOADS.files.extensions.length
    );
  });

  it("fails closed to the four supported picker categories", () => {
    expect(isExperienceUploadKind("files")).toBe(true);
    expect(isExperienceUploadKind("images")).toBe(true);
    expect(isExperienceUploadKind("audio")).toBe(true);
    expect(isExperienceUploadKind("video")).toBe(true);
    expect(isExperienceUploadKind("folder")).toBe(false);
    expect(isExperienceUploadKind("../video")).toBe(false);
    expect(isExperienceUploadKind({ kind: "files" })).toBe(false);
  });

  it("wires Build, Data Studio, and Run chat to neural ingestion with visible results", () => {
    const renderer = read("src/renderer/src/App.tsx");
    const preload = read("src/preload/index.ts");
    const ipc = read("src/main/ipc.ts");
    const service = read("src/main/brainService.ts");
    const chatAttachmentFlow = read(
      "src/renderer/src/chatAttachmentLearning.ts"
    );

    for (const label of [
      "Files & datasets",
      "Images",
      "Audio",
      "Video",
      "Whole folder"
    ]) {
      expect(renderer).toContain(label);
    }
    expect(renderer).toContain('attachExperience("images")');
    expect(renderer).toContain('attachExperience("audio")');
    expect(renderer).toContain('attachExperience("video")');
    expect(renderer).toContain('attachExperience("folder")');
    expect(renderer).toContain("attachDroppedExperience");
    expect(renderer).toContain("CHAT ATTACHMENT · NEURAL LEARNING");
    expect(renderer).toContain("encoded into neural state");
    expect(renderer).toContain("learnedSynapses");
    expect(renderer).toContain("window.omni.data.preview");
    expect(renderer).toContain("window.omni.data.start");
    const chatAttachmentStart = renderer.indexOf("const attachExperience");
    const chatAttachmentEnd = renderer.indexOf(
      "const attachDroppedExperience",
      chatAttachmentStart
    );
    const chatAttachment = renderer.slice(chatAttachmentStart, chatAttachmentEnd);
    expect(chatAttachment).not.toContain("data.ingestFolder");
    expect(chatAttachment).not.toContain("data.ingestFiles");
    expect(chatAttachment).toContain("queueChatAttachmentLearning");
    expect(chatAttachment).toContain("attachmentOperationGateRef.current.begin()");
    expect(chatAttachment).toContain("attachmentOperationGateRef.current.finish(operationLease)");
    expect(chatAttachment).toContain("setDatasetPreview({");
    expect(chatAttachmentFlow.indexOf("data.preview(request)")).toBeGreaterThan(-1);
    expect(chatAttachmentFlow.indexOf("data.start({")).toBeGreaterThan(
      chatAttachmentFlow.indexOf("data.preview(request)")
    );
    expect(renderer).toContain("DatasetActivityCard");
    expect(renderer).toContain("Cancel safely");
    expect(renderer).toContain("window.omni.data.cancelPreview");
    expect(renderer).toContain("window.omni.data.onPreviewProgress");
    expect(preload).toContain("webUtils.getPathForFile");
    expect(preload).toContain("IPC.data.previewEvent");
    expect(preload).toContain("IPC.data.cancelPreview");
    expect(preload).toContain("start: (request) => invoke(IPC.data.start, request)");
    expect(preload).toContain("IPC.data.ingestDropped");
    expect(ipc).toContain("EXPERIENCE_UPLOADS");
    expect(ipc).toContain("if (choice.canceled || choice.filePaths.length === 0) return null;");
    expect(ipc).toContain("previewControllers.delete(requestId)");
    expect(ipc).toContain("service.ingestPaths");
    expect(service).toContain('"ingest"');
    expect(service).toContain("learnedSynapses");
  });
});
