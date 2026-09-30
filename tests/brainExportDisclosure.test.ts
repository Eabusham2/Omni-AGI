import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  BRAIN_EXPORT_CONFIRMATION_DETAIL,
  BRAIN_EXPORT_DISCLOSURE
} from "../src/shared/brainExportDisclosure";

describe("saved-instance export disclosures", () => {
  it("describes preserved unsanitized content and the external-vault boundary", () => {
    expect(BRAIN_EXPORT_DISCLOSURE).toMatch(/saved chat.*temporary attention.*pending learning.*recovery points/);
    expect(BRAIN_EXPORT_DISCLOSURE).toContain("trusted recipients");
    expect(BRAIN_EXPORT_CONFIRMATION_DETAIL).toContain("not redacted");
    expect(BRAIN_EXPORT_CONFIRMATION_DETAIL).toContain("OS credential vault is excluded");
    expect(BRAIN_EXPORT_CONFIRMATION_DETAIL).toContain("not automatically resumed");
    expect(BRAIN_EXPORT_CONFIRMATION_DETAIL).toContain("not a running-process clone");
  });

  it("uses the same disclosure in the renderer and warns before every export mode", () => {
    const ipc = readFileSync(new URL("../src/main/ipc.ts", import.meta.url), "utf8");
    const handler = ipc.slice(ipc.indexOf("handle(IPC.brain.export"), ipc.indexOf("handle(IPC.brain.import"));
    expect(handler).toContain("detail: BRAIN_EXPORT_CONFIRMATION_DETAIL");
    expect(handler).not.toContain('if (mode === "private-archive")');
    expect(handler.indexOf("if (confirmation.response !== 1) return null")).toBeLessThan(handler.indexOf("dialog.showSaveDialog"));
    const renderer = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    expect(renderer).toContain("{BRAIN_EXPORT_DISCLOSURE}");
    expect(renderer).not.toContain("Private chat, unfinished training, and temporary attention are not included");
  });
});
