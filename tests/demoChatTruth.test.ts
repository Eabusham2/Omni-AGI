import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
const demo = readFileSync(resolve(root, "src/renderer/src/demo.ts"), "utf8");

describe("browser design preview chat boundary", () => {
  it("blocks submission before clearing a draft or starting a turn", () => {
    const send = app.slice(
      app.indexOf("  const send = async ("),
      app.indexOf("  const approvePendingTool = async")
    );
    const guard = send.indexOf("if (!window.omni) {");
    expect(guard).toBeGreaterThan(-1);
    expect(send.indexOf("onToast(\"Neural engine unavailable", guard)).toBeGreaterThan(guard);
    expect(send.indexOf("return;", guard)).toBeLessThan(send.indexOf("clearSubmittedChatDraft"));
    expect(send.indexOf("return;", guard)).toBeLessThan(send.indexOf("setSending(true)"));
    expect(app).toContain("disabled={!input.trim() || !window.omni}");
    expect(app).toContain("Drafts stay in the composer; no chat or learning runs.");
  });

  it("keeps static preview fixtures but has no synthetic chat or plasticity mutation", () => {
    expect(app).toContain("const demo = !window.omni;");
    expect(app).toContain('demo ? "Design preview" : "Local engine"');
    expect(app).not.toContain("makeDemoChat");
    expect(demo).not.toContain("makeDemoChat");
    expect(demo).not.toContain("plasticityEvents: brain.counters.plasticityEvents + 7");
  });
});
