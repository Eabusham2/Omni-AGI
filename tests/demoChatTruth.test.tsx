import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { ComposerSubmitControl } from "../src/renderer/src/ComposerSubmitControl";

// Keep this actual TSX-control fixture in the JSX-enabled suite; no browser/engine.
vi.mock("react", () => ({ useState: (initial: unknown) => [initial, vi.fn()], useEffect: vi.fn() }));

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
    const rejected = send.indexOf("return;", guard);
    expect(rejected).toBeGreaterThan(guard);
    for (const mutation of ["clearSubmittedChatDraft", "setSending(true)", "setOptimisticHumans", "generationPhasesRef.current.set", "taskAttachmentDraft.take()"]) {
      const position = send.indexOf(mutation);
      expect(position).toBeGreaterThan(guard);
      expect(rejected).toBeLessThan(position);
    }
    expect(app).toContain("canSend={Boolean(runtimeLoaded && input.trim() && window.omni && draftCapacity.fits && !taskAttachmentDraft.busy && !waitingInheritedFileReview)}");
    expect(app).toContain('{!window.omni ? <p className="composer-disclaimer" role="status">');
    expect(app).toContain("Preview only — no brain is connected. Sending and learning are unavailable.");

    const onSend = vi.fn(), onSteer = vi.fn(), onStop = vi.fn();
    const submit = (canSend: boolean, active: boolean, stopping: boolean) => {
      const root = ComposerSubmitControl({ active, stopping, canSend, canSteer: false, onSend, onSteer, onStop });
      const children = root.props.children as Array<ReactElement<{ className: string; disabled: boolean }> | null>;
      const button = children.find(child => child?.props.className.includes("composer__submit"));
      expect(button).toBeDefined();
      return button!;
    };
    for (const active of [false, true]) {
      expect(submit(false, active, false).props.disabled).toBe(true);
      expect(submit(true, active, false).props.disabled).toBe(false);
      expect(submit(true, active, true).props.disabled).toBe(true);
    }
    expect(onSend).not.toHaveBeenCalled();
    expect(onSteer).not.toHaveBeenCalled();
    expect(onStop).not.toHaveBeenCalled();
  });

  it("keeps static preview fixtures but has no synthetic chat or plasticity mutation", () => {
    expect(app).toContain("const demo = !window.omni;");
    expect(app).toContain('aria-label="Preview · no engine"');
    expect(app).not.toContain('demo ? "Design preview" : "Local engine"');
    expect(app).not.toContain("makeDemoChat");
    expect(demo).not.toContain("makeDemoChat");
    expect(demo).not.toContain("plasticityEvents: brain.counters.plasticityEvents + 7");
  });
});
