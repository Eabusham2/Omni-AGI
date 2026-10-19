/** Inert React/API contracts only: no desktop, provider, worker or neural work. */
import React, { isValidElement, type ReactElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LearningWorkspace } from "../src/renderer/src/LearningWorkspace";
import { EvolutionWorkspace } from "../src/renderer/src/EvolutionWorkspace";
import { UniversalSourcePicker } from "../src/renderer/src/UniversalSourcePicker";
import type { BrainDocument, DatasetManifest } from "../src/shared/types";
import type { LearningTask } from "../src/shared/learning";

const bridge = vi.hoisted(() => ({ current: undefined as undefined | { cursor: number; values: Map<number, unknown> } }));
vi.mock("react", async original => {
  const actual = await original<typeof import("react")>();
  return { ...actual, useState: (initial: unknown) => {
    const owner = bridge.current; if (!owner) return actual.useState(initial);
    const index = owner.cursor++; if (!owner.values.has(index)) owner.values.set(index, typeof initial === "function" ? (initial as () => unknown)() : initial);
    return [owner.values.get(index), (next: unknown) => owner.values.set(index, typeof next === "function" ? (next as (old: unknown) => unknown)(owner.values.get(index)) : next)];
  }, useRef: (initial: unknown) => {
    const owner = bridge.current; if (!owner) return actual.useRef(initial);
    const index = owner.cursor++; if (!owner.values.has(index)) owner.values.set(index, { current: initial }); return owner.values.get(index);
  }, useEffect: (effect: () => void, deps: readonly unknown[]) => bridge.current ? undefined : actual.useEffect(effect, deps),
  useId: () => bridge.current ? "inert-learning-heading" : actual.useId() };
});
afterEach(() => { bridge.current = undefined; vi.unstubAllGlobals(); });
function nodes(value: ReactNode): ReactElement<Record<string, any>>[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!isValidElement(value)) return [];
  const element = value as ReactElement<Record<string, any>>;
  return [element, ...nodes(element.props.children)];
}
const brain = { id: "one", config: { recursiveImprovement: true }, toolPermissions: [{ toolId: "source.self-modify", level: "ask" }] } as BrainDocument;

describe("one Training & Evolution surface", () => {
  it("renders one universal source form and no duplicate evolution goal/kind/recursive controls", () => {
    vi.stubGlobal("window", {});
    const html = renderToStaticMarkup(<LearningWorkspace brainId="one" embedded
      runtimeStatus={<section aria-label="Dataset progress">Saved progress</section>}
      renderImprovement={objective => <EvolutionWorkspace brain={brain} objective={objective} embedded onOpenPermissions={() => {}} onToast={() => {}} />} />);
    expect(html.match(/class="surface learning-workspace__form"/g)).toHaveLength(1);
    expect(html).toContain('aria-label="Add source"'); expect(html).toContain("Files &amp; media"); expect(html).toContain("Folder");
    expect(html.match(/Instructions or improvement goal/g)).toHaveLength(1);
    for (const removed of ["What may change", "Reassess recursively", "Max candidates", "Improvement objective", "Encoded source ledger"])
      expect(html).not.toContain(removed);
    expect(html).toContain("Saved progress"); expect(html).toContain("Runs &amp; candidate lineage");
    expect(html).toContain("Evaluation data"); expect(html).toContain("Start improvement run");
    expect(html).toContain("No tasks yet"); expect(html).toContain("available in the desktop app");
  });

  it("shares the exact edited goal and submits full text, manifest identity and URLs through the real component handler", async () => {
    vi.stubGlobal("requestAnimationFrame", () => 0);
    const manifest = { id: "host-owned-manifest", roots: ["/selected/material"], discoveredFiles: 3001, discoveredBytes: 900000000 } as DatasetManifest;
    const saved: LearningTask = { format: "omni-learning-task", version: 1, id: "learning-job", brainId: "one", requestId: "fixture",
      requestSha256: "fixture-only", kind: "specific-learn", revision: 1, createdAt: "2026-10-20T00:00:00Z", updatedAt: "2026-10-20T00:00:00Z",
      policy: { revision: 1, sha256: "fixture-only", constraints: {}, instructions: "", changedAt: "2026-10-20T00:00:00Z", changedBy: "user" },
      state: "queued", phase: "queued", label: "Queued", epochs: 1, pendingQuestions: [], sources: [], traversalComplete: false };
    const api = { start: vi.fn(async () => saved) } as never;
    const data = { preview: vi.fn(async () => manifest), cancelPreview: vi.fn(), onPreviewProgress: vi.fn(() => () => {}) } as never;
    const improvement = vi.fn(() => null), owner = { cursor: 0, values: new Map<number, unknown>() };
    const render = () => {
      owner.cursor = 0; bridge.current = owner;
      try { const outer = LearningWorkspace({ brainId: "one", api, data, embedded: true, renderImprovement: improvement });
        return (outer.type as (props: typeof outer.props) => ReactNode)(outer.props);
      } finally { bridge.current = undefined; }
    };
    let tree = render();
    const sourceText = "A full ordinary experience. ".repeat(1000), goal = "Improve recall of this experience";
    const textareas = nodes(tree).filter(node => node.type === "textarea");
    textareas[0]!.props.onChange({ target: { value: sourceText } });
    textareas[1]!.props.onChange({ target: { value: "https://docs.example.com/one\nhttps://docs.example.com/two" } });
    textareas[2]!.props.onChange({ target: { value: goal } });
    tree = render(); expect(improvement).toHaveBeenLastCalledWith(goal);
    nodes(tree).find(node => node.type === UniversalSourcePicker)!.props.onSelect("folder");
    await Promise.resolve(); await Promise.resolve();
    expect((data as any).preview).toHaveBeenCalledWith(expect.objectContaining({ brainId: "one", selection: "folder", policy: "pretrain" }));
    tree = render(); nodes(tree).find(node => node.type === "form")!.props.onSubmit({ preventDefault() {} });
    await Promise.resolve(); await Promise.resolve();
    expect((api as any).start).toHaveBeenCalledOnce();
    expect((api as any).start).toHaveBeenCalledWith(expect.objectContaining({ brainId: "one", instructions: goal, inputs: [
      { kind: "text", text: sourceText }, { kind: "dataset", manifestId: manifest.id },
      { kind: "web", url: "https://docs.example.com/one", followLinks: true, followExternalLinks: false, respectRobots: true },
      { kind: "web", url: "https://docs.example.com/two", followLinks: true, followExternalLinks: false, respectRobots: true }
    ] }));
    expect(JSON.stringify((api as any).start.mock.calls[0])).not.toContain("/selected/material");
    render(); expect(improvement).toHaveBeenLastCalledWith(goal);
  });

  it("keeps actual job/coverage/cursor controls without retaining the duplicate data entry or exposed ledgers", () => {
    const app = readFileSync("src/renderer/src/App.tsx", "utf8"), start = app.indexOf("function DataWorkspace({"), end = app.indexOf("\nfunction BrainMapWorkspace", start);
    expect(start).toBeGreaterThan(0); expect(end).toBeGreaterThan(start);
    const surface = app.slice(start, end);
    expect(surface.match(/<LearningWorkspace/g)).toHaveLength(1); expect(surface.match(/<EvolutionWorkspace/g)).toHaveLength(1);
    for (const required of ["api.train.onEvent", "api.data.resumable", "api.data.resume", "api.data.pause", "api.data.cancel", "runtimeJobScopedDatasetCoverage", "TrainingTelemetryMetrics"])
      expect(surface).toContain(required);
    for (const removed of ["DatasetParserPicker", "Encoded source ledger", "source-ledger-pager", "memory-health", "upload-kind-actions", "Demo source encoded", "crawler-form"])
      expect(surface).not.toContain(removed);
  });

  it("keeps parent progress callbacks stable across ordinary output and notice updates", () => {
    const app = readFileSync("src/renderer/src/App.tsx", "utf8");
    const toast = app.slice(app.indexOf("const showToast ="), app.indexOf("const showToast =") + 800);
    const update = app.slice(app.indexOf("const updateActiveBrain ="), app.indexOf("const applyActiveModeResult ="));
    expect(toast.includes("useCallback((message: string)")).toBe(true);
    expect(toast.includes("[appearanceWorkspace.notificationsEnabled]")).toBe(true);
    expect(update.includes("useCallback((document: BrainDocument)")).toBe(true);
    expect(update.trimEnd().endsWith("}, []);")).toBe(true);
  });
});
