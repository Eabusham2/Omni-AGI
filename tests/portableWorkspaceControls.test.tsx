import { describe, expect, it, vi } from "vitest";

const hookBridge = vi.hoisted(() => ({ current: undefined as undefined | {
  useState(initial: unknown): [unknown, (value: unknown) => void];
  useRef(initial: unknown): { current: unknown };
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void;
} }));
vi.mock("react", async importOriginal => {
  const actual = await importOriginal<typeof import("react")>();
  return { ...actual,
    useState: (initial: unknown) => hookBridge.current!.useState(initial),
    useRef: (initial: unknown) => hookBridge.current!.useRef(initial),
    useEffect: (effect: () => void | (() => void), dependencies?: readonly unknown[]) => hookBridge.current!.useEffect(effect, dependencies) };
});

import { isValidElement, type ReactElement, type ReactNode } from "react";
import { PortableWorkspaceControls } from "../src/renderer/src/PortableWorkspaceControls";
import type { BrainSummary, WorkingMemoryResourcePlan } from "../src/shared/types";
import type { PortableDestinationReview, PortablePresentationSettings, PortableWorkspaceApi,
  PortableWorkspaceCreateResult, PortableWorkspaceStatus } from "../src/shared/portableWorkspace";

type Props = Parameters<typeof PortableWorkspaceControls>[0];
type EffectSlot = { dependencies?: readonly unknown[]; cleanup?: void | (() => void) };
class HookHarness {
  private cursor = 0;
  private readonly states = new Map<number, unknown>();
  private readonly refs = new Map<number, { current: unknown }>();
  private readonly effects = new Map<number, EffectSlot>();
  useState(initial: unknown): [unknown, (value: unknown) => void] {
    const index = this.cursor++;
    if (!this.states.has(index)) this.states.set(index, typeof initial === "function" ? (initial as () => unknown)() : initial);
    return [this.states.get(index), value => this.states.set(index,
      typeof value === "function" ? (value as (current: unknown) => unknown)(this.states.get(index)) : value)];
  }
  useRef(initial: unknown): { current: unknown } {
    const index = this.cursor++;
    if (!this.refs.has(index)) this.refs.set(index, { current: initial });
    return this.refs.get(index)!;
  }
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void {
    const index = this.cursor++, previous = this.effects.get(index);
    const changed = !previous || dependencies === undefined || previous.dependencies === undefined ||
      dependencies.length !== previous.dependencies.length || dependencies.some((value, offset) => !Object.is(value, previous.dependencies![offset]));
    if (!changed) return;
    if (typeof previous?.cleanup === "function") previous.cleanup();
    this.effects.set(index, { dependencies: dependencies ? [...dependencies] : undefined, cleanup: effect() });
  }
  render(props: Props): ReactNode {
    this.cursor = 0; hookBridge.current = this;
    try { return PortableWorkspaceControls(props); } finally { hookBridge.current = undefined; }
  }
  unmount(): void { for (const effect of this.effects.values()) if (typeof effect.cleanup === "function") effect.cleanup(); this.effects.clear(); }
}

const BRAIN_ID = "portable-test-brain", GIB = 1024 ** 3;
const PRESENTATION: PortablePresentationSettings = { appearance: { schemaVersion: 1, mode: "dark", palette: "aqua", layout: "glass" },
  workspace: { schemaVersion: 1, arrangement: "split", notificationsEnabled: true, profiles: [] }, theme: null };
const SUMMARIES = [{ id: BRAIN_ID, name: "Portable test", preset: "micro", runtime: "cpu", updatedAt: "2026-10-15T12:00:00Z",
  concepts: 0, synapses: 0, activeMode: false }] as unknown as BrainSummary[];
const status = (state: PortableWorkspaceStatus["state"], patch: Partial<PortableWorkspaceStatus> = {}): PortableWorkspaceStatus => ({
  portable: true, state, workspaceId: "workspace-test", brainId: BRAIN_ID, name: "Portable test", ...patch
});
const plan = (allowed: boolean): WorkingMemoryResourcePlan => ({ schemaVersion: 1, allowed,
  blockers: allowed ? [] : ["Synthetic fit blocker"], warnings: ["Synthetic fit warning"],
  context: { selectedTokens: 131072 }, resources: { systemRamBudgetBytes: 24 * GIB, sharedStoragePoolBytes: 96 * GIB } } as WorkingMemoryResourcePlan);
const review = (allowed: boolean): PortableDestinationReview => ({ reviewId: allowed ? "allowed-review" : "blocked-review",
  reviewSha256: (allowed ? "a" : "b").repeat(64), workspaceId: "workspace-test", brainId: BRAIN_ID, plan: plan(allowed),
  sourceDriveSelectionsDiscarded: true, copiedExternalAuthorityDormant: true, credentialsTransferred: false });
function api(overrides: Partial<PortableWorkspaceApi> = {}): PortableWorkspaceApi {
  return { status: vi.fn(async () => ({ portable: false, state: "installed" })), create: vi.fn(async () => null),
    prepare: vi.fn(async () => status("preparing")), reviewDestination: vi.fn(async () => review(false)),
    approveDestination: vi.fn(async () => status("ready")), ...overrides };
}
function props(value: PortableWorkspaceApi, onCreated = vi.fn(), onEditSettings = vi.fn()): Props {
  return { api: value, summaries: SUMMARIES, presentation: () => PRESENTATION, onCreated, onEditSettings };
}
function deferred<T>() { let resolve!: (value: T) => void, reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const flush = async (): Promise<void> => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
function childNodes(node: ReactNode): ReactNode[] {
  if (node === null || node === undefined || typeof node === "boolean") return [];
  if (Array.isArray(node)) return node;
  if (isValidElement(node)) { const children = (node.props as { children?: ReactNode }).children; return children === undefined ? [] : Array.isArray(children) ? children : [children]; }
  return [];
}
function elements(node: ReactNode): ReactElement[] {
  const found: ReactElement[] = []; if (isValidElement(node)) found.push(node);
  for (const child of childNodes(node)) found.push(...elements(child)); return found;
}
function copy(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  return childNodes(node).map(copy).join("");
}
function button(node: ReactNode, label: string): ReactElement {
  const found = elements(node).find(element => element.type === "button" && copy(element).trim() === label);
  if (!found) throw new Error(`Missing button: ${label}`); return found;
}
function select(node: ReactNode): ReactElement {
  const found = elements(node).find(element => element.type === "select"); if (!found) throw new Error("Missing portable instance selector."); return found;
}
const click = (element: ReactElement): void => (element.props as { onClick(): void }).onClick();

describe("portable workspace controls with a synthetic typed API", () => {
  it("clears the previous API owner immediately and ignores late status/create acknowledgements", async () => {
    const aStatus = deferred<PortableWorkspaceStatus>(), bStatus = deferred<PortableWorkspaceStatus>();
    const first = api({ status: vi.fn(() => aStatus.promise) }), second = api({ status: vi.fn(() => bStatus.promise) });
    const harness = new HookHarness(), created = vi.fn();
    harness.render(props(first, created)); harness.render(props(second, created));
    let tree = harness.render(props(second, created));
    expect(copy(tree)).not.toContain("Prepare the saved files");
    aStatus.resolve(status("preparation-required")); await flush(); tree = harness.render(props(second, created));
    expect(copy(tree)).not.toContain("Prepare the saved files");
    bStatus.resolve(status("ready")); await flush(); tree = harness.render(props(second, created));
    expect(copy(tree)).toContain("This destination is ready");

    const createAck = deferred<PortableWorkspaceCreateResult | null>();
    const third = api({ create: vi.fn(() => createAck.promise) }), fourth = api();
    const actionHarness = new HookHarness(); actionHarness.render(props(third, created)); await flush();
    tree = actionHarness.render(props(third, created));
    (select(tree).props as { onChange(event: { target: { value: string } }): void }).onChange({ target: { value: BRAIN_ID } });
    tree = actionHarness.render(props(third, created)); click(button(tree, "Create portable folder"));
    actionHarness.render(props(fourth, created)); actionHarness.render(props(fourth, created));
    createAck.resolve({ destination: "synthetic", workspaceId: "late", brainId: BRAIN_ID, files: 1, bytes: 1, destinationReviewRequired: true });
    await flush(); expect(created).not.toHaveBeenCalled(); actionHarness.unmount(); harness.unmount();
  });

  it("presents preparing, blocked review, allowed review and ready without treating a preview as approval", async () => {
    const value = api({
      status: vi.fn().mockResolvedValueOnce(status("preparation-required")).mockResolvedValueOnce(status("review-required")),
      prepare: vi.fn(async () => status("preparing", { label: "Preparing exact synthetic bytes…" })),
      reviewDestination: vi.fn().mockResolvedValueOnce(review(false)).mockResolvedValueOnce(review(true)),
      approveDestination: vi.fn(async () => status("ready"))
    });
    const harness = new HookHarness(); harness.render(props(value)); await flush(); let tree = harness.render(props(value));
    click(button(tree, "Prepare saved files")); tree = harness.render(props(value));
    expect(copy(tree)).toContain("Preparing saved files…"); await flush(); tree = harness.render(props(value));
    expect(copy(tree)).toContain("Preparing exact synthetic bytes…");
    click(button(tree, "Refresh workspace")); await flush(); tree = harness.render(props(value));
    expect(copy(tree)).toContain("Review this PC before loading");
    click(button(tree, "Review destination")); await flush(); tree = harness.render(props(value));
    expect(copy(tree)).toContain("24 GiB"); expect(copy(tree)).toContain("96 GiB"); expect(copy(tree)).toContain("131,072 tokens");
    expect(copy(tree)).toContain("Synthetic fit blocker"); expect((button(tree, "Approve this resource fit").props as { disabled?: boolean }).disabled).toBe(true);
    click(button(tree, "Review destination")); await flush(); tree = harness.render(props(value));
    expect(copy(tree)).not.toContain("Synthetic fit blocker"); expect((button(tree, "Approve this resource fit").props as { disabled?: boolean }).disabled).toBe(false);
    click(button(tree, "Approve this resource fit")); await flush(); tree = harness.render(props(value));
    expect(copy(tree)).toContain("This destination is ready"); harness.unmount();
  });

  it("keeps picker cancellation quiet and shows only the current synthetic create failure", async () => {
    const create = vi.fn().mockResolvedValueOnce(null).mockRejectedValueOnce(new Error("Synthetic create failed"));
    const value = api({ create }), harness = new HookHarness(), created = vi.fn();
    harness.render(props(value, created)); await flush(); let tree = harness.render(props(value, created));
    (select(tree).props as { onChange(event: { target: { value: string } }): void }).onChange({ target: { value: BRAIN_ID } });
    tree = harness.render(props(value, created)); click(button(tree, "Create portable folder")); await flush(); tree = harness.render(props(value, created));
    expect(created).not.toHaveBeenCalled(); expect(copy(tree)).not.toContain("Synthetic create failed");
    click(button(tree, "Create portable folder")); await flush(); tree = harness.render(props(value, created));
    expect(copy(tree)).toContain("Synthetic create failed"); expect(created).not.toHaveBeenCalled(); harness.unmount();
  });
});
