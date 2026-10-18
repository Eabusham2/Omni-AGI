import React, { isValidElement, type ReactElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { UserQuestionPanel, UserQuestionCard, UserQuestionHistory, UserQuestionSettings } from "../src/renderer/src/UserQuestionPanel";
import type { UserQuestionEvent, UserQuestionRecord, UserQuestionsApi } from "../src/shared/userQuestions";

const hookBridge = vi.hoisted(() => ({ current: undefined as undefined | {
  useState(initial: unknown): [unknown, (value: unknown) => void];
  useRef(initial: unknown): { current: unknown };
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void;
  memo(value: unknown, dependencies?: readonly unknown[]): unknown;
  useId(): string;
} }));
vi.mock("react", async original => {
  const actual = await original<typeof import("react")>();
  return { ...actual,
    useState: (initial: unknown) => hookBridge.current ? hookBridge.current.useState(initial) : actual.useState(initial),
    useRef: (initial: unknown) => hookBridge.current ? hookBridge.current.useRef(initial) : actual.useRef(initial),
    useEffect: (effect: () => void | (() => void), dependencies?: readonly unknown[]) => hookBridge.current
      ? hookBridge.current.useEffect(effect, dependencies) : actual.useEffect(effect, dependencies),
    useCallback: (value: (...args: never[]) => unknown, dependencies: readonly unknown[]) => hookBridge.current
      ? hookBridge.current.memo(value, dependencies) : actual.useCallback(value, dependencies),
    useId: () => hookBridge.current ? hookBridge.current.useId() : actual.useId()
  };
});

const question = (overrides: Partial<UserQuestionRecord> = {}): UserQuestionRecord => ({
  format: "omni-user-question", version: 1, id: "fixture", brainId: "one", requestId: "request", requestSha256: "fixture-only",
  kind: "task-choice", prompt: "Which sections?", detail: "Full details remain available.\nDo not lose the long explanation.",
  mode: "multiple", minSelections: 2, maxSelections: 2, allowOther: true,
  options: [{ id: "intro", label: "Introduction", recommended: true, rank: 1 }, { id: "examples", label: "Examples", rank: 2 }],
  createdAt: "2026-10-02T12:00:00.000Z", updatedAt: "2026-10-02T12:00:00.000Z", status: "pending",
  timeout: { enabled: false, seconds: 1200, policyRevision: "default" }, ...overrides
});
const actions = { onRespond: async () => {}, onSkip: async () => {}, onResume: async () => {} };

type PanelProps = Parameters<typeof UserQuestionPanel>[0];
class QuestionHarness {
  private cursor = 0;
  private key: string | null | undefined;
  private readonly values = new Map<number, unknown>();
  private readonly effects = new Map<number, { dependencies?: readonly unknown[]; cleanup?: void | (() => void) }>();
  private changed(a: readonly unknown[] | undefined, b: readonly unknown[] | undefined): boolean {
    return !a || !b || a.length !== b.length || a.some((value, index) => !Object.is(value, b[index]));
  }
  useState(initial: unknown): [unknown, (value: unknown) => void] {
    const index = this.cursor++;
    if (!this.values.has(index)) this.values.set(index, typeof initial === "function" ? (initial as () => unknown)() : initial);
    return [this.values.get(index), value => this.values.set(index, typeof value === "function"
      ? (value as (previous: unknown) => unknown)(this.values.get(index)) : value)];
  }
  useRef(initial: unknown): { current: unknown } {
    const index = this.cursor++;
    if (!this.values.has(index)) this.values.set(index, { current: initial });
    return this.values.get(index) as { current: unknown };
  }
  memo(value: unknown, dependencies?: readonly unknown[]): unknown {
    const index = this.cursor++, previous = this.values.get(index) as { value: unknown; dependencies?: readonly unknown[] } | undefined;
    if (!previous || this.changed(previous.dependencies, dependencies)) this.values.set(index, { value, dependencies });
    return (this.values.get(index) as { value: unknown }).value;
  }
  useId(): string { return this.useRef("synthetic-question-heading").current as string; }
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void {
    const index = this.cursor++, previous = this.effects.get(index);
    if (previous && !this.changed(previous.dependencies, dependencies)) return;
    if (typeof previous?.cleanup === "function") previous.cleanup();
    this.effects.set(index, { dependencies, cleanup: effect() });
  }
  render(props: PanelProps): ReactNode {
    const element = UserQuestionPanel(props);
    if (this.key !== element.key) { this.unmount(); this.key = element.key; }
    this.cursor = 0; hookBridge.current = this;
    try { return (element.type as (value: PanelProps) => ReactNode)(element.props); }
    finally { hookBridge.current = undefined; }
  }
  unmount(): void {
    for (const effect of this.effects.values()) if (typeof effect.cleanup === "function") effect.cleanup();
    this.effects.clear(); this.values.clear();
  }
}
function nodes(node: ReactNode): ReactElement[] {
  if (Array.isArray(node)) return node.flatMap(nodes);
  if (!isValidElement(node)) return [];
  return [node, ...nodes((node.props as { children?: ReactNode }).children)];
}
function deferred<T>() { let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; }); return { promise, resolve }; }
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
function questionApi(overrides: Partial<UserQuestionsApi> = {}) {
  const listeners = new Set<(event: UserQuestionEvent) => void>();
  const api: UserQuestionsApi = {
    request: vi.fn(async () => question()), list: vi.fn(async () => []),
    respond: vi.fn(async request => question({ id: request.questionId, status: "answered", resolution: {
      id: "answer", source: "user", resolvedAt: "2026-10-02T12:01:00.000Z", response: request.response,
      authority: "task-choice-only", permissionsGranted: false
    } })),
    cancel: vi.fn(async () => question({ status: "cancelled" })), resume: vi.fn(async () => []),
    preferences: vi.fn(async () => ({ timeoutEnabled: false, timeoutSeconds: 1200 })),
    setPreferences: vi.fn(async value => ({ timeoutEnabled: value.timeoutEnabled, timeoutSeconds: value.timeoutSeconds })),
    onEvent: vi.fn(listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }), ...overrides
  };
  return { api, emit: (event: UserQuestionEvent) => { for (const listener of listeners) listener(event); }, listeners };
}

describe("ordinary conversation question surface (inert React/API contracts)", () => {
  it("hides an empty conversation, opens actual pending choices, and leaves learning questions in their task", async () => {
    const host = questionApi(), harness = new QuestionHarness();
    const props: PanelProps = { brainId: "one", api: host.api, presentation: "conversation" };
    expect(harness.render(props)).toBeNull(); await flush(); expect(harness.render(props)).toBeNull();
    host.emit({ type: "created", brainId: "one", question: question({ source: { kind: "learn" } }) });
    expect(harness.render(props)).toBeNull();
    host.emit({ type: "created", brainId: "one", question: question({ id: "native-question", source: { kind: "brain", turnId: "current-turn" } }) });
    harness.render(props); const tree = harness.render(props);
    const cards = nodes(tree).filter(element => element.type === UserQuestionCard);
    expect(cards).toHaveLength(1);
    expect((cards[0]!.props as { question: UserQuestionRecord }).question.id).toBe("native-question");
    expect((nodes(tree).find(element => element.type === "details")!.props as { open: boolean }).open).toBe(true);
    expect(host.api.respond).not.toHaveBeenCalled(); expect(host.api.resume).not.toHaveBeenCalled(); harness.unmount();
  });

  it("keeps stream events across the initial scan, then retires an old API and its late answer without a new authority", async () => {
    const scan = deferred<UserQuestionRecord[]>(), answer = deferred<UserQuestionRecord>(), resolved = vi.fn();
    const first = questionApi({ list: vi.fn(() => scan.promise), respond: vi.fn(() => answer.promise) });
    const second = questionApi(), harness = new QuestionHarness();
    const firstProps: PanelProps = { brainId: "one", api: first.api, presentation: "conversation", onResolved: resolved };
    harness.render(firstProps); first.emit({ type: "created", brainId: "one", question: question({ source: { kind: "brain" } }) });
    scan.resolve([]); await flush();
    let tree = harness.render(firstProps);
    const card = nodes(tree).find(element => element.type === UserQuestionCard)!;
    expect(card).toBeDefined();
    const waiting = (card.props as { onRespond(value: { optionIds: string[] }): Promise<void> }).onRespond({ optionIds: ["intro", "examples"] });
    const oldKey = UserQuestionPanel(firstProps).key;
    const secondProps = { ...firstProps, api: second.api };
    expect(UserQuestionPanel(secondProps).key).not.toBe(oldKey);
    expect(harness.render(secondProps)).toBeNull();
    expect(first.listeners.size).toBe(0);
    answer.resolve(question({ status: "answered", resolution: { id: "late", source: "user", resolvedAt: "2026-10-02T12:01:00.000Z",
      response: { optionIds: ["intro", "examples"] }, authority: "task-choice-only", permissionsGranted: false } }));
    await waiting; await flush(); tree = harness.render(secondProps);
    expect(tree).toBeNull(); expect(resolved).not.toHaveBeenCalled();
    expect(second.api.respond).not.toHaveBeenCalled(); expect(second.api.setPreferences).not.toHaveBeenCalled(); harness.unmount();
  });

  it("reveals current errors with Retry and refuses a receipt for a different question", async () => {
    const host = questionApi({ list: vi.fn().mockRejectedValueOnce(new Error("Synthetic question read failed")).mockResolvedValue([question()]),
      respond: vi.fn(async () => question({ id: "another-question", status: "answered" })) });
    const resolved = vi.fn(), harness = new QuestionHarness();
    const props: PanelProps = { brainId: "one", api: host.api, presentation: "conversation", onResolved: resolved };
    harness.render(props); await flush(); harness.render(props); let tree = harness.render(props);
    const retry = nodes(tree).find(element => element.type === "button" && (element.props as { children?: ReactNode }).children === "Retry")!;
    expect(retry).toBeDefined();
    (retry.props as { onClick(): void }).onClick(); harness.render(props); await flush(); tree = harness.render(props);
    const card = nodes(tree).find(element => element.type === UserQuestionCard)!;
    await expect((card.props as { onRespond(value: { optionIds: string[] }): Promise<void> }).onRespond({ optionIds: ["intro", "examples"] }))
      .rejects.toThrow("another question");
    expect(resolved).not.toHaveBeenCalled(); harness.unmount();
  });
});

describe("accessible question panel markup", () => {
  it("associates the question, checkbox choices, bounds and expandable detail with an answer form", () => {
    const markup = renderToStaticMarkup(<UserQuestionCard question={question()} now={0} {...actions} />);
    expect(markup).toContain("<fieldset");
    expect(markup).toContain("<legend");
    expect(markup).toContain("aria-labelledby=");
    expect(markup).toContain("aria-describedby=");
    expect(markup.match(/type="checkbox"/g)).toHaveLength(3);
    expect(markup).toContain("Choose 2 options.");
    expect(markup).toContain("Recommended");
    expect(markup).toContain("Other answer");
    expect(markup).toContain("<summary>Details</summary>");
    expect(markup).toContain("Do not lose the long explanation.");
    expect(markup).toMatch(/type="submit"[^>]*disabled/);
  });

  it("uses radio controls for single choice and a labelled textarea for free responses", () => {
    const radio = renderToStaticMarkup(<UserQuestionCard question={question({ mode: "single", minSelections: 1, maxSelections: 1 })} now={0} {...actions} />);
    expect(radio.match(/type="radio"/g)).toHaveLength(3);
    const written = renderToStaticMarkup(<UserQuestionCard question={question({ mode: "text", options: [], minSelections: 0, maxSelections: 0, allowOther: false })} now={0} {...actions} />);
    expect(written).toContain("Your answer");
    expect(written).toContain("<textarea");
    expect(written).not.toContain('type="radio"');
    expect(written).not.toContain('type="checkbox"');
  });

  it("keeps historical assumptions collapsed and accurately labelled", () => {
    const markup = renderToStaticMarkup(<UserQuestionHistory question={question({ status: "assumed", resolution: {
      id: "resolution", source: "timeout", resolvedAt: "2026-10-02T12:20:00.000Z", response: { optionIds: ["intro"] }, authority: "task-choice-only", permissionsGranted: false
    } })} />);
    expect(markup).toContain("Assumed after timeout");
    expect(markup).toContain("Introduction");
    expect(markup).not.toContain("Answered by you");
    expect(markup).not.toMatch(/<details[^>]* open/);
  });

  it("presents paused copies without active response controls and preserves the default disabled timeout", () => {
    const paused = renderToStaticMarkup(<UserQuestionCard question={question({ suspended: { reason: "copied-instance", fromBrainId: "parent", copiedAt: "2026-10-02T12:00:00.000Z" } })} now={0} {...actions} />);
    expect(paused).toMatch(/<fieldset[^>]*disabled/);
    expect(paused).toContain("Resume question");
    expect(paused).not.toContain("Send answer");
    const settings = renderToStaticMarkup(<UserQuestionSettings preferences={{ timeoutEnabled: false, timeoutSeconds: 1200 }} onSave={async (value) => value} />);
    expect(settings).toContain("Minutes");
    expect(settings).toContain("Seconds");
    expect(settings).toContain('value="20"');
    expect(settings).toContain('value="0"');
    expect(settings).not.toMatch(/type="checkbox"[^>]*checked/);
    expect(settings).toContain("Permissions always use their own approval.");
  });
});
