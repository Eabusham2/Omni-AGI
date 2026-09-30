import type { PreparedVideoRuntime, VideoRuntimeProgress } from "./videoRuntimeProvisioner";

export interface CodecRuntimeOwner {
  requestId: string; brainId: string; jobId: string; streamId: string; actionId: string;
}
export interface CodecRuntimeChallenge extends CodecRuntimeOwner { challengeId: string; purpose: string; }
export interface CodecRuntimeReceipt extends CodecRuntimeOwner {
  challengeId: string; outcome: "ready" | "external" | "failed" | "cancelled";
  configuration?: Record<string, unknown>; reason?: string;
}
const id = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const purposes = new Set(["decode-live-audio", "decode-audio", "decode-video", "inspect-video-audio", "encode-video"]);
export function codecRuntimeChallenge(value: unknown): CodecRuntimeChallenge | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const input = value as Record<string, unknown>;
  if (Object.keys(input).sort().join(",") !== "actionId,brainId,challengeId,jobId,purpose,requestId,streamId" ||
      typeof input.challengeId !== "string" || !/^[a-f0-9]{32}$/.test(input.challengeId) ||
      typeof input.purpose !== "string" || !purposes.has(input.purpose)) return undefined;
  for (const field of ["requestId", "brainId", "jobId", "streamId", "actionId"]) {
    const value = input[field];
    if (typeof value !== "string" || ((!value && ["requestId", "brainId"].includes(field)) || (value && !id.test(value)))) return undefined;
  }
  return input as unknown as CodecRuntimeChallenge;
}
export function sameCodecOwner(left: CodecRuntimeOwner, right: CodecRuntimeOwner): boolean {
  return ["requestId", "brainId", "jobId", "streamId", "actionId"].every((field) =>
    left[field as keyof CodecRuntimeOwner] === right[field as keyof CodecRuntimeOwner]);
}
interface Waiting {
  challenge: CodecRuntimeChallenge; signal: AbortSignal; abort(): void;
  send(receipt: CodecRuntimeReceipt): Promise<void>;
  progress(progress: VideoRuntimeProgress): void;
  cancelled: boolean; delivered: boolean;
}
interface Preparation { controller: AbortController; waiters: Set<Waiting>; promise: Promise<PreparedVideoRuntime>; }

/** Main-only pinned provisioner bridge; never enters the neural request queue. */
export class CodecRuntimeSetupBridge {
  private readonly waiting = new Map<string, Waiting>();
  private preparation?: Preparation;
  constructor(private readonly prepare: (signal: AbortSignal, progress: (value: VideoRuntimeProgress) => void) => Promise<PreparedVideoRuntime>) {}

  has(requestId: string): boolean {
    return [...this.waiting.values()].some((entry) => !entry.cancelled && entry.challenge.requestId === requestId && !entry.challenge.actionId);
  }
  cancel(predicate: (owner: CodecRuntimeOwner) => boolean): boolean {
    let cancelled = false;
    for (const entry of this.waiting.values()) if (predicate(entry.challenge)) { entry.abort(); cancelled = true; }
    return cancelled;
  }
  release(challengeId: string, owner: CodecRuntimeOwner): void {
    const entry = this.waiting.get(challengeId);
    if (!entry || !sameCodecOwner(entry.challenge, owner)) return;
    entry.cancelled = true;
    this.remove(entry);
  }
  private remove(entry: Waiting): void {
    entry.signal.removeEventListener("abort", entry.abort);
    if (this.waiting.get(entry.challenge.challengeId) === entry) this.waiting.delete(entry.challenge.challengeId);
    const group = this.preparation;
    group?.waiters.delete(entry);
    if (group && !group.waiters.size && !group.controller.signal.aborted) group.controller.abort(new Error("All owned codec setup waits ended."));
  }

  accept(challenge: CodecRuntimeChallenge, signal: AbortSignal,
    send: Waiting["send"], progress: Waiting["progress"]): void {
    if (this.waiting.has(challenge.challengeId)) throw new Error("Duplicate codec runtime challenge.");
    let entry!: Waiting;
    const deliver = async (outcome: CodecRuntimeReceipt["outcome"], extra: Pick<CodecRuntimeReceipt, "reason" | "configuration">): Promise<void> => {
      if (entry.delivered && outcome !== "cancelled") return;
      entry.delivered = true;
      const { purpose: _purpose, ...owner } = challenge;
      try { await send({ ...owner, outcome, ...extra }); }
      catch (error) { entry.delivered = false; throw error; }
    };
    entry = { challenge, signal, send, progress, cancelled: false, delivered: false, abort: () => {
      if (entry.cancelled) return;
      entry.cancelled = true;
      this.remove(entry);
      void deliver("cancelled", { reason: "This exact codec setup owner was cancelled." }).catch(() => undefined);
    } };
    this.waiting.set(challenge.challengeId, entry);
    signal.addEventListener("abort", entry.abort, { once: true });
    if (signal.aborted) { entry.abort(); return; }
    let group = this.preparation;
    if (!group || group.controller.signal.aborted) {
      const previous = group?.promise;
      const controller = new AbortController();
      const waiters = new Set<Waiting>();
      const promise = Promise.resolve(previous).catch(() => undefined).then(() =>
        this.prepare(controller.signal, (value) => { for (const waiting of waiters) if (!waiting.cancelled) waiting.progress(value); }));
      group = { controller, waiters, promise };
      this.preparation = group;
      void promise.finally(() => { if (this.preparation === group) this.preparation = undefined; }).catch(() => undefined);
    }
    group.waiters.add(entry);
    void group.promise.then(async (prepared) => {
      if (entry.cancelled || signal.aborted) return;
      if (prepared.state === "external") {
        await deliver("external", { reason: "Use only the worker's existing external FFmpeg selection." });
      } else {
        if (!prepared.artifactSha256 || !prepared.binarySha256 || !prepared.binarySizeBytes || !prepared.target) throw new Error("Codec setup omitted its pinned binary identity.");
        await deliver("ready", { configuration: { executablePath: prepared.executablePath,
          artifactSha256: prepared.artifactSha256, binarySha256: prepared.binarySha256,
          binarySizeBytes: prepared.binarySizeBytes, target: prepared.target } });
      }
    }).catch(async (error: unknown) => {
      if (!entry.cancelled) await deliver("failed", { reason: (error instanceof Error ? error.message : String(error)).slice(0, 4000) }).catch(() => undefined);
    });
    // Keep ownership until the worker releases the challenge after verification
    // and lease selection. Parent Stop can cancel that boundary wait as well.
  }
}
