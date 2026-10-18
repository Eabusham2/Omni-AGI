import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { BackgroundRequestDeferredError, ENGINE_REQUEST_NO_DEADLINE, EngineSupervisor } from "../src/main/engineSupervisor";
import { nativeAppObservationFrontierSha256 } from "../src/main/nativeAppObservation";
import type { NativeAppObservationHead, NativeAppObservationInput } from "../src/shared/nativeAppObservation";

class SyntheticWorker extends EventEmitter {
  pid = 4242; killed = false; exitCode: number | null = null;
  stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  stdin = { writable: true, write: vi.fn((_line: string, callback?: (error?: Error | null) => void) => { callback?.(null); return true; }) };
  kill = vi.fn(() => true);
}
function fixture() {
  const worker = new SyntheticWorker(), sendSignal = vi.fn(), engine = new EngineSupervisor({ appPath: process.cwd(), sendSignal });
  (engine as unknown as { child: ChildProcessWithoutNullStreams }).child = worker as unknown as ChildProcessWithoutNullStreams;
  const start = vi.spyOn(engine, "start").mockResolvedValue(true);
  const reply = (index: number, result: unknown) => {
    const request = JSON.parse(String(worker.stdin.write.mock.calls[index]?.[0]));
    (engine as unknown as { consumeLine(raw: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }));
  };
  return { engine, worker, start, reply, sendSignal };
}
function syntheticAdmission(): { input: NativeAppObservationInput; head: NativeAppObservationHead } {
  const head: NativeAppObservationHead = { format: "omni-native-app-observation-head", version: 1,
    streamId: "11111111-1111-4111-8111-111111111111", ordinal: 1, recordSha256: "a".repeat(64), dataSha256: "b".repeat(64),
    previousSha256: "0".repeat(64), baseFrontierSha256: "0".repeat(64), disposition: "observe", state: "observed",
    committedAt: new Date().toISOString(), learning: { kind: "experience", source: "mcp-app", sourceLabel: "Synthetic fixture",
      steps: 0, experienceCountBefore: 0, experienceCountAfter: 1, assemblyId: "synthetic-only", afterimageId: "", textSha256: "c".repeat(64), textBytes: 1 } };
  return { head, input: { brainId: "brain", recordJson: JSON.stringify({ streamId: head.streamId, ordinal: head.ordinal,
    previousSha256: head.previousSha256, content: { sha256: head.dataSha256 } }), recordSha256: head.recordSha256,
    dataJson: "synthetic validated native input", dataSha256: head.dataSha256, expectedFrontierSha256: head.baseFrontierSha256, disposition: "observe" } };
}
describe("loaded-only serial App observation bridge (synthetic process)", () => {
  it("never starts or loads a cold worker for a queued observation", async () => {
    const engine = new EngineSupervisor({ appPath: process.cwd() }), start = vi.spyOn(engine, "start");
    await expect(engine.appObservationFrontier("brain", "/owned/brain")).rejects.toThrow("explicitly loaded");
    expect(start).not.toHaveBeenCalled();
  });
  it("uses the existing FIFO writer and exact loaded boundary without starting a replacement", async () => {
    const f = fixture(), boundary = vi.fn();
    const pending = f.engine.appObservationFrontier("brain", "/owned/brain", undefined, boundary);
    await vi.waitFor(() => expect(f.worker.stdin.write).toHaveBeenCalledOnce());
    expect(JSON.parse(String(f.worker.stdin.write.mock.calls[0]![0]))).toMatchObject({ method: "app_observation_frontier", params: { brainId: "brain", storagePath: "/owned/brain" } });
    f.reply(0, { format: "omni-native-app-observation-frontier", version: 1, brainId: "brain", head: null, frontierSha256: "0".repeat(64) });
    await expect(pending).resolves.toMatchObject({ head: null }); expect(f.start).not.toHaveBeenCalled(); expect(boundary).toHaveBeenCalledOnce();
    await expect(f.engine.appObservationFrontier("brain", "/owned/brain", undefined, () => { throw new Error("Explicit Unload occurred"); })).rejects.toThrow("Unload occurred");
    expect(f.worker.stdin.write).toHaveBeenCalledOnce(); expect(f.worker.kill).not.toHaveBeenCalled();
  });
  it("defers behind an existing turn instead of entering a reader-side writer cycle", async () => {
    const f = fixture(), chat = f.engine.request("chat", { brainId: "brain" }, ENGINE_REQUEST_NO_DEADLINE);
    await vi.waitFor(() => expect(f.worker.stdin.write).toHaveBeenCalledOnce());
    await expect(f.engine.appObservationFrontier("brain", "/owned/brain")).rejects.toBeInstanceOf(BackgroundRequestDeferredError);
    expect(f.worker.stdin.write).toHaveBeenCalledOnce(); f.reply(0, { fixture: "existing turn settled" }); await chat;
    expect(f.worker.kill).not.toHaveBeenCalled();
  });
  it("foreground preemption cooperates without killing a committed App observation or discarding its ACK", async () => {
    const f = fixture(), value = syntheticAdmission(), states: string[] = [];
    f.engine.on("activity", event => { states.push(event.state); });
    const pending = f.engine.admitAppObservation(value.input, "/owned/brain");
    await vi.waitFor(() => expect(f.worker.stdin.write).toHaveBeenCalledOnce());
    const foreground = f.engine.claimForeground();
    expect(f.sendSignal).toHaveBeenCalledWith(f.worker.pid, process.platform === "win32" ? "SIGBREAK" : "SIGUSR1");
    expect(f.worker.kill).not.toHaveBeenCalled();
    f.reply(0, { format: "omni-native-app-observation-frontier", version: 1, brainId: "brain", head: value.head,
      frontierSha256: nativeAppObservationFrontierSha256(value.head), duplicate: false });
    await expect(pending).resolves.toMatchObject({ head: { state: "observed" } }); await foreground;
    expect(states.at(-1)).toBe("complete"); expect(f.start).not.toHaveBeenCalled(); expect(f.worker.kill).not.toHaveBeenCalled();
  });
  it("labels only a correlated unmutated native read cancellation as cancelled without releasing or replacing RAM", async () => {
    const f = fixture(), value = syntheticAdmission(), states: string[] = [], controller = new AbortController();
    f.engine.on("activity", event => { states.push(event.state); });
    const pending = f.engine.admitAppObservation(value.input, "/owned/brain", controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({ code: -32800 });
    await vi.waitFor(() => expect(f.worker.stdin.write).toHaveBeenCalledOnce()); controller.abort();
    const request = JSON.parse(String(f.worker.stdin.write.mock.calls[0]![0]));
    (f.engine as unknown as { consumeLine(raw: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: request.id,
      error: { code: -32800, message: "Synthetic read cancelled", data: { brainId: "brain", nativeAppObservation: true,
        cancelled: true, safeBoundary: true, learningApplied: false, nativeOwnerReleased: false } } }));
    await rejected; expect(states.at(-1)).toBe("cancelled"); expect(f.start).not.toHaveBeenCalled(); expect(f.worker.kill).not.toHaveBeenCalled();
  });
});
