/** Inert command adapter only; never enumerate, signal or inspect actual processes. */
import { afterEach, expect, it, vi } from "vitest";
const execute = vi.hoisted(() => vi.fn());
vi.mock("node:child_process", async () => {
  const { promisify } = await import("node:util");
  return { execFile: Object.assign(() => {}, { [promisify.custom]: (...args: unknown[]) => execute(...args) }) };
});
import { automationResidencySample, automationResidentBytes } from "../src/main/automationResources";
afterEach(() => { vi.restoreAllMocks(); execute.mockReset(); });

it("reports actual observed PID count separately from requested known owners and keeps the old RSS-sum wrapper", async () => {
  execute.mockResolvedValue({ stdout: "111 20\n333 30\n", stderr: "" });
  const missing = vi.spyOn(process, "kill").mockImplementation((pid, signal) => {
    expect(pid).toBe(222); expect(signal).toBe(0); throw Object.assign(new Error("Fixture process already exited"), { code: "ESRCH" });
  });
  expect(await automationResidencySample([111, 222, 333, 111], "linux")).toEqual({ residentBytes: 51200, requestedProcessCount: 3, sampledProcessCount: 2 });
  expect(execute).toHaveBeenCalledWith("ps", ["-o", "pid=,rss=", "-p", "111,222,333"], { timeout: 3000, maxBuffer: 65536 });
  expect(await automationResidentBytes([111, 222, 333], "linux")).toBe(51200); expect(missing).toHaveBeenCalledTimes(2);
});

it("does not disguise a missing live owner's measurement as zero RSS", async () => {
  execute.mockResolvedValue({ stdout: "111 20\n", stderr: "" });
  vi.spyOn(process, "kill").mockReturnValue(true);
  await expect(automationResidencySample([111, 222], "linux")).rejects.toThrow("A live owned process has no residency measurement");
});

it("cancels before dispatch or through the exact existing bounded command without publishing a partial sample", async () => {
  const pre = new AbortController(); pre.abort();
  await expect(automationResidencySample([111], "linux", pre.signal)).rejects.toMatchObject({ name: "AbortError" });
  expect(execute).not.toHaveBeenCalled();
  const running = new AbortController(); let entered!: () => void;
  const started = new Promise<void>(done => { entered = done; });
  execute.mockImplementation(async (_file, _args, options: { signal: AbortSignal }) => new Promise((_resolve, reject) => {
    entered(); options.signal.addEventListener("abort", () => reject(new DOMException("Fixture OS query cancelled", "AbortError")), { once: true });
  }));
  const request = automationResidencySample([111], "win32", running.signal);
  const rejected = expect(request).rejects.toMatchObject({ name: "AbortError" });
  await started; running.abort(); await rejected;
  expect(execute.mock.calls[0]![0]).toBe("powershell.exe");
  expect(execute.mock.calls[0]![2]).toEqual({ signal: running.signal, timeout: 3000, maxBuffer: 65536 });
});
