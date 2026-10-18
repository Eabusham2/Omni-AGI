import type { ChildProcess } from "node:child_process";

type OwnedChild = Pick<ChildProcess, "pid" | "exitCode" | "signalCode"> & {
  once(event: "exit", listener: () => void): unknown;
  removeListener(event: "exit", listener: () => void): unknown;
  kill(signal: "SIGKILL"): boolean;
};
export interface OwnedPackagedApplication {
  close(): Promise<void>;
}

async function bounded<T>(work: Promise<T>, milliseconds: number, label: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([work, new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded ${milliseconds} ms.`)), milliseconds);
    })]);
  } finally { if (timer) clearTimeout(timer); }
}

/** A timed-out normal quit remains a failed smoke, even after owned-child cleanup. */
export async function closeOwnedPackagedApplication(application: OwnedPackagedApplication, child: OwnedChild, options: {
  normalCloseMs?: number;
  forceExitMs?: number;
  observe?(phase: string): void;
} = {}): Promise<void> {
  if (!Number.isSafeInteger(child.pid) || child.pid! < 1) throw new Error("Packaged smoke has no exact owned child to close.");
  const alive = () => child.exitCode === null && child.signalCode === null;
  let observedExit = !alive();
  let onExit!: () => void;
  const exited = new Promise<void>(resolve => {
    onExit = () => { observedExit = true; resolve(); };
    if (observedExit) resolve(); else child.once("exit", onExit);
  });
  try {
    options.observe?.("normal-close-start");
    await bounded(Promise.all([Promise.resolve().then(() => application.close()), exited]),
      options.normalCloseMs ?? 30_000, "Packaged application's normal shutdown");
    if (child.exitCode !== 0 || child.signalCode !== null) {
      throw new Error(`Packaged application did not exit cleanly (code ${String(child.exitCode)}, signal ${String(child.signalCode)}).`);
    }
    options.observe?.("normal-close-complete");
  } catch (normalFailure) {
    options.observe?.("normal-close-failed");
    // Never search by process name, executable, profile, or PID. Only the live
    // ChildProcess returned by this isolated launch can be terminated here.
    if (!observedExit && alive()) {
      try {
        options.observe?.("owned-child-force-cleanup-start");
        if (!child.kill("SIGKILL")) throw new Error("The exact packaged-smoke child refused cleanup.");
        await bounded(exited, options.forceExitMs ?? 5_000, "Owned packaged-smoke child cleanup");
        options.observe?.("owned-child-force-cleanup-complete");
      } catch (cleanupFailure) {
        throw new AggregateError([normalFailure, cleanupFailure], "Normal shutdown failed and its exact test child did not acknowledge cleanup.");
      }
    }
    throw normalFailure;
  } finally { child.removeListener("exit", onExit); }
}
