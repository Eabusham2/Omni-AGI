/** Cancellation detaches only this waiter, never an earlier native owner. */
export async function waitHostComputePredecessors(pending: readonly Promise<unknown>[], signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted(); if (!pending.length) return;
  const ready = Promise.allSettled(pending).then(() => undefined);
  if (!signal) return ready;
  return new Promise<void>((resolve, reject) => {
    let done = false;
    const finish = (error?: unknown): void => { if (done) return; done = true; signal.removeEventListener("abort", abort); error ? reject(error) : resolve(); };
    const abort = () => finish(signal.reason ?? new Error("Host compute reservation stopped while queued."));
    signal.addEventListener("abort", abort, { once: true }); void ready.then(() => finish()); if (signal.aborted) abort();
  });
}
