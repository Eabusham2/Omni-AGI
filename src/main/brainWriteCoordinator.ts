import type { BrainRepository } from "./brainRepository";

type BrainWriteOperation<T> = () => Promise<T>;

function abortReason(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const error = new Error(
    typeof signal.reason === "string" && signal.reason.trim()
      ? signal.reason
      : "The queued brain operation was cancelled.",
  );
  error.name = "AbortError";
  return error;
}

async function waitForPredecessor(
  predecessor: Promise<void>,
  signal?: AbortSignal,
): Promise<void> {
  const ready = predecessor.catch(() => undefined);
  if (!signal) {
    await ready;
    return;
  }
  if (signal.aborted) throw abortReason(signal);

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const finish = (error?: Error): void => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve();
    };
    const onAbort = (): void => finish(abortReason(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    void ready.then(() => finish());
    if (signal.aborted) onAbort();
  });
}

/**
 * FIFO serialization for whole-document writers of one persistent brain.
 *
 * The lock must cover the complete read -> neural/IO work -> save phase. Merely
 * serializing BrainRepository.save would still let an already-stale detached
 * BrainDocument overwrite a newer commit. Multi-brain operations acquire
 * normalized IDs in one stable order so merge paths cannot deadlock.
 */
export class BrainWriteCoordinator {
  private readonly tails = new Map<string, Promise<void>>();

  async run<T>(
    brainIds: string | readonly string[],
    operation: BrainWriteOperation<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    const ids = [
      ...new Set(
        (typeof brainIds === "string" ? [brainIds] : brainIds).map((id) =>
          id.trim(),
        ),
      ),
    ]
      .filter(Boolean)
      .sort((left, right) => left.localeCompare(right));
    if (ids.length === 0)
      throw new Error("A brain writer requires at least one brain ID.");

    const acquire = (index: number): Promise<T> => {
      const id = ids[index];
      return id === undefined
        ? operation()
        : this.runForOne(id, () => acquire(index + 1), signal);
    };
    return acquire(0);
  }

  private async runForOne<T>(
    brainId: string,
    operation: BrainWriteOperation<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    const predecessor = this.tails.get(brainId) ?? Promise.resolve();
    let release = (): void => undefined;
    const barrier = new Promise<void>((resolve) => {
      release = resolve;
    });
    // A failed caller never poisons the queue: completion is represented only
    // by its explicitly released barrier, chained after the prior writer.
    const tail = predecessor.catch(() => undefined).then(() => barrier);
    this.tails.set(brainId, tail);

    try {
      await waitForPredecessor(predecessor, signal);
      if (signal?.aborted) throw abortReason(signal);
      // Once acquired, cancellation is owned by the operation. Do not release
      // on the abort event itself: it may still be unwinding a worker request or
      // save, and a successor must not observe that late mutation concurrently.
      return await operation();
    } finally {
      release();
      // An aborted waiter may release before its predecessor completes. Keep
      // its tail registered until the ordered chain actually settles, or a new
      // writer could incorrectly bypass the still-running predecessor.
      void tail.then(() => {
        if (this.tails.get(brainId) === tail) this.tails.delete(brainId);
      });
    }
  }
}

const COORDINATORS = new WeakMap<BrainRepository, BrainWriteCoordinator>();

export function brainWriteCoordinator(
  repository: BrainRepository,
): BrainWriteCoordinator {
  const existing = COORDINATORS.get(repository);
  if (existing) return existing;
  const created = new BrainWriteCoordinator();
  COORDINATORS.set(repository, created);
  return created;
}

export function withBrainWrite<T>(
  repository: BrainRepository,
  brainIds: string | readonly string[],
  operation: BrainWriteOperation<T>,
  signal?: AbortSignal,
): Promise<T> {
  return brainWriteCoordinator(repository).run(brainIds, operation, signal);
}
