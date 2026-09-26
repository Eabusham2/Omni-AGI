import type { PersistedSubstrateOverview } from "../../shared/types";

export interface LibrarySubstrateHydrationOptions {
  brainIds: string[];
  load: (brainId: string) => Promise<PersistedSubstrateOverview | null>;
  onResolved: (overview: PersistedSubstrateOverview) => void;
  cancelled: () => boolean;
  wait?: (milliseconds: number) => Promise<void>;
  /** Test-only bound. Production retries slowly while Library remains open. */
  maximumAttempts?: number;
}

const RETRY_DELAYS_MS = [200, 600, 1_500, 5_000, 15_000] as const;

/**
 * Hydrate missing card totals independently from the initial Library list.
 * An engine may atomically swap its pointer while list() is reading it; one
 * transient validation miss must not leave the card on a zero mirror forever.
 */
export async function hydrateLibrarySubstrateTotals(
  options: LibrarySubstrateHydrationOptions
): Promise<void> {
  const pending = new Set(options.brainIds.filter(Boolean));
  const wait = options.wait ?? ((milliseconds: number) =>
    new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds)));
  let attempt = 0;
  while (
    pending.size > 0 &&
    !options.cancelled() &&
    (options.maximumAttempts === undefined || attempt < options.maximumAttempts)
  ) {
    const ids = [...pending];
    const results = await Promise.all(ids.map(async (brainId) => {
      try {
        return await options.load(brainId);
      } catch {
        return null;
      }
    }));
    if (options.cancelled()) return;
    results.forEach((overview, index) => {
      const brainId = ids[index];
      if (!brainId || !overview || overview.brainId !== brainId) return;
      pending.delete(brainId);
      options.onResolved(overview);
    });
    attempt += 1;
    if (
      pending.size === 0 ||
      options.cancelled() ||
      (options.maximumAttempts !== undefined && attempt >= options.maximumAttempts)
    ) return;
    await wait(
      RETRY_DELAYS_MS[Math.min(attempt - 1, RETRY_DELAYS_MS.length - 1)] ??
        15_000
    );
  }
}
