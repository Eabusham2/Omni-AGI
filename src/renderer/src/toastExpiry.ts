export interface ToastExpiry { pause(): void; resume(): void; dispose(): void }

/** UI-only reading time; never a neural/work deadline. */
export function toastReadingTime(text: string): number { return Math.max(4000, Math.min(18000, 3000 + text.length * 45)); }

/** One owner, one monotonic remaining duration. Disposal retires stale callbacks. */
export function createToastExpiry(durationMs: number, expire: () => void, now: () => number, schedule: (work: () => void, delay: number) => number, cancel: (handle: number) => void): ToastExpiry {
  if (!Number.isFinite(durationMs) || durationMs < 0) throw new Error("Invalid notification reading duration.");
  let remaining = durationMs, started: number | undefined, handle: number | undefined, retired = false, generation = 0;
  const pause = () => {
    if (retired) return;
    generation++;
    if (started !== undefined) remaining = Math.max(0, remaining - Math.max(0, now() - started));
    started = undefined; if (handle !== undefined) cancel(handle); handle = undefined;
  };
  const resume = () => {
    if (retired || handle !== undefined) return;
    const attempt = ++generation;
    started = now(); handle = schedule(() => { if (retired || generation !== attempt) return; handle = undefined; started = undefined; retired = true; expire(); }, remaining);
  };
  return { pause, resume, dispose() { pause(); retired = true; } };
}
