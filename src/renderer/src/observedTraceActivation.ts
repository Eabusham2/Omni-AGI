/** A mean of observed values, never an inferred fraction of the whole brain. */
export function meanObservedTraceActivation(
  observations: ReadonlyArray<{ activation: number }> | undefined
): number | null {
  if (!observations?.length) return null;
  if (observations.some((item) => !Number.isFinite(item.activation))) return null;
  const mean = observations.reduce((total, item) => total + item.activation, 0) /
    observations.length;
  return Math.max(0, Math.min(1, mean));
}
