export const GIB_BYTES = 1_073_741_824;

/**
 * A crawl has no knowable final byte count. Reserve a rolling first GiB so it
 * never enters the device planner as zero; the shared pool can then grow or
 * pause transactionally as the persisted frontier discovers more material.
 */
export const INITIAL_WEB_CRAWL_RESERVE_BYTES = GIB_BYTES;

export interface BuildResourcePlanningInput {
  kind: string;
  bytes?: number;
}

export function plannedTrainingSourceBytes(
  resources: readonly BuildResourcePlanningInput[]
): number {
  return resources.reduce((total, resource) => {
    if (resource.kind === "web") {
      return total + INITIAL_WEB_CRAWL_RESERVE_BYTES;
    }
    const bytes = resource.bytes ?? 0;
    return total + (Number.isFinite(bytes) && bytes > 0 ? Math.floor(bytes) : 0);
  }, 0);
}
