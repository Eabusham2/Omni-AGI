import type { AuthorizedSpillStorageSnapshot } from "../shared/deviceResources";

/** Capacity and live availability are deliberately different. A temporary
 * external allocation pauses admission, never rewrites a selected window. */
export function authorizedSpillCapacity(snapshot: AuthorizedSpillStorageSnapshot, input: {
  selectedPoolBytes: number; primaryRequiredBytes: number;
}): { capacityBytes: number; currentlyAvailableBytes: number; largestSingleAllocationBytes: number } {
  const count = (value: number): number => {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Invalid exact spill-capacity accounting.");
    return value;
  };
  if (snapshot.authorized !== true) throw new Error("Drive capacity requires local authority.");
  Object.values(input).forEach(count);
  [snapshot.largestSharedPoolBytes, snapshot.sharedUsedBytes, snapshot.sharedPendingBytes, snapshot.ownerBytes].forEach(count);
  const groups = new Set<string>();
  let capacity = 0, available = 0, largest = 0;
  for (const group of snapshot.groups) {
    if (groups.has(group.physicalGroup)) throw new Error("A physical spill pool was counted twice.");
    groups.add(group.physicalGroup);
    [group.ceilingBytes, group.availableBytes, group.allocatedBytes, group.pendingBytes,
      group.unmaterializedPromiseBytes, group.ownerBytes, group.allocationUnitBytes].forEach(count);
    const primary = group.primaryVolume ? input.primaryRequiredBytes : 0;
    const room = Math.max(0, group.ceilingBytes - primary);
    capacity = count(capacity + room);
    largest = Math.max(largest, room);
    // Existing pages owned by this instance can satisfy its reviewed budget.
    // Other instances' pages and all outstanding writes remain charged.
    const quotaRoom = Math.max(0, group.ceilingBytes - group.allocatedBytes - group.pendingBytes + group.ownerBytes - primary);
    const physicalRoom = Math.max(0, group.availableBytes - group.pendingBytes - group.unmaterializedPromiseBytes + group.ownerBytes - primary);
    available = count(available + Math.min(room, quotaRoom, physicalRoom));
  }
  const pool = Math.max(input.selectedPoolBytes, snapshot.largestSharedPoolBytes);
  const globalCapacity = Math.max(0, pool - input.primaryRequiredBytes);
  const globalAvailable = Math.max(0, pool - snapshot.sharedUsedBytes - snapshot.sharedPendingBytes + snapshot.ownerBytes - input.primaryRequiredBytes);
  return { capacityBytes: Math.min(capacity, globalCapacity),
    currentlyAvailableBytes: Math.min(available, globalAvailable), largestSingleAllocationBytes: Math.min(largest, globalCapacity) };
}
