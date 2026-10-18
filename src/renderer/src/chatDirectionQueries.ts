import type { SteerDirectionRecord, SteerDirectionsApi } from "../../shared/steerDirections";

/** The100-turn IPC boundary is a batch, never a limit on retained history. */
export async function visitChatDirectionTurns(
  api: Pick<SteerDirectionsApi, "forTurns">,
  brainId: string,
  turnIds: string[],
  accept: (records: SteerDirectionRecord[]) => void,
  current: () => boolean
): Promise<void> {
  const ids = [...new Set(turnIds)];
  for (let offset = 0; offset < ids.length; offset += 100) {
    if (!current()) return;
    const records = await api.forTurns(brainId, ids.slice(offset, offset + 100));
    if (!current()) return;
    accept(records);
  }
}
