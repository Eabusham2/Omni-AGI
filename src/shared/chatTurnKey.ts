/** A copied paused turn keeps its native IDs; host ownership is brain scoped. */
export const chatTurnKey = (brainId: string, turnId: string): string => JSON.stringify([brainId, turnId]);
