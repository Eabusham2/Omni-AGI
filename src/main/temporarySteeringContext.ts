import { createHash } from "node:crypto";

/** Private main-to-worker payload. It is not part of the renderer API. */
export interface TemporarySteeringContext {
  format: "omni-temporary-steering-input-v1";
  brainId: string;
  successorTurnId: string;
  attentionEpoch: number;
  inputs: Array<{ turnId: string; content: string; inputSha256: string }>;
}

export function inheritTemporarySteeringContext(
  predecessor: {
    brainId: string; turnId: string; input: string; attentionEpoch?: number;
    neuralSteered?: boolean; earlySteeredYield?: boolean;
    temporarySteeringContext?: TemporarySteeringContext;
  },
  successorTurnId: string,
  successorAttentionEpoch: number | undefined
): TemporarySteeringContext | undefined {
  if (!predecessor.neuralSteered || predecessor.attentionEpoch === undefined ||
      predecessor.attentionEpoch !== successorAttentionEpoch) return undefined;
  const prior = predecessor.temporarySteeringContext;
  if (prior && (prior.brainId !== predecessor.brainId || prior.successorTurnId !== predecessor.turnId ||
      prior.attentionEpoch !== predecessor.attentionEpoch)) throw new Error("Temporary steering context lost its exact owner.");
  const inputs = [...(prior?.inputs ?? [])];
  if (predecessor.earlySteeredYield) {
    const content = predecessor.input.replace(/\0/g, "").trim();
    if (!content) throw new Error("The unfinished user input is unavailable.");
    inputs.push({ turnId: predecessor.turnId, content,
      inputSha256: createHash("sha256").update(content, "utf8").digest("hex") });
  }
  if (!inputs.length) return undefined;
  const unique = new Map<string, TemporarySteeringContext["inputs"][number]>();
  for (const item of inputs) {
    const previous = unique.get(item.turnId);
    if (previous && previous.inputSha256 !== item.inputSha256) throw new Error("An unfinished turn changed its bound input.");
    unique.set(item.turnId, item);
  }
  return { format: "omni-temporary-steering-input-v1", brainId: predecessor.brainId,
    successorTurnId, attentionEpoch: predecessor.attentionEpoch, inputs: [...unique.values()] };
}
