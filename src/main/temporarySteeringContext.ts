import { createHash } from "node:crypto";
import type { NativeAudioInput } from "../shared/nativeAudioChat";
import { verifiedNativeAudioInput } from "./nativeAudioIdentity";

/** Private main-to-worker payload. It is not part of the renderer API. */
export interface TemporarySteeringContext {
  format: "omni-temporary-steering-input-v1";
  brainId: string;
  successorTurnId: string;
  attentionEpoch: number;
  inputs: Array<{ turnId: string; content: string; inputSha256: string }>;
  audioInputs?: NativeAudioInput[];
}

export function inheritTemporarySteeringContext(
  predecessor: {
    brainId: string; turnId: string; input: string; attentionEpoch?: number;
    neuralSteered?: boolean; earlySteeredYield?: boolean;
    temporarySteeringContext?: TemporarySteeringContext;
    nativeAudioInput?: NativeAudioInput;
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
  const audioInputs = [...(prior?.audioInputs ?? [])];
  if (predecessor.earlySteeredYield) {
    if (predecessor.nativeAudioInput) {
      const audio = verifiedNativeAudioInput(predecessor.nativeAudioInput);
      if (predecessor.input !== "" || audio.attachment.brainId !== predecessor.brainId || audio.attachment.turnId !== predecessor.turnId) throw new Error("The unfinished audio input lost its actual owner.");
      audioInputs.push(audio);
    } else {
      const content = predecessor.input.replace(/\0/g, "").trim();
      if (!content) throw new Error("The unfinished user input is unavailable.");
      inputs.push({ turnId: predecessor.turnId, content,
        inputSha256: createHash("sha256").update(content, "utf8").digest("hex") });
    }
  }
  if (!inputs.length && !audioInputs.length) return undefined;
  const unique = new Map<string, TemporarySteeringContext["inputs"][number]>();
  for (const item of inputs) {
    const previous = unique.get(item.turnId);
    if (previous && previous.inputSha256 !== item.inputSha256) throw new Error("An unfinished turn changed its bound input.");
    unique.set(item.turnId, item);
  }
  const uniqueAudio = new Map<string, NativeAudioInput>();
  for (const audio of audioInputs) {
    const input = verifiedNativeAudioInput(audio), previous = uniqueAudio.get(input.attachment.turnId);
    if (previous && previous.inputSha256 !== input.inputSha256) throw new Error("An unfinished audio turn changed its actual raw identity.");
    uniqueAudio.set(input.attachment.turnId, input);
  }
  return { format: "omni-temporary-steering-input-v1", brainId: predecessor.brainId,
    successorTurnId, attentionEpoch: predecessor.attentionEpoch, inputs: [...unique.values()],
    ...(uniqueAudio.size ? { audioInputs: [...uniqueAudio.values()] } : {}) };
}
