import { randomUUID } from "node:crypto";
import type {
  BrainDocument,
  ChatGenerationEnd,
  ChatMessage,
  ChatResult,
  ThoughtTrace
} from "../shared/types";

/**
 * Persist presentation state for a response produced by the authoritative
 * OmniCortex neural worker. Electron does not create or mutate a second idea,
 * concept, synapse, or working-memory substrate here.
 */
export function recordNeuralChat(
  brain: BrainDocument,
  input: string,
  generatedResponse: string,
  generationEnd?: ChatGenerationEnd
): ChatResult {
  const cleanInput = input.replace(/\0/g, "").trim();
  const response = generatedResponse.replace(/\0/g, "").trim();
  if (!cleanInput) throw new Error("A chat message cannot be empty.");
  if (generationEnd === "no-reply" ? generatedResponse !== "" : !response) {
    throw new Error("The neural worker returned an invalid response completion.");
  }

  const now = new Date().toISOString();
  const traceId = randomUUID();
  const humanMessage: ChatMessage = {
    id: randomUUID(),
    role: "human",
    content: cleanInput,
    createdAt: now,
    runtime: "adaptive-core",
    status: "complete",
    ...(generationEnd ? { generationEnd } : {})
  };
  const brainMessage: ChatMessage = {
    id: randomUUID(),
    role: "brain",
    content: response,
    createdAt: new Date().toISOString(),
    traceId,
    runtime: "adaptive-core",
    status: "complete",
    ...(generationEnd ? { generationEnd } : {})
  };
  brain.messages.push(humanMessage, brainMessage);

  const trace: ThoughtTrace = {
    id: traceId,
    createdAt: now,
    input: cleanInput,
    seed: 0,
    runtime: "adaptive-core",
    activatedConcepts: [],
    recalledIdeas: [],
    driveScores: { novelty: 0, coherence: 0, curiosity: 0 },
    branches: 1,
    selectedBranch: 0,
    steps: [],
    note:
      "The authoritative neural worker supplies measured trace data; Electron stores only this presentation record."
  };
  brain.traces.push(trace);
  return { brain, humanMessage, brainMessage, trace, ...(generationEnd ? { generationEnd } : {}) };
}
