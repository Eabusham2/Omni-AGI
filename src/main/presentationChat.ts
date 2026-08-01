import { randomUUID } from "node:crypto";
import type {
  BrainDocument,
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
  generatedResponse: string
): ChatResult {
  const cleanInput = input.replace(/\0/g, "").trim();
  const response = generatedResponse.replace(/\0/g, "").trim();
  if (!cleanInput) throw new Error("A chat message cannot be empty.");
  if (!response) throw new Error("The neural worker returned an empty response.");

  const now = new Date().toISOString();
  const traceId = randomUUID();
  const humanMessage: ChatMessage = {
    id: randomUUID(),
    role: "human",
    content: cleanInput,
    createdAt: now,
    runtime: "adaptive-core",
    status: "complete"
  };
  const brainMessage: ChatMessage = {
    id: randomUUID(),
    role: "brain",
    content: response,
    createdAt: new Date().toISOString(),
    traceId,
    runtime: "adaptive-core",
    status: "complete"
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
  brain.traces = brain.traces.slice(-2_000);
  return { brain, humanMessage, brainMessage, trace };
}
