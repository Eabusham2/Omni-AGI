import { createHash, randomUUID } from "node:crypto";
import type {
  BrainDocument,
  ChatMessage,
  ChatResult,
  Idea,
  RecallResult,
  ThoughtTrace
} from "../shared/types";
import {
  classifyIdea,
  extractConcepts,
  isGreeting,
  isQuestion,
  normalizeConcept,
  preview,
  splitIntoIdeas
} from "./core/language";
import {
  decodeFingerprint,
  encodeFingerprint,
  seededRandom,
  similarity,
  textSeed
} from "./core/vectorSymbolic";

interface LearningDelta {
  ideas: number;
  concepts: number;
  synapses: number;
  conceptIds: string[];
}

/**
 * Legacy deterministic mechanics retained only for isolated unit fixtures.
 * Production neural creation, learning, recall, and chat are owned by the
 * authoritative Python OmniCortex worker; only recordNeuralChat below is used
 * to persist its presentation state.
 */
const FALLBACK_DYNAMICS = Object.freeze({
  membraneLeak: 0.82,
  firingThreshold: 0.56,
  stdpWindow: 8,
  consolidationRate: 0.06,
  forgettingRate: 0.002
});

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function hashId(prefix: string, value: string): string {
  return `${prefix}_${createHash("sha256").update(value).digest("hex").slice(0, 24)}`;
}

function conceptId(key: string): string {
  return hashId("c", normalizeConcept(key));
}

function synapseId(sourceId: string, targetId: string): string {
  return hashId("s", `${sourceId}>${targetId}`);
}

function effectiveWeight(latent: number): -1 | 0 | 1 {
  if (latent >= 0.2) return 1;
  if (latent <= -0.2) return -1;
  return 0;
}

function activeIdeaText(idea: Idea): string {
  return idea.statement ?? idea.sourceLabel ?? "an encoded idea";
}

function purgeWorkingMemory(brain: BrainDocument, now: string): void {
  const timestamp = Date.parse(now);
  brain.workingMemory = brain.workingMemory
    .filter((item) => Date.parse(item.expiresAt) > timestamp)
    .sort((left, right) => right.activation - left.activation)
    .slice(0, brain.config.workingMemorySlots);
}

function updateLiquidState(brain: BrainDocument, stimulus: number, now: string): void {
  const values =
    brain.liquidState.values.length > 0
      ? brain.liquidState.values
      : Array.from({ length: 16 }, () => 0);
  const constants =
    brain.liquidState.timeConstants.length === values.length
      ? brain.liquidState.timeConstants
      : values.map((_, index) => 0.25 + index * 0.05);
  brain.liquidState.values = values.map((value, index) => {
    const tau = Math.max(0.05, constants[index] ?? 0.5);
    const gate = 1 - Math.exp(-1 / tau);
    const recurrent = Math.tanh(value * 0.72 + stimulus * (0.2 + (index % 5) * 0.06));
    return clamp(value + gate * (recurrent - value), -1, 1);
  });
  brain.liquidState.timeConstants = constants;
  brain.liquidState.lastUpdatedAt = now;
}

function applySynapticUpdate(
  brain: BrainDocument,
  sourceId: string,
  targetId: string,
  timing: number,
  salience: number,
  now: string
): boolean {
  if (sourceId === targetId) return false;
  const id = synapseId(sourceId, targetId);
  const current = brain.synapses[id];
  const window = FALLBACK_DYNAMICS.stdpWindow;
  const causal = timing >= 0 ? 1 : -0.55;
  const stdp = causal * Math.exp(-Math.abs(timing) / window);
  const stability = current?.stability ?? 0.05;
  const plasticity = current?.plasticity ?? 1;
  const learningRate =
    brain.config.learningRate *
    salience *
    stdp *
    plasticity *
    (1 - stability * 0.7);
  const latentWeight = clamp((current?.latentWeight ?? 0) + learningRate, -1, 1);
  brain.synapses[id] = {
    id,
    sourceId,
    targetId,
    effectiveWeight: effectiveWeight(latentWeight),
    latentWeight,
    stability: clamp(stability + 0.004 * Math.abs(stdp)),
    plasticity: clamp(plasticity * 0.9995, 0.05, 1),
    uses: (current?.uses ?? 0) + 1,
    lastUpdatedAt: now
  };
  brain.counters.plasticityEvents += 1;
  return current === undefined;
}

function organicNoise(brain: BrainDocument): number {
  const active = Object.values(brain.concepts).filter((concept) => concept.activation > 0.05);
  const uncertainty =
    active.length === 0
      ? 0.5
      : active.reduce((sum, concept) => sum + concept.uncertainty, 0) / active.length;
  const pressure = clamp(brain.workingMemory.length / Math.max(1, brain.config.workingMemorySlots));
  const recurrentActivity =
    brain.liquidState.values.length === 0
      ? 0
      : brain.liquidState.values.reduce((sum, value) => sum + Math.abs(value), 0) /
        brain.liquidState.values.length;
  return clamp(
    0.01 + uncertainty * 0.045 + (1 - pressure) * 0.015 + recurrentActivity * 0.01,
    0.01,
    0.08
  );
}

export function learnText(
  brain: BrainDocument,
  text: string,
  source: Idea["source"],
  sourceLabel?: string
): LearningDelta {
  const cleanText = text.replace(/\0/g, "").trim();
  if (!cleanText) return { ideas: 0, concepts: 0, synapses: 0, conceptIds: [] };
  const now = new Date().toISOString();
  const extracted = extractConcepts(cleanText, 128);
  const random = seededRandom(textSeed(cleanText, brain.counters.plasticityEvents));
  const variability = organicNoise(brain);
  let newConcepts = 0;
  let newSynapses = 0;
  const activatedIds: string[] = [];

  for (const extractedConcept of extracted) {
    const id = conceptId(extractedConcept.key);
    const existing = brain.concepts[id];
    const noisyInput = extractedConcept.salience + (random() - 0.5) * variability;
    const membrane =
      (existing?.activation ?? 0) * FALLBACK_DYNAMICS.membraneLeak + noisyInput;
    const fired = membrane >= FALLBACK_DYNAMICS.firingThreshold;
    const activation = clamp(fired ? membrane : membrane * 0.55);
    if (!existing) {
      newConcepts += 1;
      brain.concepts[id] = {
        id,
        label: extractedConcept.label,
        activation,
        importance: clamp(extractedConcept.salience * 0.6),
        uncertainty: 0.72,
        exposures: 1,
        createdAt: now,
        lastActivatedAt: now,
        aliases: []
      };
    } else if (existing) {
      if (
        normalizeConcept(existing.label) !== normalizeConcept(extractedConcept.label) &&
        !existing.aliases.includes(extractedConcept.label)
      ) {
        existing.aliases.push(extractedConcept.label);
      }
      existing.activation = activation;
      existing.importance = clamp(
        existing.importance * 0.92 + extractedConcept.salience * 0.08
      );
      existing.uncertainty = clamp(existing.uncertainty * 0.965);
      existing.exposures += 1;
      existing.lastActivatedAt = now;
    }
    if (brain.concepts[id] && fired) activatedIds.push(id);
  }

  const positional = extracted
    .map((item) => ({ ...item, id: conceptId(item.key) }))
    .filter((item) => brain.concepts[item.id] !== undefined);
  for (let leftIndex = 0; leftIndex < positional.length; leftIndex += 1) {
    const left = positional[leftIndex];
    if (!left) continue;
    const end = Math.min(
      positional.length,
      leftIndex + Math.max(2, FALLBACK_DYNAMICS.stdpWindow)
    );
    for (let rightIndex = leftIndex + 1; rightIndex < end; rightIndex += 1) {
      const right = positional[rightIndex];
      if (!right) continue;
      const distance = right.position - left.position;
      if (
        applySynapticUpdate(
          brain,
          left.id,
          right.id,
          distance,
          (left.salience + right.salience) / 2,
          now
        )
      ) {
        newSynapses += 1;
      }
      if (
        applySynapticUpdate(
          brain,
          right.id,
          left.id,
          -distance,
          (left.salience + right.salience) / 2,
          now
        )
      ) {
        newSynapses += 1;
      }
    }
  }

  let newIdeas = 0;
  for (const statement of splitIntoIdeas(cleanText)) {
    const labels = extractConcepts(statement, 32).map((item) => item.key);
    if (labels.length === 0) continue;
    const fingerprint = encodeFingerprint(labels);
    const existing = brain.ideas.find((idea) => idea.fingerprint === fingerprint);
    if (existing) {
      existing.rehearsals += 1;
      existing.confidence = clamp(existing.confidence + 0.025);
      existing.importance = clamp(existing.importance + 0.012);
      continue;
    }
    const recipe = brain.config.memoryRecipe;
    const retainStatement =
      recipe === "total-recall" ||
      (recipe === "human-consolidation" && brain.config.retainSourceText);
    brain.ideas.push({
      id: randomUUID(),
      statement: retainStatement ? statement : undefined,
      fingerprint,
      conceptIds: labels.map(conceptId).filter((id) => brain.concepts[id] !== undefined),
      kind: classifyIdea(statement),
      source,
      confidence: 0.56,
      importance: 0.5,
      rehearsals: 1,
      createdAt: now,
      sourceLabel
    });
    newIdeas += 1;
  }

  const halfLifeMinutes =
    30 + Math.log2(Math.max(2, brain.config.workingMemorySlots)) * 5;
  const expiry = new Date(
    Date.parse(now) + halfLifeMinutes * 60_000 * 2
  ).toISOString();
  for (const id of activatedIds) {
    const concept = brain.concepts[id];
    if (!concept) continue;
    const existing = brain.workingMemory.find((item) => item.conceptId === id);
    if (existing) {
      existing.activation = Math.max(existing.activation, concept.activation);
      existing.expiresAt = expiry;
    } else {
      brain.workingMemory.push({
        conceptId: id,
        activation: concept.activation,
        enteredAt: now,
        expiresAt: expiry
      });
    }
  }
  purgeWorkingMemory(brain, now);
  updateLiquidState(brain, activatedIds.length / Math.max(1, extracted.length), now);
  return {
    ideas: newIdeas,
    concepts: newConcepts,
    synapses: newSynapses,
    conceptIds: activatedIds
  };
}

export function recallIdeas(brain: BrainDocument, input: string, limit = 5): RecallResult[] {
  const queryLabels = extractConcepts(input, 48).map((item) => item.key);
  const queryIds = new Set(queryLabels.map(conceptId));
  const queryFingerprint = decodeFingerprint(encodeFingerprint(queryLabels));
  return brain.ideas
    .map((idea): RecallResult => {
      const overlap =
        idea.conceptIds.length === 0
          ? 0
          : idea.conceptIds.filter((id) => queryIds.has(id)).length /
            Math.sqrt(Math.max(1, idea.conceptIds.length * queryIds.size));
      const vsaSimilarity = similarity(queryFingerprint, decodeFingerprint(idea.fingerprint));
      const recency = idea.lastRecalledAt
        ? Math.exp(-(Date.now() - Date.parse(idea.lastRecalledAt)) / (14 * 86_400_000))
        : 0.1;
      const score =
        overlap * 0.48 +
        vsaSimilarity * 0.25 +
        idea.confidence * 0.12 +
        idea.importance * 0.1 +
        recency * 0.05;
      return { idea, score, overlap, vsaSimilarity };
    })
    .filter((result) => result.overlap > 0 || result.score > 0.45)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}

function generateFallbackReply(
  brain: BrainDocument,
  input: string,
  recalled: RecallResult[],
  random: () => number
): string {
  if (isGreeting(input)) {
    const variants = ["Hey.", "Hello.", "Hey—what are you thinking about?"];
    return variants[Math.floor(random() * variants.length)] ?? variants[0] ?? "Hello.";
  }
  const remembered = recalled
    .map((result) => activeIdeaText(result.idea))
    .filter((statement) => !input.includes(statement))
    .slice(0, 3);
  if (remembered.length === 0) {
    return isQuestion(input)
      ? "I do not have enough learned structure to answer that yet. Teach me, or add material in Data Studio."
      : "I registered that, but this blank mind has not learned enough language structure to form a richer reply yet.";
  }
  const lead = isQuestion(input)
    ? "The strongest learned connection I can recover is"
    : "That activates something I learned";
  const joining = remembered.map((statement) => `“${preview(statement, "encoded idea", 180)}”`).join("; ");
  const uncertainty =
    recalled[0] && recalled[0].score < 0.62 ? " I may be connecting it loosely." : "";
  return `${lead}: ${joining}.${uncertainty}`;
}

export function runFallbackChat(
  brain: BrainDocument,
  input: string,
  generatedResponse?: string
): ChatResult {
  const cleanInput = input.replace(/\0/g, "").trim().slice(0, 100_000);
  if (!cleanInput) throw new Error("A chat message cannot be empty.");
  const now = new Date().toISOString();
  const turn = brain.counters.inferenceCount;
  const seed = textSeed(cleanInput, turn);
  const random = seededRandom(seed);
  const recalledBeforeLearning = recallIdeas(brain, cleanInput, 6);
  const humanMessage: ChatMessage = {
    id: randomUUID(),
    role: "human",
    content: cleanInput,
    createdAt: now,
    runtime: "adaptive-core",
    status: "complete"
  };
  brain.messages.push(humanMessage);
  const learned = brain.config.onlineLearning
    ? learnText(brain, cleanInput, "conversation", "continuous chat")
    : { ideas: 0, concepts: 0, synapses: 0, conceptIds: [] };
  const availableWorkspace = Math.max(
    1,
    brain.config.workingMemorySlots - brain.workingMemory.length
  );
  const branchDemand =
    1 +
    Math.ceil(
      Math.log2(
        1 + learned.concepts + learned.synapses * 0.1 + recalledBeforeLearning.length
      )
    );
  const branchCount = Math.max(
    1,
    Math.min(Math.ceil(Math.sqrt(availableWorkspace)), branchDemand)
  );
  const selectedBranch = Math.floor(random() * branchCount);
  const response =
    generatedResponse?.replace(/\0/g, "").trim().slice(0, 200_000) ||
    generateFallbackReply(brain, cleanInput, recalledBeforeLearning, random);
  const traceId = randomUUID();
  const brainMessage: ChatMessage = {
    id: randomUUID(),
    role: "brain",
    content: response,
    createdAt: new Date().toISOString(),
    traceId,
    runtime: "adaptive-core",
    status: "complete"
  };
  brain.messages.push(brainMessage);
  if (brain.config.onlineLearning) {
    learnText(brain, response, "self", "self-generated language");
  }
  for (const recalled of recalledBeforeLearning) {
    recalled.idea.lastRecalledAt = now;
    recalled.idea.rehearsals += 1;
  }
  brain.counters.inferenceCount += 1;
  const activatedConcepts = learned.conceptIds
    .map((id) => brain.concepts[id])
    .filter((concept): concept is NonNullable<typeof concept> => concept !== undefined)
    .sort((left, right) => right.activation - left.activation)
    .slice(0, 12)
    .map((concept) => ({
      id: concept.id,
      label: concept.label,
      activation: concept.activation
    }));
  const novelty = clamp(
    learned.concepts / Math.max(1, learned.concepts + extractedCount(cleanInput))
  );
  const coherence = recalledBeforeLearning[0]?.score ?? 0;
  const uncertainty =
    activatedConcepts.length === 0
      ? 1
      : activatedConcepts.reduce(
          (sum, activated) => sum + (brain.concepts[activated.id]?.uncertainty ?? 1),
          0
        ) / activatedConcepts.length;
  const recurrentTension = Math.abs(brain.liquidState.values[0] ?? 0);
  const curiosity = clamp(
    novelty * 0.42 +
      uncertainty * 0.28 +
      (1 - coherence) * 0.2 +
      recurrentTension * 0.1
  );
  const trace: ThoughtTrace = {
    id: traceId,
    createdAt: now,
    input: cleanInput,
    seed,
    runtime: "adaptive-core",
    activatedConcepts,
    recalledIdeas: recalledBeforeLearning.map((result) => ({
      id: result.idea.id,
      preview: preview(result.idea.statement, result.idea.sourceLabel ?? "parameterized idea"),
      score: result.score
    })),
    driveScores: { novelty, coherence, curiosity },
    branches: branchCount,
    selectedBranch,
    steps: [
      {
        stage: "sensory-boundary",
        detail: `Extracted and fired ${activatedConcepts.length} concept assemblies.`
      },
      {
        stage: "plasticity",
        detail: `Created ${learned.synapses} synapses and updated local STDP traces.`
      },
      {
        stage: "associative-recall",
        detail: `Recovered ${recalledBeforeLearning.length} parameterized ideas.`
      },
      {
        stage: "liquid-routing",
        detail: `Explored ${branchCount} recurrent branch${branchCount === 1 ? "" : "es"}.`,
        value: String(selectedBranch)
      }
    ],
    note:
      "This trace reports operational activations and mutations. Generated prose is a self-report, not a guaranteed private chain of thought."
  };
  brain.traces.push(trace);
  brain.traces = brain.traces.slice(-2_000);
  return { brain, humanMessage, brainMessage, trace };
}

/**
 * Record presentation state for a response produced by the authoritative
 * Python neural substrate. This deliberately does not create a second concept
 * graph, idea store, synapse set, or working-memory state in Electron.
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

function extractedCount(text: string): number {
  return extractConcepts(text, 128).length;
}

export function consolidateBrain(brain: BrainDocument): BrainDocument {
  const now = new Date().toISOString();
  const decay = FALLBACK_DYNAMICS.forgettingRate;
  for (const concept of Object.values(brain.concepts)) {
    const days = Math.max(0, (Date.now() - Date.parse(concept.lastActivatedAt)) / 86_400_000);
    concept.activation = clamp(concept.activation * Math.exp(-decay * Math.max(1, days)));
    concept.importance = clamp(
      concept.importance +
        FALLBACK_DYNAMICS.consolidationRate * Math.log1p(concept.exposures) * 0.01
    );
  }
  for (const synapse of Object.values(brain.synapses)) {
    const rehearsal = Math.log1p(synapse.uses) / 12;
    synapse.stability = clamp(
      synapse.stability + FALLBACK_DYNAMICS.consolidationRate * rehearsal
    );
    synapse.plasticity = clamp(1 - synapse.stability * 0.7, 0.05, 1);
    synapse.latentWeight *= 1 - decay * (1 - synapse.stability);
    synapse.effectiveWeight = effectiveWeight(synapse.latentWeight);
  }
  for (const idea of brain.ideas) {
    idea.confidence = clamp(
      idea.confidence +
        FALLBACK_DYNAMICS.consolidationRate * Math.log1p(idea.rehearsals) * 0.02
    );
  }
  purgeWorkingMemory(brain, now);
  brain.counters.consolidationCycles += 1;
  brain.journal = [
    ...(brain.journal ?? []),
    {
      id: randomUUID(),
      createdAt: now,
      kind: "consolidation",
      summary: `Consolidated ${brain.ideas.length} ideas and ${Object.keys(brain.synapses).length} synapses.`
    }
  ];
  return brain;
}
