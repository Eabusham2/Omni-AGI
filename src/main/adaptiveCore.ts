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
  if (!brain.config.liquidDynamics) return;
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
  if (sourceId === targetId || Object.keys(brain.synapses).length >= brain.config.maxSynapses) {
    return false;
  }
  const id = synapseId(sourceId, targetId);
  const current = brain.synapses[id];
  const window = Math.max(1, brain.config.stdpWindow);
  const causal = timing >= 0 ? 1 : -0.55;
  const stdp =
    brain.config.stdpPlasticity && brain.config.spikingDynamics
      ? causal * Math.exp(-Math.abs(timing) / window)
      : 0.35;
  const stability = current?.stability ?? 0.05;
  const plasticity = current?.plasticity ?? 1;
  const learningRate =
    brain.config.learningRate *
    salience *
    stdp *
    plasticity *
    (brain.config.metaplasticity ? 1 - stability * 0.7 : 1);
  const latentWeight = clamp((current?.latentWeight ?? 0) + learningRate, -1, 1);
  brain.synapses[id] = {
    id,
    sourceId,
    targetId,
    effectiveWeight: brain.config.ternaryWeights
      ? effectiveWeight(latentWeight)
      : latentWeight > 0
        ? 1
        : latentWeight < 0
          ? -1
          : 0,
    latentWeight,
    stability: clamp(stability + 0.004 * Math.abs(stdp)),
    plasticity: clamp(plasticity * 0.9995, 0.05, 1),
    uses: (current?.uses ?? 0) + 1,
    lastUpdatedAt: now
  };
  brain.counters.plasticityEvents += 1;
  return current === undefined;
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
  let newConcepts = 0;
  let newSynapses = 0;
  const activatedIds: string[] = [];

  for (const extractedConcept of extracted) {
    const id = conceptId(extractedConcept.key);
    const existing = brain.concepts[id];
    const noisyInput = extractedConcept.salience + (random() - 0.5) * brain.config.noise;
    const membrane = (existing?.activation ?? 0) * brain.config.membraneLeak + noisyInput;
    const fired = !brain.config.spikingDynamics || membrane >= brain.config.firingThreshold;
    const activation = clamp(fired ? membrane : membrane * 0.55);
    if (!existing && Object.keys(brain.concepts).length < brain.config.maxConcepts) {
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
    const end = Math.min(positional.length, leftIndex + Math.max(2, brain.config.stdpWindow));
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
  if (brain.config.storeAtomicIdeas) {
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
      const recipe = brain.config.memoryRecipe ?? "human-consolidation";
      const retainStatement =
        recipe === "total-recall" || (recipe === "human-consolidation" && brain.config.retainSourceText);
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
  }

  const expiry = new Date(
    Date.parse(now) + Math.max(1, brain.config.shortTermHalfLifeMinutes) * 60_000 * 2
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
      const vsaSimilarity = brain.config.vectorSymbolicMemory
        ? similarity(queryFingerprint, decodeFingerprint(idea.fingerprint))
        : 0;
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
  const branchCount = Math.max(1, brain.config.parallelThoughts);
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
  if (brain.config.onlineLearning && brain.config.learnFromOwnMessages) {
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
  const curiosity = clamp(
    brain.config.curiosityDrive * (0.4 + novelty * 0.6) +
      (brain.liquidState.values[0] ?? 0) * 0.1
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

function extractedCount(text: string): number {
  return extractConcepts(text, 128).length;
}

export function consolidateBrain(brain: BrainDocument): BrainDocument {
  const now = new Date().toISOString();
  const decay = brain.config.forgettingRate;
  for (const concept of Object.values(brain.concepts)) {
    const days = Math.max(0, (Date.now() - Date.parse(concept.lastActivatedAt)) / 86_400_000);
    concept.activation = clamp(concept.activation * Math.exp(-decay * Math.max(1, days)));
    concept.importance = clamp(
      concept.importance + brain.config.consolidationRate * Math.log1p(concept.exposures) * 0.01
    );
  }
  for (const synapse of Object.values(brain.synapses)) {
    const rehearsal = Math.log1p(synapse.uses) / 12;
    synapse.stability = clamp(
      synapse.stability + brain.config.consolidationRate * rehearsal
    );
    synapse.plasticity = clamp(1 - synapse.stability * 0.7, 0.05, 1);
    synapse.latentWeight *= 1 - decay * (1 - synapse.stability);
    synapse.effectiveWeight = brain.config.ternaryWeights
      ? effectiveWeight(synapse.latentWeight)
      : synapse.latentWeight > 0
        ? 1
        : synapse.latentWeight < 0
          ? -1
          : 0;
  }
  for (const idea of brain.ideas) {
    idea.confidence = clamp(
      idea.confidence + brain.config.consolidationRate * Math.log1p(idea.rehearsals) * 0.02
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
