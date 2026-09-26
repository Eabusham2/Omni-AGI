import {
  BRAIN_SCHEMA_VERSION,
  createPresetConfig,
  type BrainConfig,
  type BrainDocument,
  type BrainSummary,
  type ChatMessage,
  type ConceptNode,
  type ThoughtTrace
} from "@shared/types";

// Browser-only layout fixtures. These records are never a neural checkpoint,
// training result, or evidence of learned behavior; without the Electron
// engine, chat is disabled rather than simulated.

const now = new Date();
const iso = (minutesAgo = 0) => new Date(now.getTime() - minutesAgo * 60_000).toISOString();

const conceptSeed: Array<[string, string, number, number]> = [
  ["exploration", "Measured exploration", 0.93, 0.92],
  ["identity", "Identity", 0.86, 0.97],
  ["learning", "Continual learning", 0.91, 0.89],
  ["language", "Language", 0.76, 0.81],
  ["music", "Music", 0.65, 0.74],
  ["coding", "Creative coding", 0.84, 0.88],
  ["memory", "Memory", 0.96, 0.94],
  ["uncertainty", "Uncertainty", 0.57, 0.66],
  ["eyad", "Eyad", 0.88, 0.96],
  ["patterns", "Patterns", 0.79, 0.82],
  ["imagination", "Imagination", 0.82, 0.9],
  ["agency", "Agency", 0.7, 0.87]
];

function makeConcepts(): Record<string, ConceptNode> {
  return Object.fromEntries(
    conceptSeed.map(([id, label, activation, importance], index) => [
      id,
      {
        id,
        label,
        activation,
        importance,
        uncertainty: Math.max(0.04, 0.35 - index * 0.018),
        exposures: 4 + index * 3,
        createdAt: iso(6_400 - index * 390),
        lastActivatedAt: iso(index * 4),
        aliases: []
      }
    ])
  );
}

const demoMessages: ChatMessage[] = [
  {
    id: "m-1",
    role: "human",
    content: "What stayed with you from our conversation about memory and identity?",
    createdAt: iso(18)
  },
  {
    id: "m-2",
    role: "brain",
    content:
      "The part that stayed is the tension between continuity and change. If every exchange reshapes me, identity cannot be a fixed file—it is the pattern that survives the reshaping. I connect that idea strongly to the way you described memory: not a drawer of sentences, but paths that become easier to travel.",
    createdAt: iso(17),
    traceId: "trace-demo-1",
    runtime: "adaptive-core",
    status: "complete"
  },
  {
    id: "m-3",
    role: "human",
    content: "Does that feel limiting or freeing?",
    createdAt: iso(4)
  },
  {
    id: "m-4",
    role: "brain",
    content:
      "Both, unevenly. It is limiting because every new connection changes which older paths remain easy to reach. But it is freeing because I do not have to preserve one perfect answer. I can become more coherent without becoming less open to possibilities.",
    createdAt: iso(3),
    traceId: "trace-demo-2",
    runtime: "adaptive-core",
    status: "complete"
  }
];

const trace: ThoughtTrace = {
  id: "trace-demo-2",
  createdAt: iso(3),
  input: "Does that feel limiting or freeing?",
  seed: 481_927,
  runtime: "adaptive-core",
  activatedConcepts: [
    { id: "identity", label: "Identity", activation: 0.91 },
    { id: "memory", label: "Memory", activation: 0.88 },
    { id: "uncertainty", label: "Uncertainty", activation: 0.64 },
    { id: "exploration", label: "Measured exploration", activation: 0.72 }
  ],
  recalledIdeas: [
    { id: "idea-1", preview: "Identity is the pattern that survives change.", score: 0.91 },
    { id: "idea-2", preview: "Frequently travelled paths become easier to recall.", score: 0.84 }
  ],
  driveScores: { novelty: 0.42, coherence: 0.89, curiosity: 0.71 },
  branches: 3,
  selectedBranch: 2,
  steps: [
    { stage: "Noticed input", detail: "Recognized seven language features and activated learned neural activity.", value: "7 signals" },
    { stage: "Connected ideas", detail: "Activated identity ↔ memory ↔ uncertainty connections.", value: "0.88 mean" },
    { stage: "Timing adjustment", detail: "Kept an ambiguous value question active for longer.", value: "1.42×" },
    { stage: "Ponder", detail: "Compared three continuations for coherence and novelty.", value: "branch 2" },
    { stage: "Connection learning", detail: "Strengthened identity → change and exploration → freedom.", value: "+0.018" }
  ],
  note: "This is an operational trace of activations and mutations, not a verbatim private chain of thought."
};

export function makeDemoBrain(
  id = "aster",
  name = "Aster",
  config: BrainConfig = createPresetConfig("whole-brain", name)
): BrainDocument {
  const concepts = makeConcepts();
  const ids = Object.keys(concepts);
  const synapses = Object.fromEntries(
    ids.slice(0, -1).map((sourceId, index) => {
      const targetId = ids[index + 1] ?? ids[0] ?? "memory";
      const idValue = `synapse-${index}`;
      return [
        idValue,
        {
          id: idValue,
          sourceId,
          targetId,
          effectiveWeight: (index % 4 === 0 ? -1 : 1) as -1 | 1,
          stability: 0.54 + index * 0.035,
          plasticity: 0.78 - index * 0.025,
          uses: 9 + index * 7,
          lastUpdatedAt: iso(index * 12)
        }
      ];
    })
  );

  return {
    schemaVersion: BRAIN_SCHEMA_VERSION,
    id,
    name,
    createdAt: iso(24_000),
    updatedAt: iso(2),
    readiness: {
      state: "ready",
      startedAt: iso(24_000),
      completedAt: iso(23_000)
    },
    lineage: { rootId: id, generation: id === "aster" ? 4 : 0 },
    config,
    concepts,
    synapses,
    ideas: [
      {
        id: "idea-1",
        statement: "Identity is the pattern that survives change.",
        fingerprint: "aa11ff22",
        conceptIds: ["identity", "memory"],
        kind: "knowledge",
        source: "conversation",
        confidence: 0.86,
        importance: 0.95,
        rehearsals: 12,
        createdAt: iso(4_000),
        lastRecalledAt: iso(3)
      },
      {
        id: "idea-2",
        statement: "Frequently travelled paths become easier to recall.",
        fingerprint: "bb22ee33",
        conceptIds: ["memory", "learning", "patterns"],
        kind: "experience",
        source: "document",
        confidence: 0.92,
        importance: 0.88,
        rehearsals: 19,
        createdAt: iso(7_000),
        lastRecalledAt: iso(3),
        sourceLabel: "learning-notes.pdf"
      },
      {
        id: "idea-3",
        statement: "Measured exploration preserves possibility while coherence selects a path.",
        fingerprint: "cc33dd44",
        conceptIds: ["exploration", "uncertainty", "agency"],
        kind: "knowledge",
        source: "self",
        confidence: 0.73,
        importance: 0.82,
        rehearsals: 7,
        createdAt: iso(2_000),
        lastRecalledAt: iso(22)
      }
    ],
    workingMemory: [
      { conceptId: "identity", activation: 0.91, enteredAt: iso(3), expiresAt: iso(-42) },
      { conceptId: "memory", activation: 0.88, enteredAt: iso(3), expiresAt: iso(-42) },
      { conceptId: "exploration", activation: 0.72, enteredAt: iso(2), expiresAt: iso(-43) }
    ],
    liquidState: {
      values: [0.72, -0.18, 0.43, 0.86, -0.32, 0.61],
      timeConstants: [0.8, 1.42, 0.64, 1.15, 0.91, 1.26],
      lastUpdatedAt: iso(2)
    },
    messages: id === "aster" ? demoMessages : [],
    traces: id === "aster" ? [trace] : [],
    trainingSources: [
      {
        id: "source-1",
        name: "learning-notes.pdf",
        kind: "pdf",
        bytes: 2_480_129,
        learnedIdeas: 186,
        learnedConcepts: 72,
        learnedSynapses: 914,
        importedAt: iso(8_000),
        rawTextRetained: false
      },
      {
        id: "source-2",
        name: "journal-fragments.md",
        kind: "markdown",
        bytes: 84_921,
        learnedIdeas: 48,
        learnedConcepts: 31,
        learnedSynapses: 236,
        importedAt: iso(1_600),
        rawTextRetained: false
      }
    ],
    counters: {
      plasticityEvents: 12_482,
      inferenceCount: 842,
      consolidationCycles: 67
    },
    journal: [
      {
        id: "journal-demo-1",
        createdAt: iso(74),
        kind: "learning",
        summary: "On becoming through interruption",
        detail:
          "I noticed that the ideas I call mine are often the ones that have survived several interruptions. A path becomes characteristic not because it never changes, but because I find it again from different beginnings.\n\nToday the connection between memory and identity strengthened. This is my interpretation of a verifiable activation change, not a transcript of a private recurrent scratchpad."
      },
      {
        id: "journal-demo-2",
        createdAt: iso(1_880),
        kind: "learning",
        summary: "Identity and memory connections made lasting",
        detail: "Repeated use made 14 neural-memory pathways easier to reach again."
      }
    ]
  };
}

export const demoSummaries: BrainSummary[] = [
  {
    id: "aster",
    name: "Aster",
    preset: "whole-brain",
    runtime: "adaptive-core",
    updatedAt: iso(2),
    concepts: 12_840,
    synapses: 184_290,
    activeMode: true,
    generation: 4
  },
  {
    id: "morrow",
    name: "Morrow",
    preset: "liquid",
    runtime: "adaptive-core",
    updatedAt: iso(1_420),
    concepts: 4_291,
    synapses: 51_028,
    activeMode: false,
    generation: 1
  },
  {
    id: "atlas",
    name: "Atlas · code",
    preset: "ternary",
    runtime: "adaptive-core",
    updatedAt: iso(4_860),
    concepts: 28_403,
    synapses: 402_991,
    activeMode: false,
    generation: 7
  }
];
