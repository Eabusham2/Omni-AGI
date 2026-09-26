import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  presentJournalKind,
  presentMemoryOperation,
  presentTraceEvidence,
  presentTraceSignalMeasures
} from "../src/renderer/src/memoryPresentation";

const rendererApp = readFileSync(
  resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
  "utf8"
);

describe("memory presentation", () => {
  it("translates internal memory terms without changing stored events", () => {
    expect(presentMemoryOperation("Plasticity")).toBe("connection learning");
    expect(presentMemoryOperation("Forming temporal latents")).toBe(
      "Forming temporal neural memory"
    );
    expect(presentMemoryOperation("Latent replay and consolidation")).toBe(
      "memory rehearsal and adaptive retention"
    );
    expect(presentMemoryOperation("Human Consolidation")).toBe("adaptive retention");
    expect(presentMemoryOperation("Background consolidation")).toBe("adaptive retention");
    expect(presentMemoryOperation("Grammar-selected fact lookup")).toBe(
      "fact lookup"
    );
    expect(presentMemoryOperation("Raw key/value lookup")).toBe(
      "key/value lookup"
    );
    expect(presentMemoryOperation("Fact lookup")).toBe("Fact lookup");
    expect(presentJournalKind("consolidation")).toBe("adaptive retention");
    expect(presentJournalKind("human-consolidation")).toBe("adaptive retention");
    expect(presentJournalKind("background-consolidation")).toBe("adaptive retention");
    expect(presentJournalKind("tool-use")).toBe("tool use");
  });

  it("derives unordered trace mechanisms and measurements without positional stages", () => {
    const trace = {
      id: "trace-order-independent",
      activatedConcepts: [
        { id: "identity", label: "Identity", activation: 0.91 }
      ],
      recalledIdeas: [
        { id: "idea-one", preview: "Identity persists through change.", score: 0.84 }
      ],
      branches: 3,
      selectedBranch: 2,
      steps: [
        {
          stage: "Ponder",
          detail: "Compared three candidate continuations.",
          value: "branch 2"
        },
        {
          stage: "Connection learning",
          detail: "Strengthened a measured pathway.",
          value: "+0.018"
        },
        {
          stage: "Activated ideas",
          detail: "Spread activity across learned associations.",
          value: "0.91 mean"
        },
        {
          stage: "Background consolidation",
          detail: "Replayed retained neural activity.",
          value: "1 update"
        }
      ]
    };
    const presented = presentTraceEvidence(trace);

    expect(presented.mechanisms).toMatchObject([
      { kind: "branching", title: "Ponder", measure: "branch 2" },
      { kind: "adaptation", title: "Connection learning", measure: "+0.018" },
      { kind: "activation", title: "Activated ideas", measure: "0.91 mean" },
      { kind: "recall", title: "adaptive retention", measure: "1 update" }
    ]);
    expect(presented.activatedConcepts).toEqual(trace.activatedConcepts);
    expect(presented.recalledIdeas).toEqual(trace.recalledIdeas);
    expect(presented.candidates).toEqual([
      {
        id: "trace-order-independent:candidate:1",
        label: "Candidate 1",
        outcome: "compared"
      },
      {
        id: "trace-order-independent:candidate:2",
        label: "Candidate 2",
        outcome: "selected"
      },
      {
        id: "trace-order-independent:candidate:3",
        label: "Candidate 3",
        outcome: "compared"
      }
    ]);
    expect(presented.mechanisms.every((mechanism) => !("ordinal" in mechanism)))
      .toBe(true);

    const reordered = presentTraceEvidence({
      ...trace,
      steps: [...trace.steps].reverse()
    });
    expect(reordered.activatedConcepts).toEqual(presented.activatedConcepts);
    expect(reordered.recalledIdeas).toEqual(presented.recalledIdeas);
    expect(reordered.candidates).toEqual(presented.candidates);
    expect(Object.fromEntries(
      reordered.mechanisms.map((mechanism) => [mechanism.title, mechanism.kind])
    )).toEqual(Object.fromEntries(
      presented.mechanisms.map((mechanism) => [mechanism.title, mechanism.kind])
    ));
  });

  it("keeps signal/vector activity separate from captured recall preview counts", () => {
    const trace = {
      steps: [
        {
          stage: "neural-associative-sequence recall",
          detail: "Spread signed activity through the recalled pathway.",
          value: "25 active, 33 inhibited signals"
        },
        {
          stage: "vector activation",
          detail: "Measured current recurrent activity.",
          value: "1 active vectors"
        },
        {
          stage: "timing",
          detail: "Recorded elapsed time.",
          value: "4 ms"
        }
      ]
    };

    expect(presentTraceSignalMeasures(trace)).toEqual([
      "25 active, 33 inhibited signals",
      "1 active vectors"
    ]);
    expect(rendererApp).not.toContain("Ideas activated");
    expect(rendererApp).toContain("Recalled idea previews");
    expect(rendererApp).toContain("Neural signal measures");
  });

  it("keeps fixed memory stages and unexplained jargon out of ordinary surfaces", () => {
    const root = resolve(import.meta.dirname, "..");
    const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
    const evolution = readFileSync(
      resolve(root, "src/renderer/src/EvolutionWorkspace.tsx"),
      "utf8"
    );
    const demo = readFileSync(resolve(root, "src/renderer/src/demo.ts"), "utf8");
    const training = readFileSync(resolve(root, "docs/TRAINING.md"), "utf8");
    const architecture = readFileSync(resolve(root, "docs/ARCHITECTURE.md"), "utf8");
    const catalogFormats = readFileSync(resolve(root, "docs/CATALOG_FORMATS.md"), "utf8");
    const trainingCopy = training.replace(/\s+/g, " ");
    const architectureCopy = architecture.replace(/\s+/g, " ");
    const catalogCopy = catalogFormats.replace(/\s+/g, " ");

    expect(app).not.toContain("Plasticity stages");
    expect(app).not.toContain("plasticity events</span>");
    expect(app).not.toContain('["Neural workspace"');
    expect(app.toLocaleLowerCase()).not.toContain("neural workspace");
    expect(app).not.toContain("Memory settling");
    expect(app).not.toContain("Fading scratch trail");
    expect(app).not.toContain("Active focus");
    expect(app.toLocaleLowerCase()).not.toContain("connected memory");
    expect(app.toLocaleLowerCase()).not.toContain("connected patterns");
    expect(app).not.toContain("Active assemblies");
    expect(app).not.toContain("active neural assemblies");
    expect(app).not.toContain("neural-memory slots");
    expect(app).not.toContain('["Active learned neural activity"');
    expect(app).not.toContain('["Working-memory space"');
    expect(app).not.toContain("memory occupancy");
    expect(app).not.toContain("EXPERIENCE PIPELINE");
    expect(app).not.toContain("Learn fully");
    expect(app).not.toContain("mutation stages will be recorded");
    expect(app).not.toContain("steps.length} stages");
    expect(app).not.toContain("trace-step__index");
    expect(app).not.toContain("trace-step__line");
    expect(app).not.toContain("BACKGROUND NEURAL LEARNING");
    expect(app).not.toContain('memoryRecipe: "human-consolidation"');
    expect(app).toContain("LEARNING FROM DATA");
    expect(app).toContain("Unordered evidence from this event; it is not a fixed memory sequence.");
    expect(evolution).not.toContain("latent replay");
    expect(evolution.toLocaleLowerCase()).not.toContain("connected patterns");
    expect(demo).not.toContain('stage: "Plasticity"');
    expect(demo).not.toContain('kind: "consolidation"');
    expect(demo.toLocaleLowerCase()).not.toContain("connected patterns");
    expect(training).not.toContain("**Consolidate:**");
    expect(trainingCopy).toContain("they are not a fixed pipeline");
    expect(trainingCopy).toContain("there is no manual consolidation step");
    expect(trainingCopy).toContain("fast temporal and episodic synapses");
    expect(trainingCopy).toContain("replay can carry selected activity into slower learned-weight updates");
    expect(architectureCopy).toContain("it is not a fixed memory-stage conveyor");
    expect(architectureCopy).toContain("spreading activation, rehearsal, stability, interference, decay");
    expect(catalogCopy).toContain("not a menu of fixed memory stages or manual consolidation controls");

    // Technical vocabulary remains available only in the deliberately opened
    // research-details disclosure rather than as a personality/memory control.
    const memoryStoryStart = app.indexOf(
      '<section className="working-memory-explainer memory-story-card"'
    );
    const diagnosticsStart = app.indexOf(
      '<details className="research-diagnostics">',
      memoryStoryStart
    );
    const ordinaryMemoryStory = app.slice(memoryStoryStart, diagnosticsStart);
    expect(memoryStoryStart).toBeGreaterThan(-1);
    expect(diagnosticsStart).toBeGreaterThan(memoryStoryStart);
    expect(ordinaryMemoryStory).toContain("Each whole experience changes what is active now");
    expect(ordinaryMemoryStory).not.toMatch(/\b(?:LIF|STDP|assembl(?:y|ies)|slow weights)\b/);
    expect(app).toContain('<details className="research-diagnostics">');
    expect(app).toContain("Each whole experience changes what is active now");
    expect(app).toContain("there is no manual memory step");
    expect(app).toContain("<small>Working activity</small>");
    expect(app).toContain("<small>Fast temporal memory</small>");
    expect(app).toContain("<small>Spreading recall</small>");
    expect(app).toContain("<small>Slow learning</small>");
    expect(app).toContain("<small>Retention dynamics</small>");
    expect(app).toContain("LIF + episodic STDP synapses");
    expect(app).toContain("Replay into learned weights");
    expect(app).toContain("Salience + interference + stability + decay");
    expect(app).toContain("function BrainMapWorkspace(");
    expect(app).toContain('entity: clustered ? "overview" : zoom >= 1.8 ? "neurons" : "assemblies"');
  });
});
