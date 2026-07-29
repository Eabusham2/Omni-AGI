import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG } from "../src/shared/types";

describe("organic worker cognition and neural feedback service", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-feedback-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("sends response feedback to worker STDP without mutating the desktop graph", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Feedback brain"
    });
    const message = {
      id: randomUUID(),
      role: "brain" as const,
      content: "A learned response.",
      createdAt: new Date().toISOString(),
      traceId: "trace-feedback"
    };
    brain.messages.push(message);
    brain.synapses.desktop = {
      id: "desktop",
      sourceId: "a",
      targetId: "b",
      effectiveWeight: 1,
      latentWeight: 0.8,
      stability: 0.4,
      plasticity: 1,
      uses: 2,
      lastUpdatedAt: new Date(0).toISOString()
    };
    await repository.save(brain);
    const tryRequest = vi.fn().mockResolvedValue({
      direction: "down",
      stdp: { stdp_update: 0.5, plasticity_events: 12 },
      synapseChecksumBefore: "a".repeat(64),
      synapseChecksumAfter: "b".repeat(64),
      rewardModel: false,
      rlhf: false,
      metrics: { plasticityEvents: 12 }
    });
    const service = new BrainService(
      repository,
      { tryRequest } as unknown as EngineSupervisor
    );

    const updated = await service.feedback({
      brainId: brain.id,
      messageId: message.id,
      direction: "down"
    });

    expect(tryRequest).toHaveBeenCalledWith(
      "feedback",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: repository.brainDirectory(brain.id),
        messageId: message.id,
        traceId: message.traceId,
        text: message.content,
        direction: "down"
      },
      120_000
    );
    expect(updated.synapses.desktop).toEqual(brain.synapses.desktop);
    expect(updated.counters.plasticityEvents).toBe(12);
    expect(updated.journal?.at(-1)).toMatchObject({
      kind: "learning",
      summary: "Integrated down feedback through neural STDP."
    });
  });

  it("normalizes worker-proposed idle actions as organic and exposes only enabled tools", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Idle brain",
      idleCognition: true
    });
    const tryRequest = vi.fn().mockResolvedValue({
      brainId: brain.id,
      ran: true,
      actions: [
        {
          kind: "imagine",
          toolId: "modality.imagine",
          action: "generate",
          arguments: { modality: "image", conceptIds: ["assembly-a"] },
          confidence: 0.91
        }
      ],
      metrics: { plasticityEvents: 7 }
    });
    const service = new BrainService(
      repository,
      { tryRequest } as unknown as EngineSupervisor
    );

    const result = await service.idleCycle(brain.id, 0);

    expect(result).toMatchObject({
      ran: true,
      actions: [
        {
          kind: "imagine",
          source: "organic",
          toolId: "modality.imagine",
          action: "generate"
        }
      ]
    });
    const workerParams = tryRequest.mock.calls[0]?.[1] as {
      toolSchemas: Array<{ id: string; grant: string }>;
    };
    expect(workerParams.toolSchemas.some((schema) => schema.id === "source.self-modify")).toBe(
      false
    );
    expect(workerParams.toolSchemas).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "modality.imagine", grant: "auto" })
      ])
    );
  });

  it("persists a prompt-free organic talk action in the continuous chat", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Spontaneous brain",
      idleCognition: true
    });
    const tryRequest = vi.fn().mockResolvedValue({
      brainId: brain.id,
      ran: true,
      trace: {
        id: "idle-trace",
        createdAt: new Date().toISOString(),
        mode: "ponder",
        seed: 17,
        promptTokenCount: 0,
        hiddenBehavioralPrompt: false,
        activeAssemblyIds: ["assembly-a"],
        organicState: {},
        liquidControls: {},
        stdpUpdate: 0.01,
        spikeRate: 0.2,
        rehearsal: {
          loss: 0.4,
          reconstructionLoss: 0.2,
          temporalLoss: 0.1,
          stabilityLoss: 0.1
        },
        parameterChecksumBefore: "a".repeat(64),
        parameterChecksumAfter: "b".repeat(64),
        parameterDeltaNorm: 0.2,
        actionPolicyScores: {
          talk: 0.93,
          tool: 0.01,
          imagine: 0.01,
          agent: 0.01,
          ponder: 0.01,
          learn: 0.01,
          evolve: 0.01,
          stop: 0.01
        },
        proposedActionKinds: ["talk"],
        note: "Measured prompt-free activity."
      },
      actions: [
        {
          kind: "talk",
          arguments: {
            message: "I wonder whether these two memories share a pattern.",
            promptTokenCount: 0,
            organic: true
          },
          confidence: 0.93
        }
      ],
      metrics: { plasticityEvents: 8 }
    });
    const service = new BrainService(
      repository,
      { tryRequest } as unknown as EngineSupervisor
    );

    const result = await service.idleCycle(brain.id, 0);
    const reloaded = await repository.get(brain.id);

    expect(result.actions).toEqual([
      expect.objectContaining({ kind: "talk", source: "organic" })
    ]);
    expect(reloaded.messages.at(-1)).toMatchObject({
      role: "brain",
      content: "I wonder whether these two memories share a pattern.",
      traceId: "idle-trace",
      runtime: "adaptive-core"
    });
    expect(reloaded.journal?.at(-1)).toMatchObject({
      kind: "reflection",
      summary: "Spoke from prompt-free idle cognition."
    });
  });
});
