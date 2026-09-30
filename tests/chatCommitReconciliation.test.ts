import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG } from "../src/shared/types";

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

const input = "Hi. Please answer with one short friendly sentence.";
const answer = "(This will be a text-to-speech option in Windows Vista)";
const humanMessageId = "6d613254a9be43889ac22012e85cfe49";
const brainMessageId = "f7bbe605aae24e0083bb63108234707f";
const traceId = "ee58d831f29d47bbba906daac6fd4ebe";
const createdAt = "2026-09-07T04:23:48.000Z";
const engineUpdatedAt = "2026-09-07T04:28:58.000Z";
const inputSha256 = sha256(input);
const checksumBefore = "1".repeat(64);
const checksumAfter = "2".repeat(64);
const substrateGeneration = "3".repeat(64);
const mutableStateGeneration = "4".repeat(64);

function committedEngineState(brainId: string, turnId?: string) {
  return {
    brain_id: brainId,
    updated_at: engineUpdatedAt,
    messages: [
      {
        id: humanMessageId,
        role: "human",
        content: input,
        created_at: createdAt,
        ...(turnId ? { turn_id: turnId } : {})
      },
      {
        id: brainMessageId,
        role: "brain",
        content: answer,
        created_at: createdAt,
        ...(turnId ? { turn_id: turnId } : {})
      }
    ],
    traces: [
      {
        id: traceId,
        created_at: createdAt,
        input_sha256: inputSha256,
        parameter_checksum_before: checksumBefore,
        parameter_checksum_after: checksumAfter,
        parameter_delta_norm: 0.125,
        stdp_update: 0.25,
        train_loss: 0.5,
        ponder_steps: 3,
        steps: [{ stage: "generate", detail: "Committed exact reply." }],
        ...(turnId ? { turn_id: turnId } : {})
      }
    ],
    counters: {
      inference_count: 1,
      plasticity_events: 17_587_341,
      consolidation_cycles: 0
    },
    ...(turnId
      ? {
          completed_chat_turns: [
            {
              format: "omni-completed-chat-turn",
              formatVersion: 1,
              turnId,
              inputSha256,
              humanMessageId,
              brainMessageId,
              traceId,
              inferenceCount: 1,
              parameterChecksumAfter: checksumAfter,
              committedAt: engineUpdatedAt
            }
          ]
        }
      : {}),
    substrate: {
      persistence: { activeGeneration: substrateGeneration }
    },
    mutable_state: { activeGeneration: mutableStateGeneration }
  };
}

function committedReceipt(brainId: string, turnId: string, legacyMatched = false) {
  return {
    format: "omni-chat-turn-receipt-query",
    formatVersion: 1,
    brainId,
    turnId,
    committed: true,
    turnCommitted: true,
    legacyMatched,
    inputSha256,
    humanMessage: {
      id: humanMessageId,
      role: "human",
      content: input,
      createdAt
    },
    brainMessage: {
      id: brainMessageId,
      role: "brain",
      content: answer,
      createdAt,
      traceId
    },
    trace: {
      ...committedEngineState(brainId, legacyMatched ? undefined : turnId).traces[0]
    },
    inferenceCount: 1,
    plasticityEvents: 17_587_341,
    consolidationCycles: 0,
    parameterChecksumAfter: checksumAfter,
    engineUpdatedAt,
    substrateGeneration,
    mutableStateGeneration,
    idempotentCompletion: true
  };
}

function committedWorkerResult(brainId: string, turnId: string, replayed: boolean) {
  const state = committedEngineState(brainId, turnId);
  return {
    text: answer,
    response: answer,
    content: answer,
    humanMessage: state.messages[0],
    message: state.messages[1],
    trace: state.traces[0],
    metrics: {
      plasticityEvents: 17_587_341,
      counters: {
        inference_count: 1,
        consolidation_cycles: 0
      }
    },
    actions: [],
    turnReceipt: {
      format: "omni-completed-chat-turn",
      formatVersion: 1,
      turnId,
      inputSha256,
      humanMessageId,
      brainMessageId,
      traceId,
      inferenceCount: 1,
      parameterChecksumAfter: checksumAfter,
      committedAt: engineUpdatedAt
    },
    turnCommitted: true,
    idempotentCompletion: replayed
  };
}

function noReplyEngineState(brainId: string, turnId: string) {
  const state = committedEngineState(brainId, turnId);
  return {
    ...state,
    messages: state.messages.map((message) => ({ ...message,
      content: message.role === "brain" ? "" : message.content, generation_end: "no-reply" })),
    traces: state.traces.map((trace) => ({ ...trace, generation_stop_reason: "no-reply",
      generation_decoder_stop_reason: "eos", generation_no_reply_reason: "no-decoded-text",
      generated_token_count: 1, generation_printable_text_characters: 0 })),
    completed_chat_turns: state.completed_chat_turns!.map((receipt) => ({ ...receipt, generationEnd: "no-reply" }))
  };
}

describe("atomic neural chat commit reconciliation", () => {
  const temporaryRoots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      temporaryRoots.splice(0).map((root) =>
        rm(root, { recursive: true, force: true })
      )
    );
  });

  async function fixture(name: string) {
    const root = await mkdtemp(join(tmpdir(), "omni-chat-receipt-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name });
    await mkdir(join(repository.brainDirectory(brain.id), "engine"), {
      recursive: true
    });
    return { repository, brain };
  }

  it("recovers an exact committed retry after context shrinks without reloading or learning", async () => {
    const { repository, brain } = await fixture("Completed input after smaller context");
    brain.config.contextWindowTokens = 8;
    await repository.save(brain);
    const turnId = "turn-completed-before-window-shrank";
    await writeFile(join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify(committedEngineState(brain.id, turnId)), "utf8");
    const request = vi.fn(async () => { throw new Error("No worker is permitted for a committed retry."); });
    const requestStream = vi.fn(async () => { throw new Error("No new decode is permitted."); });
    const claimForeground = vi.fn();
    const service = new BrainService(repository, { request, requestStream, claimForeground } as unknown as EngineSupervisor);
    const first = await service.chat(brain.id, input, undefined, undefined, turnId);
    const second = await service.chat(brain.id, input, undefined, undefined, turnId);
    expect(first.brainMessage.content).toBe(answer);
    expect(second.brainMessage.id).toBe(first.brainMessage.id);
    expect((await repository.get(brain.id)).counters.inferenceCount).toBe(1);
    await expect(service.chat(brain.id, input + " changed", undefined, undefined, turnId)).rejects.toThrow("Send is blocked");
    expect(request).not.toHaveBeenCalled();
    expect(requestStream).not.toHaveBeenCalled();
    expect(claimForeground).not.toHaveBeenCalled();
  });

  it("recovers the exact authoritative turn when Stop wins after the worker's atomic commit", async () => {
    const { repository, brain } = await fixture("Commit then cancel");
    const controller = new AbortController();
    const turnId = "turn-commit-before-cancel";
    const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method === "load") return {};
      if (method === "chat_receipt") {
        expect(params).toMatchObject({
          brainId: brain.id,
          storagePath: repository.brainDirectory(brain.id),
          turnId,
          inputSha256,
          minimumInferenceCount: 0
        });
        return committedReceipt(brain.id, turnId);
      }
      throw new Error(`Unexpected worker method: ${method}`);
    });
    const requestStream = vi.fn(async () => {
      // This is the exact failure window: the worker's durable commit exists,
      // then Stop aborts the desktop request before repository.save can run.
      await writeFile(
        join(repository.brainDirectory(brain.id), "engine", "brain.json"),
        JSON.stringify(committedEngineState(brain.id, turnId)),
        "utf8"
      );
      controller.abort();
      return { text: answer };
    });
    const service = new BrainService(
      repository,
      {
        claimForeground: vi.fn(async () => undefined),
        request,
        requestStream
      } as unknown as EngineSupervisor
    );

    const result = await service.chat(
      brain.id,
      input,
      controller.signal,
      undefined,
      turnId
    );

    expect(result.humanMessage).toMatchObject({
      id: humanMessageId,
      role: "human",
      content: input,
      turnId,
      status: "complete"
    });
    expect(result.brainMessage).toMatchObject({
      id: brainMessageId,
      role: "brain",
      content: answer,
      turnId,
      traceId,
      status: "complete"
    });
    expect(result.trace.id).toBe(traceId);
    expect((await repository.conversationPage(brain.id)).entries.flatMap(
      (entry) => entry.message ? [entry.message.id] : []
    )).toEqual([
      humanMessageId,
      brainMessageId
    ]);
    expect(result.brain.counters).toMatchObject({
      inferenceCount: 1,
      plasticityEvents: 17_587_341,
      consolidationCycles: 0
    });

    const reopened = await service.getReconciledBrain(brain.id);
    expect(reopened.messages).toEqual([]);
    expect((await repository.conversationPage(brain.id)).entries.flatMap(
      (entry) => entry.message ? [entry.message.id] : []
    )).toEqual([
      humanMessageId,
      brainMessageId
    ]);
    expect(reopened.traces.map((trace) => trace.id)).toEqual([traceId]);
    // The persisted checkpoint receipt proves this turn locally; no worker
    // restart/query is needed just to recover its UI projection.
    expect(request.mock.calls.filter(([method]) => method === "chat_receipt")).toHaveLength(0);
  });

  it("recovers a legacy split on open and remains idempotent across concurrent get and restart", async () => {
    const { repository, brain } = await fixture("Restart reconciliation");
    await writeFile(
      join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify(committedEngineState(brain.id)),
      "utf8"
    );
    const syntheticTurnId = `reconcile-${sha256(traceId)}`;
    const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "chat_receipt") {
        throw new Error(`Unexpected worker method: ${method}`);
      }
      expect(params).toMatchObject({
        brainId: brain.id,
        turnId: syntheticTurnId,
        inputSha256,
        minimumInferenceCount: 0
      });
      return committedReceipt(brain.id, syntheticTurnId, true);
    });
    const engine = { request } as unknown as EngineSupervisor;
    const firstService = new BrainService(repository, engine);

    const [firstOpen, simultaneousGet] = await Promise.all([
      firstService.getReconciledBrain(brain.id),
      firstService.getReconciledBrain(brain.id)
    ]);
    for (const recovered of [firstOpen, simultaneousGet]) {
      expect(recovered.messages).toEqual([]);
      expect(recovered.traces.map((trace) => trace.id)).toEqual([]);
      expect(recovered.counters.inferenceCount).toBe(1);
    }
    expect((await repository.recentConversationEvidence(brain.id)).flatMap(
      (entry) => entry.trace ? [entry.trace.id] : []
    )).toEqual([traceId]);

    // A new service instance represents app restart. The synchronized counter
    // makes the split detector a no-op and the exact pair is never duplicated.
    const restartedService = new BrainService(repository, engine);
    const afterRestart = await restartedService.getReconciledBrain(brain.id);
    expect(afterRestart.messages).toHaveLength(0);
    expect(afterRestart.traces).toHaveLength(1);
    expect((await repository.conversationPage(brain.id)).entries.flatMap(
      (entry) => entry.message ? [entry.message.id] : []
    )).toEqual([humanMessageId, brainMessageId]);
  });

  it("uses exact worker ids on ordinary success and never duplicates an idempotent replay", async () => {
    const { repository, brain } = await fixture("Idempotent acknowledgement replay");
    const turnId = "turn-idempotent-replay";
    let calls = 0;
    const requestStream = vi.fn(async () => {
      calls += 1;
      return committedWorkerResult(brain.id, turnId, calls > 1);
    });
    const service = new BrainService(
      repository,
      {
        request: vi.fn(async (method: string) => {
          if (method === "load") return {};
          throw new Error(`Unexpected worker method: ${method}`);
        }),
        requestStream
      } as unknown as EngineSupervisor
    );

    const first = await service.chat(brain.id, input, undefined, undefined, turnId);
    const replay = await service.chat(brain.id, input, undefined, undefined, turnId);

    for (const result of [first, replay]) {
      expect(result.humanMessage.id).toBe(humanMessageId);
      expect(result.humanMessage.turnId).toBe(turnId);
      expect(result.brainMessage.id).toBe(brainMessageId);
      expect(result.brainMessage.turnId).toBe(turnId);
      expect(result.trace.id).toBe(traceId);
      expect(result.brain.messages).toHaveLength(0);
      expect(result.brain.traces).toHaveLength(0);
    }
    const saved = await repository.get(brain.id);
    expect((await repository.conversationPage(brain.id)).entries.flatMap(
      (entry) => entry.message ? [{ id: entry.message.id, turnId: entry.message.turnId }] : []
    )).toEqual([
      { id: humanMessageId, turnId },
      { id: brainMessageId, turnId }
    ]);
    expect(saved.traces.map((trace) => trace.id)).toEqual([traceId]);
    expect(saved.counters.inferenceCount).toBe(1);
  });

  it("commits typed no-reply with zero assistant text and keeps exact ids on idempotent replay", async () => {
    const { repository, brain } = await fixture("No printable reply");
    const turnId = "turn-no-reply";
    const state = noReplyEngineState(brain.id, turnId);
    let calls = 0;
    const requestStream = vi.fn(async () => {
      await writeFile(join(repository.brainDirectory(brain.id), "engine", "brain.json"), JSON.stringify(state), "utf8");
      return { ...committedWorkerResult(brain.id, turnId, calls++ > 0), text: "", response: "", content: "",
        noReply: true, humanMessage: state.messages[0], message: state.messages[1], trace: state.traces[0],
        turnReceipt: state.completed_chat_turns[0] };
    });
    const engine = { request: vi.fn(async (method: string) => {
      if (method === "load") return {};
      throw new Error(`Unexpected worker method: ${method}`);
    }), requestStream } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine);
    for (const result of [await service.chat(brain.id, input, undefined, undefined, turnId),
      await service.chat(brain.id, input, undefined, undefined, turnId)]) {
      expect(result.generationEnd).toBe("no-reply");
      expect(result.humanMessage).toMatchObject({ id: humanMessageId, content: input, generationEnd: "no-reply" });
      expect(result.brainMessage).toMatchObject({ id: brainMessageId, content: "", generationEnd: "no-reply" });
      expect(result.trace.generation).toEqual({ disposition: "no-reply", decoderStopReason: "eos",
        noReplyReason: "no-decoded-text", generatedTokenCount: 1, printableTextCharacters: 0 });
    }
    const messages = (await repository.conversationPage(brain.id)).entries.flatMap((entry) => entry.message ? [entry.message] : []);
    expect(messages.map((message) => message.content)).toEqual([input, ""]);
    expect(messages.every((message) => message.generationEnd === "no-reply")).toBe(true);
    expect((await repository.get(brain.id)).counters.inferenceCount).toBe(1);
  });

  it("recovers no-reply after a lost acknowledgement and stays visible across restart without a worker load", async () => {
    const { repository, brain } = await fixture("No-reply acknowledgement recovery");
    const turnId = "turn-no-reply-lost-ack";
    const state = noReplyEngineState(brain.id, turnId);
    const request = vi.fn(async (method: string) => {
      if (method === "load") return {};
      throw new Error(`No-reply reconciliation must not load or query a worker: ${method}`);
    });
    const service = new BrainService(repository, { request, requestStream: vi.fn(async () => {
      await writeFile(join(repository.brainDirectory(brain.id), "engine", "brain.json"), JSON.stringify(state), "utf8");
      throw new Error("simulated acknowledgement loss after atomic no-reply commit");
    }) } as unknown as EngineSupervisor);
    const recovered = await service.chat(brain.id, input, undefined, undefined, turnId);
    expect(recovered).toMatchObject({ generationEnd: "no-reply", humanMessage: { content: input },
      brainMessage: { content: "", generationEnd: "no-reply" } });
    expect(recovered.trace.generation?.printableTextCharacters).toBe(0);
    const restarted = new BrainService(repository, { request: vi.fn(async () => { throw new Error("unexpected worker"); }) } as unknown as EngineSupervisor);
    await restarted.getReconciledBrain(brain.id);
    const entries = (await repository.conversationPage(brain.id)).entries;
    expect(entries.flatMap((entry) => entry.message ? [{ id: entry.message.id, content: entry.message.content,
      generationEnd: entry.message.generationEnd }] : [])).toEqual([
      { id: humanMessageId, content: input, generationEnd: "no-reply" },
      { id: brainMessageId, content: "", generationEnd: "no-reply" }
    ]);
    expect((await repository.recentConversationEvidence(brain.id)).flatMap((entry) => entry.trace ? [entry.trace.id] : []))
      .toEqual([traceId]);
    expect(request.mock.calls.map(([method]) => method)).toEqual(["load"]);
  });

  it("reconciles a no-reply split on open using its receipt without manufacturing a new response", async () => {
    const { repository, brain } = await fixture("No-reply split on open");
    await writeFile(join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify(noReplyEngineState(brain.id, "turn-no-reply-on-open")), "utf8");
    const service = new BrainService(repository, { request: vi.fn(async () => { throw new Error("unexpected worker"); }) } as unknown as EngineSupervisor);
    const reopened = await service.getReconciledBrain(brain.id);
    expect(reopened.counters.inferenceCount).toBe(1);
    expect((await repository.conversationPage(brain.id)).entries.flatMap((entry) => entry.message ? [entry.message.content] : []))
      .toEqual([input, ""]);
  });

  it("does not invent a turn when cancellation has no authoritative commit receipt", async () => {
    const { repository, brain } = await fixture("Cancelled before commit");
    const controller = new AbortController();
    const turnId = "turn-cancelled-before-commit";
    const request = vi.fn(async (method: string) => {
      if (method === "load") return {};
      if (method === "chat_receipt") {
        return {
          format: "omni-chat-turn-receipt-query",
          formatVersion: 1,
          brainId: brain.id,
          turnId,
          committed: false,
          turnCommitted: false,
          inputSha256,
          inferenceCount: 0
        };
      }
      throw new Error(`Unexpected worker method: ${method}`);
    });
    const service = new BrainService(
      repository,
      {
        request,
        requestStream: vi.fn(async () => {
          controller.abort();
          return { text: "provisional only" };
        })
      } as unknown as EngineSupervisor
    );

    await expect(
      service.chat(brain.id, input, controller.signal, undefined, turnId)
    ).rejects.toMatchObject({ name: "AbortError" });
    const saved = await repository.get(brain.id);
    expect(saved.messages).toEqual([]);
    expect(saved.traces).toEqual([]);
    expect(saved.counters.inferenceCount).toBe(0);
  });

  it("runs receipt reconciliation from open/history reads and app restart", async () => {
    const [rawIpcSource, rawStartupSource] = await Promise.all([
      readFile(join(process.cwd(), "src/main/ipc.ts"), "utf8"),
      readFile(join(process.cwd(), "src/main/index.ts"), "utf8")
    ]);
    const ipcSource = rawIpcSource.replace(/\r\n/g, "\n");
    const startupSource = rawStartupSource.replace(/\r\n/g, "\n");
    expect(ipcSource).toContain(
      "handle(IPC.brain.get, (_event, id: string) =>\n    service.getReconciledBrain"
    );
    expect(ipcSource).toContain(
      "await service.getReconciledBrain(brainId)"
    );
    expect(ipcSource).toContain(
      "(await service.getReconciledBrain(requireId(brainId))).traces"
    );
    expect(startupSource).toContain(
      "await service.getReconciledBrain(summary.id).catch"
    );
  });
});
