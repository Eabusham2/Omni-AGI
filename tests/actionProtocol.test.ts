import { describe, expect, it, vi } from "vitest";
import {
  normalizeStructuredAction,
  parseModelActions
} from "../src/main/actionProtocol";
import { ChatActionController } from "../src/main/chatActionController";
import type {
  ChatResult,
  EvolutionRun,
  RuntimeJob,
  ToolExecutionResult
} from "../src/shared/types";

function chatResult(content: string, proposedActions?: ChatResult["proposedActions"]): ChatResult {
  const now = new Date().toISOString();
  return {
    brain: { messages: [] } as unknown as ChatResult["brain"],
    humanMessage: { id: "human", role: "human", content: "input", createdAt: now },
    brainMessage: { id: "brain", role: "brain", content, createdAt: now },
    trace: { id: "trace" } as unknown as ChatResult["trace"],
    proposedActions
  };
}

describe("structured chat actions", () => {
  it("never manufactures an action from human or model prose", () => {
    for (const prose of [
      "Create an image of a cobalt memory palace",
      "Ask three agents to compare the hypotheses",
      "Recursively improve your own source",
      '/tool web.fetch fetch {"url":"https://example.com"}',
      '<omni-tool>{"toolId":"web.search","action":"search"}</omni-tool>'
    ]) {
      expect(parseModelActions(prose)).toEqual([]);
    }
  });

  it("accepts only the dedicated model action channel and ignores response-text tags", () => {
    expect(
      normalizeStructuredAction(
        {
          kind: "agent",
          arguments: { objective: "test both approaches" },
          confidence: 1.7
        },
        "brain"
      )
    ).toMatchObject({
      kind: "agent",
      source: "brain",
      toolId: "agent.fork",
      action: "start",
      confidence: 1
    });
    const actions = parseModelActions(
      '<omni-tool>{"toolId":"web.search","action":"search","arguments":{"query":"ternary kernels"}}</omni-tool>',
      [
        {
          kind: "imagine",
          arguments: { modality: "audio", prompt: "rain" }
        }
      ]
    );
    expect(actions).toHaveLength(1);
    expect(actions.map((action) => action.kind)).toEqual(["imagine"]);
  });

  it("executes natural actions in the trusted controller and feeds visible results back as experience", async () => {
    const proposed = {
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: {
        modality: "image",
        conceptIds: ["emergent-city"]
      }
    };
    const service = {
      chat: vi
        .fn(async (
          _brainId: string,
          _input: string,
          _signal?: AbortSignal,
          onStream?: (event: {
            type: "chat-action";
            sequence: number;
            actionId: string;
            action: typeof proposed;
          }) => void
        ) => {
          if (service.chat.mock.calls.length === 1) {
            onStream?.({
              type: "chat-action",
              sequence: 0,
              actionId: "11111111111111111111111111111111",
              action: proposed
            });
            return chatResult("I will form that image.");
          }
          return chatResult("The generated artifact is now part of this turn.");
        })
    };
    const execution: ToolExecutionResult = {
      id: "execution",
      toolId: "modality.imagine",
      action: "generate",
      state: "complete",
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      output: { artifactPath: "artifact.png" }
    };
    const tools = { execute: vi.fn().mockResolvedValue(execution), cancel: vi.fn(() => 0) };
    const evolution = { start: vi.fn() };
    const controller = new ChatActionController(service, tools, evolution);
    const streamed: string[] = [];
    controller.on("event", (event) => streamed.push(event.state));

    const result = await controller.send("brain-1", "Generate an image of an emergent city");

    expect(tools.execute).toHaveBeenCalledWith(
      {
        brainId: "brain-1",
        toolId: "modality.imagine",
        action: "generate",
        arguments: {
          modality: "image",
          conceptIds: ["emergent-city"],
          neuralActionId: "11111111111111111111111111111111"
        }
      },
      expect.any(Function)
    );
    expect(service.chat).toHaveBeenCalledTimes(2);
    expect(service.chat.mock.calls[1]?.[1]).toContain("[Visible structured action result]");
    expect(result.actionEvents).toEqual([
      expect.objectContaining({ state: "complete", execution })
    ]);
    expect(streamed).toEqual(["proposed", "running", "complete"]);
  });

  it("settles repeated imagination intent when only transient assemblies change", async () => {
    const recurring = (assemblyId: string) => ({
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: {
        modality: "image",
        assemblyIds: [assemblyId]
      }
    });
    const service = {
      chat: vi
        .fn()
        .mockResolvedValueOnce(chatResult("I formed the scene.", [recurring("first")]))
        .mockResolvedValueOnce(chatResult("The artifact changed my active assembly.", [recurring("second")]))
    };
    const tools = {
      execute: vi.fn().mockResolvedValue({
        id: "one-image",
        toolId: "modality.imagine",
        action: "generate",
        state: "complete",
        startedAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
        output: { artifactPath: "one-image.png" }
      } satisfies ToolExecutionResult),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });

    const result = await controller.send("brain-settling", "Visualize this scene.");

    expect(service.chat).toHaveBeenCalledTimes(2);
    expect(tools.execute).toHaveBeenCalledTimes(1);
    expect(result.actionEvents).toHaveLength(1);
    expect(result.brainMessage.content).toBe(
      "The artifact changed my active assembly."
    );
  });

  it("streams organic imagination progress while the response remains in flight", async () => {
    const proposed = {
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: {
        modality: "video",
        conceptIds: ["story-scene", "rain"]
      }
    };
    let finishInitial!: (result: ChatResult) => void;
    const initial = new Promise<ChatResult>((resolve) => {
      finishInitial = resolve;
    });
    let calls = 0;
    const service = {
      chat: vi.fn((
        _brainId: string,
        _input: string,
        _signal?: AbortSignal,
        onStream?: (event:
          | { type: "chat-token"; sequence: number; delta: string }
          | {
              type: "chat-action";
              sequence: number;
              actionId: string;
              action: typeof proposed;
            }
          | {
              type: "modality-preview";
              sequence: number;
              actionId: string;
              preview: {
                revision: number;
                progress: number;
                statusLabel: string;
                mimeType: string;
                dataUrl: string;
              };
            }
        ) => void
      ) => {
        calls += 1;
        if (calls > 1) {
          return Promise.resolve(chatResult("The moving scene is now part of this turn."));
        }
        onStream?.({ type: "chat-token", sequence: 0, delta: "The scene " });
        onStream?.({
          type: "chat-action",
          sequence: 1,
          actionId: "22222222222222222222222222222222",
          action: proposed
        });
        onStream?.({
          type: "modality-preview",
          sequence: 2,
          actionId: "22222222222222222222222222222222",
          preview: {
            revision: 0,
            progress: 0.08,
            statusLabel: "A first temporal sketch",
            mimeType: "video/mp4",
            dataUrl: "data:video/mp4;base64,Zmlyc3Q="
          }
        });
        return initial;
      })
    };
    const execution: ToolExecutionResult = {
      id: "video-execution",
      toolId: "modality.imagine",
      action: "generate",
      state: "complete",
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      output: {
        dataUrl: "data:video/mp4;base64,fixture",
        mimeType: "video/mp4"
      }
    };
    const updates: RuntimeJob[] = [
      {
        id: "video-job",
        brainId: "brain-story",
        kind: "video",
        state: "running",
        progress: 0.18,
        label: "Forming temporal latents",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: "video-job",
        brainId: "brain-story",
        kind: "video",
        state: "running",
        progress: 0.72,
        label: "Decoding frames and sound",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: "video-job",
        brainId: "brain-story",
        kind: "video",
        state: "running",
        progress: 0.9,
        label: "Refreshing the live preview",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        preview: {
          revision: 1,
          progress: 0.9,
          statusLabel: "Refreshing the live preview",
          mimeType: "video/mp4",
          dataUrl: "data:video/mp4;base64,c2Vjb25k"
        }
      }
    ];
    const tools = {
      execute: vi.fn(
        async (
          _invocation: unknown,
          onProgress?: (job: RuntimeJob) => void
        ): Promise<ToolExecutionResult> => {
          await Promise.resolve();
          for (const update of updates) onProgress?.(update);
          return execution;
        }
      ),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const streamed: Array<{
      state: string;
      progress?: number;
      label?: string;
    }> = [];
    controller.on("event", (event) =>
      streamed.push({
        state: event.state,
        progress: event.progress,
        label: event.statusLabel
      })
    );
    const turnStream: string[] = [];
    controller.on("stream", (event) => turnStream.push(event.type));

    const pending = controller.send(
      "brain-story",
      "Continue the story.",
      undefined,
      "turn-story"
    );
    await vi.waitFor(() => expect(tools.execute).toHaveBeenCalledOnce());
    expect(calls).toBe(1);
    expect(turnStream.slice(0, 3)).toEqual([
      "chat-state",
      "chat-token",
      "chat-action"
    ]);
    expect(turnStream).toContain("modality-preview");
    finishInitial(chatResult("The scene became motion in my workspace."));
    const result = await pending;

    expect(streamed).toEqual([
      { state: "proposed", progress: undefined, label: undefined },
      { state: "running", progress: undefined, label: undefined },
      { state: "running", progress: 0.08, label: "A first temporal sketch" },
      { state: "running", progress: 0.18, label: "Forming temporal latents" },
      { state: "running", progress: 0.72, label: "Decoding frames and sound" },
      { state: "running", progress: 0.9, label: "Refreshing the live preview" },
      { state: "complete", progress: 0.9, label: "Refreshing the live preview" }
    ]);
    expect(result.actionEvents?.[0]).toMatchObject({
      state: "complete",
      runtimeJobId: "video-job",
      progress: 0.9,
      preview: { revision: 1 },
      execution
    });
  });

  it("keeps a typed worker action authoritative without a human regex fallback", async () => {
    const proposed = {
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "audio", conceptIds: ["learned-choice"] }
    };
    const service = {
      chat: vi
        .fn()
        .mockResolvedValueOnce(chatResult("I heard it instead.", [proposed]))
        .mockResolvedValueOnce(chatResult("The sound returned."))
    };
    const tools = {
      execute: vi.fn().mockResolvedValue({
        id: "audio",
        toolId: "modality.imagine",
        action: "generate",
        state: "complete",
        startedAt: new Date().toISOString()
      } satisfies ToolExecutionResult),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });

    await controller.send("brain-choice", "Create an image of a city");

    expect(tools.execute).toHaveBeenCalledWith(
      expect.objectContaining({
        arguments: { modality: "audio", conceptIds: ["learned-choice"] }
      }),
      expect.any(Function)
    );
  });

  it("executes a ponder action as an additional prompt-free neural cycle", async () => {
    const ponder = {
      kind: "ponder" as const,
      source: "brain" as const,
      arguments: { assemblyIds: ["unresolved-assembly"], organic: true },
      confidence: 0.91
    };
    const service = {
      chat: vi.fn().mockResolvedValue(
        chatResult("I need another internal pass.", [ponder])
      ),
      idleCycle: vi.fn().mockResolvedValue({
        brainId: "brain-ponder",
        ran: true,
        actions: [],
        trace: {
          mode: "ponder",
          promptTokenCount: 0,
          hiddenBehavioralPrompt: false,
          parameterDeltaNorm: 0.02
        }
      })
    };
    const tools = {
      execute: vi.fn(),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(
      service,
      tools,
      { start: vi.fn() }
    );

    const result = await controller.send(
      "brain-ponder",
      "Stay with the unresolved relation."
    );

    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.idleCycle).toHaveBeenCalledOnce();
    expect(service.idleCycle).toHaveBeenCalledWith("brain-ponder", 0);
    expect(tools.execute).not.toHaveBeenCalled();
    expect(result.actionEvents).toEqual([
      expect.objectContaining({
        state: "complete",
        action: expect.objectContaining({ kind: "ponder" })
      })
    ]);
  });

  it("runs model-proposed actions, stops on approval, and archives evolve requests", async () => {
    const proposed = {
      kind: "tool" as const,
      source: "brain" as const,
      toolId: "windows.files",
      action: "write",
      arguments: { path: "C:\\candidate.txt", content: "candidate" }
    };
    const service = {
      chat: vi.fn(async (
        _brainId: string,
        _input: string,
        _signal?: AbortSignal,
        onStream?: (event: {
          type: "chat-action";
          sequence: number;
          actionId: string;
          action: typeof proposed;
        }) => void
      ) => {
        onStream?.({
          type: "chat-action",
          sequence: 0,
          actionId: "permission-action",
          action: proposed
        });
        return chatResult("proposal");
      })
    };
    const tools = {
      execute: vi.fn().mockResolvedValue({
        id: "approval",
        toolId: "windows.files",
        action: "write",
        state: "approval-required",
        approvalToken: "locked",
        startedAt: new Date().toISOString()
      } satisfies ToolExecutionResult),
      cancel: vi.fn(() => 0)
    };
    const evolution = { start: vi.fn() };
    const controller = new ChatActionController(service, tools, evolution);
    const result = await controller.send("brain-2", "write the candidate");
    expect(service.chat).toHaveBeenCalledTimes(1);
    expect(result.actionEvents?.[0]).toMatchObject({
      state: "approval-required",
      execution: { approvalToken: "locked" }
    });

    const run: EvolutionRun = {
      id: "run-1",
      brainId: "brain-2",
      objective: "reduce latency",
      state: "experimenting",
      recursive: true,
      generation: 0,
      candidateIds: ["candidate-1"],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    const evolveService = {
      chat: vi
        .fn()
        .mockResolvedValueOnce(chatResult("Starting an experiment.", [{
          kind: "evolve",
          source: "brain",
          toolId: "source.self-modify",
          action: "propose",
          arguments: {
            objective: "reduce latency",
            candidateKind: "architecture",
            addExperts: 2,
            texts: ["organic expert-growth evidence"]
          }
        }]))
        .mockResolvedValueOnce(chatResult("The isolated candidate exists."))
    };
    const evolve = { start: vi.fn().mockResolvedValue(run) };
    const evolveController = new ChatActionController(evolveService, tools, evolve);
    const evolved = await evolveController.send(
      "brain-2",
      "Improve your own source to reduce latency"
    );
    expect(evolve.start).toHaveBeenCalledWith({
      brainId: "brain-2",
      objective: "reduce latency",
      recursive: true,
      candidateKind: "architecture",
      texts: ["organic expert-growth evidence"],
      sourceIds: undefined,
      epochs: undefined,
      learningRate: undefined,
      latentReplay: undefined,
      objectives: undefined,
      architectureChange: {
        mutation: "grow-experts",
        addExperts: 2
      }
    });
    expect(evolved.actionEvents?.[0]).toMatchObject({
      state: "complete",
      evolutionRunId: "run-1"
    });

    const typedEdits = [
      {
        path: "src/measured-maintenance.ts",
        content: "export const measuredMaintenance = true;\n",
        expectedSha256: null
      }
    ];
    const sourceEvolve = { start: vi.fn().mockResolvedValue(run) };
    const sourceController = new ChatActionController(
      {
        chat: vi
          .fn()
          .mockResolvedValueOnce(chatResult("Authoring an isolated source candidate.", [{
            kind: "evolve",
            source: "brain",
            toolId: "source.self-modify",
            action: "propose",
            arguments: {
              objective: "record measured maintenance",
              candidateKind: "source",
              sourceEdits: typedEdits
            }
          }]))
          .mockResolvedValueOnce(chatResult("The typed candidate was recorded."))
      },
      tools,
      sourceEvolve
    );
    await sourceController.send("brain-2", "Record the measured maintenance patch");
    expect(sourceEvolve.start).toHaveBeenCalledWith(
      expect.objectContaining({
        brainId: "brain-2",
        objective: "record measured maintenance",
        candidateKind: "source",
        sourceEdits: typedEdits
      })
    );
  });

  it("routes organic idle actions through the same permission and audit events", async () => {
    const organic = {
      kind: "tool" as const,
      source: "organic" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query: "unresolved ternary association" },
      confidence: 0.93
    };
    const service = {
      chat: vi.fn(),
      idleCycle: vi.fn().mockResolvedValue({
        brainId: "brain-organic",
        ran: true,
        actions: [organic]
      })
    };
    const execution: ToolExecutionResult = {
      id: "organic-approval",
      toolId: "web.search",
      action: "search",
      state: "approval-required",
      startedAt: new Date().toISOString(),
      approvalToken: "ask-first"
    };
    const tools = {
      execute: vi.fn().mockResolvedValue(execution),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const streamed: string[] = [];
    controller.on("event", (event) => streamed.push(event.state));

    const result = await controller.idle("brain-organic", 15);

    expect(service.idleCycle).toHaveBeenCalledWith("brain-organic", 15);
    expect(service.chat).not.toHaveBeenCalled();
    expect(tools.execute).toHaveBeenCalledWith({
      brainId: "brain-organic",
      toolId: "web.search",
      action: "search",
      arguments: { query: "unresolved ternary association" }
    });
    expect(result.actionEvents).toEqual([
      expect.objectContaining({
        state: "approval-required",
        action: expect.objectContaining({ source: "organic" }),
        execution
      })
    ]);
    expect(streamed).toEqual(["proposed", "running", "approval-required"]);
  });

  it("cancels an active neural turn and its queued tools by turn id", async () => {
    const service = {
      chat: vi.fn((
        _brainId: string,
        _input: string,
        signal?: AbortSignal
      ): Promise<ChatResult> => new Promise((_resolve, reject) => {
        const abort = (): void => reject(new Error("neural turn cancelled"));
        signal?.addEventListener("abort", abort, { once: true });
        if (signal?.aborted) abort();
      }))
    };
    const tools = {
      execute: vi.fn(),
      cancel: vi.fn(() => 2)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const states: string[] = [];
    controller.on("stream", (event) => {
      if (event.type === "chat-state") states.push(event.state);
    });

    const pending = controller.send(
      "brain-cancel",
      "Continue.",
      undefined,
      "turn-cancel"
    );
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    expect(controller.cancel("brain-cancel", "another-turn")).toBe(0);
    expect(controller.cancel("brain-cancel", "turn-cancel")).toBe(3);

    await expect(pending).rejects.toThrow(/cancelled/i);
    expect(tools.cancel).toHaveBeenCalledWith("brain-cancel");
    expect(states).toEqual(["started", "cancelled"]);
  });
});
