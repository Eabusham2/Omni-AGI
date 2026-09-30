import { describe, expect, it, vi } from "vitest";
import {
  normalizeStructuredAction,
  parseModelActions
} from "../src/main/actionProtocol";
import {
  ChatActionController,
  confirmedArgumentTrainingFields
} from "../src/main/chatActionController";
import type { NeuralChatStreamEvent } from "../src/main/brainService";
import type {
  ActionEvent,
  ChatResult,
  ChatStreamEvent,
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
  it("starts an admitted streamed tool during decoding and reuses its identity after commit", async () => {
    let release!: () => void;
    const boundary = new Promise<void>((resolve) => { release = resolve; });
    const action = {
      kind: "tool" as const, source: "brain" as const,
      actionId: "abcdef0123456789abcdef0123456789",
      toolId: "web.search", action: "search", arguments: { query: "typed source" }
    };
    const service = {
      chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined, onStream: ((event: NeuralChatStreamEvent) => void) | undefined) => {
        onStream?.({ type: "chat-action", sequence: 0, actionId: action.actionId, action });
        await boundary;
        return chatResult("native text", [action]);
      }),
      learnStructuredExperience: vi.fn(async () => ({})),
      learnToolRouteOutcome: vi.fn(async () => ({ processed: true, applied: false, duplicate: false, ready: false, steps: 0 }))
    };
    const tools = {
      execute: vi.fn(async () => ({ id: "execution", toolId: action.toolId, action: action.action, state: "complete" as const, startedAt: "now", finishedAt: "now", output: { sources: [] } })),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const pending = controller.send("fixture", "native source", undefined, "fixture-turn");
    await vi.waitFor(() => expect(tools.execute).toHaveBeenCalledOnce());
    expect(service.learnStructuredExperience).not.toHaveBeenCalled();
    expect(service.learnToolRouteOutcome).not.toHaveBeenCalled();
    release();
    const result = await pending;
    expect(tools.execute).toHaveBeenCalledOnce();
    expect(result.actionEvents).toHaveLength(1);
    expect(result.actionEvents?.[0]?.neuralActionId).toBe(action.actionId);
    expect(service.learnToolRouteOutcome).toHaveBeenCalledOnce();
  });

  it("learns complete host payloads without deleting user content or transport credentials into targets", () => {
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "web.search", action: "search",
      arguments: { query: "liquid neural circuits", internal: "not a target" }
    })).toEqual({ typedActionArguments: { query: "liquid neural circuits", internal: "not a target" } });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "system.shell", action: "run",
      arguments: { command: "cat /tmp/private", cwd: "/tmp" }
    })).toEqual({ typedActionArguments: { command: "cat /tmp/private", cwd: "/tmp" } });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "web.search", action: "search",
      arguments: { query: "api_key=sk-secret-credential" }
    })).toEqual({ typedActionArguments: { query: "api_key=sk-secret-credential" } });
    expect(confirmedArgumentTrainingFields({
      kind: "evolve", source: "brain", toolId: "source.self-modify", action: "propose",
      arguments: {
        objective: "Improve retention", candidateKind: "substrate",
        sourceEdits: [{ path: "/tmp/private", content: "do not train" }]
      }
    })).toEqual({ typedActionArguments: {
      objective: "Improve retention", candidateKind: "substrate",
      sourceEdits: [{ path: "/tmp/private", content: "do not train" }]
    } });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "mcp.custom", action: "call",
      arguments: { content: "user-authored content", authorization: "Bearer host-only", api_key: "host-key" }
    })).toEqual({ typedActionArguments: { content: "user-authored content" } });
  });

  it("learns complete browser plans and ordinary typed text while excluding explicitly sensitive entry", () => {
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "browser.automation", action: "task",
      arguments: {
        url: "https://example.com", steps: [{ kind: "click", selector: "#go" }],
        assemblyIds: ["internal"]
      }
    })).toEqual({
      typedActionArguments: {
        url: "https://example.com", steps: [{ kind: "click", selector: "#go" }]
      }
    });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "browser.automation", action: "task",
      arguments: {
        url: "https://example.com",
        steps: [{ kind: "type", selector: "#secret", value: "private text" }]
      }
    })).toEqual({ typedActionArguments: {
      url: "https://example.com", steps: [{ kind: "type", selector: "#secret", value: "private text" }]
    } });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "browser.automation", action: "task",
      arguments: {
        url: "https://example.com",
        steps: [{ kind: "click", selector: "#a" }, { kind: "press", key: "Enter" }]
      }
    })).toEqual({ typedActionArguments: {
      url: "https://example.com", steps: [{ kind: "click", selector: "#a" }, { kind: "press", key: "Enter" }]
    } });
    expect(confirmedArgumentTrainingFields({
      kind: "tool", source: "brain", toolId: "browser.automation", action: "task",
      arguments: { url: "https://example.com", steps: [{ kind: "type", selector: "#credential", value: "host-secret", sensitive: true }] }
    })).toEqual({});
  });

  it("never manufactures an action from human or model prose", () => {
    for (const prose of [
      "Create an image of a cobalt memory palace",
      "Ask three agents to compare the hypotheses",
      "Recursively improve your own source",
      "Open the creativity workspace so we can visualize this story",
      '/tool web.fetch fetch {"url":"https://example.com"}',
      '<omni-tool>{"toolId":"web.search","action":"search"}</omni-tool>'
    ]) {
      expect(parseModelActions(prose)).toEqual([]);
    }
  });

  it("requires complete typed neural actions and ignores response-text tags", () => {
    expect(
      normalizeStructuredAction(
        {
          kind: "agent",
          arguments: { objective: "test both approaches" },
          confidence: 1.7
        },
        "brain"
      )
    ).toBeUndefined();
    expect(
      normalizeStructuredAction(
        {
          kind: "agent",
          toolId: "agent.fork",
          action: "start",
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
          toolId: "modality.imagine",
          action: "generate",
          arguments: { modality: "audio", prompt: "rain" }
        }
      ]
    );
    expect(actions).toHaveLength(1);
    expect(actions.map((action) => action.kind)).toEqual(["imagine"]);
    expect(normalizeStructuredAction({
      kind: "imagine",
      toolId: "agent.fork",
      action: "start",
      arguments: { modality: "image" }
    }, "brain")).toBeUndefined();
    expect(normalizeStructuredAction({
      kind: "tool",
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image" }
    }, "brain")).toBeUndefined();
    expect(
      normalizeStructuredAction(
        {
          kind: "tool",
          toolId: "studio.ui",
          action: "open-creativity",
          arguments: {}
        },
        "brain"
      )
    ).toEqual({
      kind: "tool",
      source: "brain",
      toolId: "studio.ui",
      action: "open-creativity",
      arguments: {},
      confidence: undefined
    });
  });

  it("keeps every distinct byte-bounded neural action instead of truncating after eight", () => {
    const proposed = Array.from({ length: 12 }, (_, index) => ({
      kind: "tool",
      toolId: "web.search",
      action: "search",
      arguments: { query: `independent evidence ${index}` }
    }));

    const actions = parseModelActions("", proposed);

    expect(actions).toHaveLength(12);
    expect(actions[11]).toMatchObject({
      toolId: "web.search",
      action: "search",
      arguments: { query: "independent evidence 11" }
    });
  });

  it("learns a completed visible action without manufacturing a hidden chat turn", async () => {
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
      chat: vi.fn(async (
        _brainId: string,
        input: string,
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
          actionId: "11111111111111111111111111111111",
          action: proposed
        });
        const initial = chatResult("I will form that image.");
        initial.humanMessage = { ...initial.humanMessage, content: input };
        return initial;
      }),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
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
    const phases: ChatStreamEvent[] = [];
    const globalEvents: string[] = [];
    controller.on("event", (event) => globalEvents.push(event.state));
    controller.on("stream", (event) => {
      if (event.type === "chat-action") streamed.push(event.actionEvent.state);
      if (event.type === "chat-phase") phases.push(event);
    });

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
      expect.any(Function),
      expect.any(String)
    );
    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.learnStructuredExperience).toHaveBeenCalledOnce();
    const visibleExperience = service.learnStructuredExperience.mock.calls[0]?.[1].content;
    expect(visibleExperience).toMatch(/^\[Visible structured action result\]/);
    expect(visibleExperience).toContain("kind: imagine");
    expect(visibleExperience).toContain("tool: modality.imagine");
    expect(visibleExperience).toContain("action: generate");
    expect(visibleExperience).toContain("requested-by: brain");
    expect(visibleExperience).toContain('"artifactPath": "artifact.png"');
    expect(visibleExperience).not.toMatch(/(?:role|prompt):\s*system\b/i);
    expect(visibleExperience).not.toMatch(/persona|refusal/i);
    expect(result.humanMessage).toMatchObject({
      role: "human",
      content: "Generate an image of an emergent city"
    });
    expect(result.brainMessage.content).toBe("I will form that image.");
    expect(phases).toContainEqual(expect.objectContaining({
      phase: "action-result-learning",
      replyComplete: true,
      turnCommitted: true,
      learning: true,
      saving: true
    }));
    expect(result.actionEvents).toEqual([
      expect.objectContaining({ state: "complete", execution })
    ]);
    expect(streamed).toEqual(["proposed", "running", "complete"]);
    expect(globalEvents).toEqual([]);
  });

  it("resolves an Ask-approved action in main without a synthetic chat continuation", async () => {
    const proposed = {
      kind: "tool" as const,
      source: "brain" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query: "approval-bound evidence" }
    };
    const initial = chatResult("I need your approval before searching.", [proposed]);
    initial.humanMessage = {
      ...initial.humanMessage,
      content: "Search for approval-bound evidence"
    };
    const service = {
      chat: vi.fn().mockResolvedValue(initial),
      learnStructuredExperience: vi.fn().mockResolvedValue({}),
      learnToolRouteOutcome: vi.fn().mockResolvedValue({
        processed: true,
        applied: true,
        duplicate: false,
        ready: true,
        steps: 1
      }),
      recordConversationActions: vi.fn().mockResolvedValue(undefined)
    };
    const approvalToken = "approval-token-1";
    const tools = {
      execute: vi.fn()
        .mockResolvedValueOnce({
          id: "approval-challenge",
          toolId: "web.search",
          action: "search",
          state: "approval-required",
          startedAt: new Date().toISOString(),
          approvalToken,
          approvalExpiresAt: new Date(Date.now() + 30_000).toISOString()
        } satisfies ToolExecutionResult)
        .mockResolvedValueOnce({
          id: "approved-execution",
          toolId: "web.search",
          action: "search",
          state: "complete",
          startedAt: new Date().toISOString(),
          finishedAt: new Date().toISOString(),
          output: { results: ["bound result"] }
        } satisfies ToolExecutionResult),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const turnStream: ChatStreamEvent[] = [];
    const globalEvents: ActionEvent[] = [];
    controller.on("stream", (event) => turnStream.push(event));
    controller.on("event", (event) => globalEvents.push(event));

    const chat = await controller.send(
      "brain-approved-action",
      "Search for approval-bound evidence"
    );
    const pending = chat.actionEvents?.[0];
    expect(pending).toMatchObject({
      state: "approval-required",
      action: proposed,
      execution: { approvalToken }
    });
    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.learnStructuredExperience).not.toHaveBeenCalled();
    const streamCountBeforeApproval = turnStream.length;

    const approved = await controller.approveAction({
      brainId: "brain-approved-action",
      actionEventId: pending!.id,
      approvalToken
    });

    expect(tools.execute).toHaveBeenCalledTimes(2);
    expect(tools.execute.mock.calls[1]?.[0]).toEqual({
      brainId: "brain-approved-action",
      toolId: "web.search",
      action: "search",
      arguments: { query: "approval-bound evidence" },
      approvalToken
    });
    expect(tools.execute.mock.calls[1]?.[2]).toBe(pending!.id);
    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.learnStructuredExperience).toHaveBeenCalledOnce();
    expect(service.learnToolRouteOutcome).toHaveBeenCalledOnce();
    expect(service.learnToolRouteOutcome).toHaveBeenCalledWith(
      "brain-approved-action",
      {
        eventId: pending!.id,
        utterance: "Search for approval-bound evidence",
        toolId: "web.search",
        action: "search",
        arguments: { typedActionArguments: { query: "approval-bound evidence" } }
      },
      undefined
    );
    expect(service.learnStructuredExperience.mock.calls[0]?.[1].content).toContain(
      '"bound result"'
    );
    expect(approved).toMatchObject({
      learned: true,
      actionEvent: {
        id: pending!.id,
        state: "complete",
        statusLabel: "Action complete · result learned"
      }
    });
    expect(turnStream).toHaveLength(streamCountBeforeApproval);
    expect(globalEvents.map((event) => event.state)).toEqual([
      "running",
      "complete",
      "complete"
    ]);
    expect(service.recordConversationActions).toHaveBeenCalledTimes(2);
    await expect(controller.approveAction({
      brainId: "brain-approved-action",
      actionEventId: pending!.id,
      approvalToken
    })).rejects.toThrow(/unavailable|expired/i);
  });

  it("does not recursively ask the action policy again after imagination completes", async () => {
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
      chat: vi.fn().mockResolvedValue(
        chatResult("I formed the scene.", [recurring("first")])
      ),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
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

    expect(service.chat).toHaveBeenCalledOnce();
    expect(tools.execute).toHaveBeenCalledTimes(1);
    expect(service.learnStructuredExperience).toHaveBeenCalledOnce();
    expect(result.actionEvents).toHaveLength(1);
    expect(result.brainMessage.content).toBe("I formed the scene.");
  });

  it("learns each completed action once while keeping one visible chat pair", async () => {
    const actions = ["first evidence", "second evidence"].map((query) => ({
      kind: "tool" as const,
      source: "brain" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query }
    }));
    const initial = chatResult("I gathered both visible results.", actions);
    initial.humanMessage = {
      ...initial.humanMessage,
      content: "Research both questions."
    };
    const service = {
      chat: vi.fn().mockResolvedValue(initial),
      learnStructuredExperience: vi.fn().mockResolvedValue({}),
      learnToolRouteOutcome: vi.fn().mockResolvedValue({
        processed: true,
        applied: true,
        duplicate: false,
        ready: true,
        steps: 1
      })
    };
    const tools = {
      execute: vi.fn(async (invocation: {
        arguments: Record<string, unknown>;
      }) => ({
        id: `execution-${String(invocation.arguments.query)}`,
        toolId: "web.search",
        action: "search",
        state: "complete" as const,
        startedAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
        output: { result: String(invocation.arguments.query) }
      })),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });

    const result = await controller.send(
      "brain-two-actions",
      "Research both questions."
    );

    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.chat).toHaveBeenCalledWith(
      "brain-two-actions",
      "Research both questions.",
      expect.any(AbortSignal),
      expect.any(Function),
      expect.any(String)
    );
    expect(tools.execute).toHaveBeenCalledTimes(2);
    expect(service.learnStructuredExperience).toHaveBeenCalledTimes(2);
    expect(service.learnToolRouteOutcome).toHaveBeenCalledTimes(2);
    expect(service.learnToolRouteOutcome.mock.calls.map((call) => call[1])).toEqual([
      expect.objectContaining({
        utterance: "Research both questions.",
        toolId: "web.search",
        action: "search"
      }),
      expect.objectContaining({
        utterance: "Research both questions.",
        toolId: "web.search",
        action: "search"
      })
    ]);
    expect(service.learnStructuredExperience.mock.calls.map((call) => call[1].content))
      .toEqual([
        expect.stringContaining("first evidence"),
        expect.stringContaining("second evidence")
      ]);
    expect(result.humanMessage.content).toBe("Research both questions.");
    expect(result.brainMessage.content).toBe("I gathered both visible results.");
    expect(result.actionEvents).toHaveLength(2);
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
      }),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
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
    const globalEvents: unknown[] = [];
    controller.on("event", (event) => globalEvents.push(event));
    const turnStream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => {
      turnStream.push(event);
      if (event.type === "chat-action") {
        streamed.push({
          state: event.actionEvent.state,
          progress: event.actionEvent.progress,
          label: event.actionEvent.statusLabel
        });
      }
    });

    const pending = controller.send(
      "brain-story",
      "Continue the story.",
      undefined,
      "turn-story"
    );
    await vi.waitFor(() =>
      expect(turnStream.some((event) => event.type === "modality-preview")).toBe(true)
    );
    expect(calls).toBe(1);
    expect(tools.execute).not.toHaveBeenCalled();
    expect(turnStream.slice(0, 3).map((event) => event.type)).toEqual([
      "chat-state",
      "chat-token",
      "chat-action"
    ]);
    finishInitial(chatResult("The scene became motion in my workspace."));
    await vi.waitFor(() => expect(tools.execute).toHaveBeenCalledOnce());
    const result = await pending;

    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.learnStructuredExperience).toHaveBeenCalledOnce();
    expect(streamed).toEqual([
      { state: "proposed", progress: undefined, label: undefined },
      { state: "running", progress: 0.08, label: "A first temporal sketch" },
      { state: "running", progress: 0.08, label: "A first temporal sketch" },
      { state: "running", progress: 0.18, label: "Forming temporal latents" },
      { state: "running", progress: 0.72, label: "Decoding frames and sound" },
      { state: "running", progress: 0.9, label: "Refreshing the live preview" },
      { state: "complete", progress: 0.9, label: "Refreshing the live preview" }
    ]);
    expect(globalEvents).toEqual([]);
    const serializedTurnStream = JSON.stringify(turnStream);
    expect(serializedTurnStream.match(/c2Vjb25k/g)).toHaveLength(1);
    expect(serializedTurnStream.match(/Zmlyc3Q=/g)).toHaveLength(1);
    for (const event of turnStream) {
      if (event.type !== "chat-action") continue;
      expect(event.actionEvent.preview).toBeUndefined();
      expect(event.actionEvent.execution?.output).toBeUndefined();
    }
    expect(result.actionEvents?.[0]).toMatchObject({
      state: "complete",
      runtimeJobId: "video-job",
      progress: 0.9,
      preview: { revision: 1 },
      execution
    });
  });

  it("finalizes a streamed inline imagination when the worker turn fails", async () => {
    const proposed = {
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image" }
    };
    const service = {
      chat: vi.fn(async (
        _brainId: string,
        _input: string,
        _signal?: AbortSignal,
        onStream?: (event: NeuralChatStreamEvent) => void
      ): Promise<ChatResult> => {
        onStream?.({
          type: "chat-action",
          sequence: 0,
          actionId: "33333333333333333333333333333333",
          action: proposed
        });
        onStream?.({
          type: "modality-preview",
          sequence: 1,
          actionId: "33333333333333333333333333333333",
          preview: {
            revision: 0,
            progress: 0.2,
            statusLabel: "First CPU-isolated decoder revision",
            mimeType: "image/png",
            dataUrl: "data:image/png;base64,cHJldmlldw=="
          }
        });
        throw new Error("neural worker exited during inline imagination");
      })
    };
    const tools = {
      execute: vi.fn(),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const states: string[] = [];
    controller.on("stream", (event: ChatStreamEvent) => {
      if (event.type === "chat-action") states.push(event.actionEvent.state);
    });

    await expect(controller.send("brain-mps", "Recall this file.")).rejects.toThrow(
      /worker exited/i
    );
    expect(states).toEqual(["proposed", "running", "failed"]);
    expect(tools.execute).not.toHaveBeenCalled();
  });

  it("commits the original turn before tool audit and learns its result outside chat history", async () => {
    const proposed = {
      kind: "imagine" as const,
      source: "brain" as const,
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image", conceptIds: ["persisted-scene"] }
    };
    let releaseInitial!: () => void;
    const initialGate = new Promise<void>((resolve) => {
      releaseInitial = resolve;
    });
    let markInitialCommitted!: () => void;
    const initialCommitted = new Promise<void>((resolve) => {
      markInitialCommitted = resolve;
    });
    let persistedMessages: string[] = [];
    const learnedExperiences: string[] = [];
    let calls = 0;
    const service = {
      chat: vi.fn(async (
        _brainId: string,
        input: string,
        _signal?: AbortSignal,
        onStream?: (event: {
          type: "chat-action";
          sequence: number;
          actionId: string;
          action: typeof proposed;
        }) => void
      ) => {
        calls += 1;
        if (calls === 1) {
          const documentBeforeChat = [...persistedMessages];
          onStream?.({
            type: "chat-action",
            sequence: 0,
            actionId: "33333333333333333333333333333333",
            action: proposed
          });
          await initialGate;
          persistedMessages = [...documentBeforeChat, input];
          markInitialCommitted();
          return chatResult("I committed the scene before its tool audit.");
        }
        throw new Error(`Unexpected recursive chat input: ${input}`);
      }),
      learnStructuredExperience: vi.fn(async (
        _brainId: string,
        experience: { content: string }
      ) => {
        learnedExperiences.push(experience.content);
        return {};
      })
    };
    const execution: ToolExecutionResult = {
      id: "audited-image",
      toolId: "modality.imagine",
      action: "generate",
      state: "complete",
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      output: { artifactPath: "persisted-scene.png" }
    };
    const tools = {
      execute: vi.fn(async () => {
        // Model ToolExecutor.audit's former read/modify/write race: an eager
        // execution captures a stale document, waits for chat commit, and can
        // then overwrite the original turn with only its audit entry.
        const documentBeforeAudit = [...persistedMessages];
        await initialCommitted;
        persistedMessages = [...documentBeforeAudit, "tool-audit"];
        return execution;
      }),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const states: string[] = [];
    controller.on("stream", (event) => {
      if (event.type === "chat-action") states.push(event.actionEvent.state);
    });

    const pending = controller.send(
      "brain-persistence",
      "make an image from this internal scene"
    );
    await vi.waitFor(() => expect(states).toContain("proposed"));
    expect(tools.execute).not.toHaveBeenCalled();

    releaseInitial();
    const result = await pending;

    expect(tools.execute).toHaveBeenCalledOnce();
    expect(persistedMessages[0]).toBe(
      "make an image from this internal scene"
    );
    expect(persistedMessages).toContain("tool-audit");
    expect(persistedMessages).toEqual([
      "make an image from this internal scene",
      "tool-audit"
    ]);
    expect(service.chat).toHaveBeenCalledOnce();
    expect(learnedExperiences).toHaveLength(1);
    expect(learnedExperiences[0]).toContain("[Visible structured action result]");
    expect(result.actionEvents).toEqual([
      expect.objectContaining({ state: "complete", execution })
    ]);
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
      chat: vi.fn().mockResolvedValue(chatResult("I heard it instead.", [proposed])),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
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
      expect.any(Function),
      expect.any(String)
    );
    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.learnStructuredExperience).toHaveBeenCalledOnce();
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

  it("consumes a pre-speech foundation Ponder exactly once", async () => {
    const trace = {
      activated: true,
      activated_by: "on-demand-action",
      passes: 4,
      converged: true,
      stop_reason: "converged"
    };
    const ponder = {
      kind: "ponder" as const,
      source: "brain" as const,
      arguments: {
        assemblyIds: ["active-assembly"],
        organic: false,
        completedInTurn: true,
        ponderTrace: trace
      },
      confidence: 0.88
    };
    const service = {
      chat: vi.fn().mockResolvedValue(
        chatResult("The pre-speech refinement changed this answer.", [ponder])
      ),
      idleCycle: vi.fn()
    };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );

    const result = await controller.send("brain-pre-speech", "Ponder this relation.");

    expect(service.chat).toHaveBeenCalledOnce();
    expect(service.idleCycle).not.toHaveBeenCalled();
    expect(result.actionEvents).toEqual([
      expect.objectContaining({
        state: "complete",
        action: expect.objectContaining({
          kind: "ponder",
          arguments: expect.objectContaining({
            completedInTurn: true,
            ponderTrace: trace
          })
        })
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
      chat: vi.fn().mockResolvedValue(chatResult("Starting an experiment.", [{
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
        }])),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
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
    expect(evolveService.chat).toHaveBeenCalledOnce();
    expect(evolveService.learnStructuredExperience).toHaveBeenCalledOnce();
    expect(evolved.brainMessage.content).toBe("Starting an experiment.");
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
    const sourceService = {
      chat: vi.fn().mockResolvedValue(
        chatResult("Authoring an isolated source candidate.", [{
          kind: "evolve",
          source: "brain",
          toolId: "source.self-modify",
          action: "propose",
          arguments: {
            objective: "record measured maintenance",
            candidateKind: "neural",
            sourceEdits: typedEdits
          }
        }])
      ),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
    };
    const sourceController = new ChatActionController(
      sourceService,
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
    expect(sourceService.chat).toHaveBeenCalledOnce();
    expect(sourceService.learnStructuredExperience).toHaveBeenCalledOnce();
  });

  it("never silently rewrites a neural evolution candidate into substrate", async () => {
    const exercise = async (
      argumentsValue: Record<string, unknown>,
      source: "brain" | "organic" | "human" = "brain"
    ) => {
      const service = {
        chat: vi.fn().mockResolvedValue(chatResult("Evaluating an idea.", [{
          kind: "evolve",
          source,
          toolId: "source.self-modify",
          action: "propose",
          arguments: argumentsValue
        }]))
      };
      const tools = { execute: vi.fn(), cancel: vi.fn() };
      const evolution = { start: vi.fn().mockResolvedValue({
        id: "candidate-run", state: "experimenting"
      }) };
      const result = await new ChatActionController(service, tools, evolution)
        .send("brain-evolution-gate", "Improve retention");
      return { result, evolution };
    };

    for (const [argumentsValue, expectedError] of [
      [{ objective: "Improve retention" }, "selected candidate kind"],
      [{ objective: "Improve retention", candidateKind: "source" }, "exact typed source edits"],
      [{ objective: "Improve retention", candidateKind: "data" }, "source evidence"],
      [{ objective: "Improve retention", candidateKind: "architecture" }, "expert-growth amount"]
    ] as const) {
      const { result, evolution } = await exercise(argumentsValue);
      expect(evolution.start).not.toHaveBeenCalled();
      expect(result.actionEvents?.[0]).toMatchObject({
        state: "failed",
        error: expect.stringContaining(expectedError)
      });
    }

    const organic = await exercise({
      objective: "Improve the active concept links",
      candidateKind: "neural",
      latentReplay: true
    }, "organic");
    expect(organic.evolution.start).toHaveBeenCalledWith(
      expect.objectContaining({ candidateKind: "neural" })
    );

    const manual = await exercise({ objective: "Review a local experiment" }, "human");
    expect(manual.evolution.start).toHaveBeenCalledWith(
      expect.objectContaining({ candidateKind: "substrate" })
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

  it("preserves every distinct neural action selected during an idle cycle", async () => {
    const actions = Array.from({ length: 4 }, (_, index) => ({
      kind: "tool" as const,
      source: "organic" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query: `organic uncertainty ${index}` }
    }));
    const service = {
      chat: vi.fn(),
      idleCycle: vi.fn().mockResolvedValue({
        brainId: "brain-organic-many",
        ran: true,
        actions
      }),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
    };
    const tools = {
      execute: vi.fn(async (invocation: { toolId: string; action: string }) => ({
        id: `execution-${tools.execute.mock.calls.length}`,
        toolId: invocation.toolId,
        action: invocation.action,
        state: "complete" as const,
        startedAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
        output: { evidence: true }
      })),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });

    const result = await controller.idle("brain-organic-many", 0);

    expect(result.actionEvents).toHaveLength(4);
    expect(tools.execute).toHaveBeenCalledTimes(4);
    expect(service.learnStructuredExperience).toHaveBeenCalledTimes(4);
  });

  it("learns completed organic research evidence without turning it into a hidden prompt", async () => {
    const service = {
      chat: vi.fn(),
      idleCycle: vi.fn().mockResolvedValue({
        brainId: "brain-organic-learning",
        ran: true,
        actions: [{
          kind: "tool" as const,
          source: "organic" as const,
          toolId: "web.search",
          action: "search",
          arguments: { query: "active unresolved assembly", organic: true }
        }]
      }),
      learnStructuredExperience: vi.fn().mockResolvedValue({})
    };
    const tools = {
      execute: vi.fn().mockResolvedValue({
        id: "organic-search-complete",
        toolId: "web.search",
        action: "search",
        state: "complete",
        startedAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
        output: { results: [{ title: "Evidence", url: "https://example.org" }] }
      } satisfies ToolExecutionResult),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });

    const result = await controller.idle("brain-organic-learning", 0);

    expect(result.actionEvents?.[0]?.state).toBe("complete");
    expect(service.chat).not.toHaveBeenCalled();
    expect(service.learnStructuredExperience).toHaveBeenCalledWith(
      "brain-organic-learning",
      expect.objectContaining({
        sourceLabel: "organic visible action evidence",
        content: expect.stringContaining("https://example.org")
      })
    );
  });

  it("executes an organic creativity workspace capability as a visible action", async () => {
    const organic = {
      kind: "tool" as const,
      source: "organic" as const,
      toolId: "studio.ui",
      action: "open-creativity",
      arguments: { assemblyIds: ["story-scene"], organic: true },
      confidence: 0.88
    };
    const service = {
      chat: vi.fn(),
      idleCycle: vi.fn().mockResolvedValue({
        brainId: "brain-organic-creativity",
        ran: true,
        actions: [organic]
      })
    };
    const execution: ToolExecutionResult = {
      id: "organic-creativity",
      toolId: "studio.ui",
      action: "open-creativity",
      state: "complete",
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      output: {
        workspace: "creativity",
        view: "imagine",
        local: true,
        reversible: true
      }
    };
    const tools = {
      execute: vi.fn().mockResolvedValue(execution),
      cancel: vi.fn(() => 0)
    };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const events: Array<{ state: string; toolId?: string }> = [];
    controller.on("event", (event) => events.push({
      state: event.state,
      toolId: event.action.toolId
    }));

    const result = await controller.idle("brain-organic-creativity", 0);

    expect(tools.execute).toHaveBeenCalledWith({
      brainId: "brain-organic-creativity",
      toolId: "studio.ui",
      action: "open-creativity",
      arguments: { assemblyIds: ["story-scene"], organic: true }
    });
    expect(result.actionEvents).toEqual([
      expect.objectContaining({
        state: "complete",
        action: expect.objectContaining({
          source: "organic",
          toolId: "studio.ui",
          action: "open-creativity"
        }),
        execution
      })
    ]);
    expect(events).toEqual([
      { state: "proposed", toolId: "studio.ui" },
      { state: "running", toolId: "studio.ui" },
      { state: "complete", toolId: "studio.ui" }
    ]);
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
    const cancellations: Array<{ brainId: string; turnId?: string }> = [];
    controller.on("stream", (event) => {
      if (event.type === "chat-state") states.push(event.state);
    });
    controller.on("neural-cancelled", (event) => cancellations.push(event));

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
    expect(tools.cancel).toHaveBeenCalledWith("brain-cancel", "turn-cancel");
    expect(states).toEqual(["started", "cancelled"]);
    expect(cancellations).toEqual([
      { brainId: "brain-cancel", turnId: "turn-cancel" }
    ]);
  });

  it("reports a queued chat cancellation without cancelling its evolution blocker", async () => {
    const service = {
      chat: vi.fn((
        _brainId: string,
        _input: string,
        signal?: AbortSignal,
        onStream?: (event: NeuralChatStreamEvent) => void
      ): Promise<ChatResult> => {
        onStream?.({
          type: "runtime-activity",
          state: "queued",
          queue: {
            position: 1,
            queuedBehind: {
              requestId: "evolution-run",
              owner: "evolution",
              label: "Neural evolution",
              method: "evolution.propose",
              brainId: "brain-queue",
              jobId: "evolution-run"
            }
          }
        });
        return new Promise((_resolve, reject) => {
          const abort = (): void => reject(new Error("queued chat cancelled"));
          signal?.addEventListener("abort", abort, { once: true });
          if (signal?.aborted) abort();
        });
      })
    };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(
      service,
      tools,
      { start: vi.fn() }
    );
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => stream.push(event));
    const turn = controller.send(
      "brain-queue",
      "Wait behind the experiment.",
      undefined,
      "queued-turn"
    );

    await vi.waitFor(() => expect(stream).toContainEqual(
      expect.objectContaining({
        type: "chat-state",
        state: "queued",
        queue: expect.objectContaining({
          queuedBehind: expect.objectContaining({ owner: "evolution" })
        })
      })
    ));
    await expect(
      controller.cancelAndWait("brain-queue", "queued-turn")
    ).resolves.toBe(1);
    await expect(turn).rejects.toThrow(/cancelled/i);

    expect(tools.cancel).toHaveBeenCalledWith("brain-queue", "queued-turn");
    expect(tools.cancel).not.toHaveBeenCalledWith("brain-queue", "evolution-run");
    expect(stream.at(-1)).toMatchObject({
      type: "chat-state",
      state: "cancelled",
      cancellation: {
        phase: "queued",
        unrelatedActivityContinues: true,
        workerTerminationAcknowledged: false
      }
    });
  });

  it("publishes reply-complete learning as provisional until the terminal commit", async () => {
    let finishTurn!: (result: ChatResult) => void;
    const unfinishedTurn = new Promise<ChatResult>((resolve) => {
      finishTurn = resolve;
    });
    let emitStream: ((event: NeuralChatStreamEvent) => void) | undefined;
    const service = {
      chat: vi.fn((
        _brainId: string,
        _input: string,
        _signal?: AbortSignal,
        onStream?: (event: NeuralChatStreamEvent) => void
      ) => {
        emitStream = onStream;
        return unfinishedTurn;
      })
    };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => stream.push(event));

    const pending = controller.send(
      "brain-phase",
      "Keep the reply visible while it learns.",
      undefined,
      "turn-phase"
    );
    expect(emitStream).toBeTypeOf("function");
    emitStream?.({
      type: "chat-phase",
      sequence: 4,
      phase: "reply-complete-learning",
      replyComplete: true,
      turnCommitted: false,
      learning: true,
      saving: true
    });

    expect(stream).toHaveLength(2);
    expect(stream[0]).toMatchObject({ type: "chat-state", state: "started" });
    expect(stream[1]).toMatchObject({
      type: "chat-phase",
      brainId: "brain-phase",
      turnId: "turn-phase",
      sequence: 1,
      phase: "reply-complete-learning",
      replyComplete: true,
      turnCommitted: false,
      learning: true,
      saving: true
    });
    expect(stream.some(
      (event) => event.type === "chat-state" && event.state === "complete"
    )).toBe(false);

    finishTurn(chatResult("The reply remained visible."));
    await pending;
    expect(stream.at(-1)).toMatchObject({
      type: "chat-state",
      state: "complete",
      sequence: 3
    });
  });

  it("carries steering as typed correlation metadata without changing model input", async () => {
    const service = { chat: vi.fn().mockResolvedValue(chatResult("replacement")) };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );
    const metadata = {
      kind: "steer" as const,
      replacesTurnId: "turn-before",
      source: "human" as const,
      createdAt: "2026-08-10T21:00:00.000Z"
    };
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => stream.push(event));

    const response = await controller.send(
      "brain-steer",
      "Focus only on the memory evidence.",
      undefined,
      "turn-replacement",
      metadata
    );

    expect(service.chat.mock.calls[0]?.[1]).toBe(
      "Focus only on the memory evidence."
    );
    expect(service.chat.mock.calls[0]?.[1]).not.toContain("turn-before");
    expect(response.turnMetadata).toEqual(metadata);
    expect(stream[0]).toMatchObject({
      type: "chat-state",
      state: "started",
      turnMetadata: metadata
    });
  });

  it("hands an active Steer turn to the same warm runtime after its predecessor settles", async () => {
    let finishOriginal!: (result: ChatResult) => void;
    const originalResult = new Promise<ChatResult>((resolve) => {
      finishOriginal = resolve;
    });
    let originalSignal: AbortSignal | undefined;
    const service = {
      chat: vi.fn((
        _brainId: string,
        input: string,
        signal?: AbortSignal
      ): Promise<ChatResult> => {
        if (input === "Begin broadly.") {
          originalSignal = signal;
          return originalResult;
        }
        return Promise.resolve(chatResult("warm handoff complete"));
      })
    };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(
      service,
      tools,
      { start: vi.fn() }
    );
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => stream.push(event));

    const original = controller.send(
      "brain-warm-steer",
      "Begin broadly.",
      undefined,
      "turn-original"
    );
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());

    const metadata = {
      kind: "steer" as const,
      replacesTurnId: "turn-original",
      source: "human" as const,
      createdAt: "2026-09-19T12:00:00.000Z"
    };
    const steered = controller.send(
      "brain-warm-steer",
      "Focus on the saved evidence.",
      undefined,
      "turn-steered",
      metadata
    );

    await vi.waitFor(() => expect(stream).toContainEqual(
      expect.objectContaining({
        type: "chat-state",
        turnId: "turn-steered",
        state: "queued",
        turnMetadata: metadata,
        queue: {
          position: 1,
          queuedBehind: expect.objectContaining({
            requestId: "turn-original",
            owner: "chat",
            turnId: "turn-original"
          })
        }
      })
    ));
    expect(service.chat).toHaveBeenCalledOnce();
    expect(originalSignal?.aborted).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();

    finishOriginal(chatResult("original committed"));
    await original;
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(2));
    await expect(steered).resolves.toMatchObject({ turnMetadata: metadata });

    expect(originalSignal?.aborted).toBe(false);
    expect(service.chat.mock.calls.map((call) => call[1])).toEqual([
      "Begin broadly.",
      "Focus on the saved evidence."
    ]);
    expect(stream.flatMap((event) =>
      event.type === "chat-state" && event.turnId === "turn-steered"
        ? [event.state]
        : []
    )).toEqual(["queued", "started", "complete"]);
  });

  it("cancels a queued warm Steer handoff without aborting the active predecessor", async () => {
    let finishOriginal!: (result: ChatResult) => void;
    const originalResult = new Promise<ChatResult>((resolve) => {
      finishOriginal = resolve;
    });
    let originalSignal: AbortSignal | undefined;
    const service = {
      chat: vi.fn((
        _brainId: string,
        _input: string,
        signal?: AbortSignal
      ): Promise<ChatResult> => {
        originalSignal = signal;
        return originalResult;
      })
    };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(
      service,
      tools,
      { start: vi.fn() }
    );
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event) => stream.push(event));

    const original = controller.send(
      "brain-cancel-warm-steer",
      "Keep running.",
      undefined,
      "turn-active"
    );
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    const steered = controller.send(
      "brain-cancel-warm-steer",
      "Do this afterward.",
      undefined,
      "turn-queued-steer",
      {
        kind: "steer",
        replacesTurnId: "turn-active",
        source: "human",
        createdAt: "2026-09-19T12:01:00.000Z"
      }
    );
    await vi.waitFor(() => expect(stream).toContainEqual(
      expect.objectContaining({
        type: "chat-state",
        turnId: "turn-queued-steer",
        state: "queued"
      })
    ));

    await expect(
      controller.cancelAndWait("brain-cancel-warm-steer", "turn-queued-steer")
    ).resolves.toBe(1);
    await expect(steered).rejects.toMatchObject({ name: "AbortError" });
    expect(originalSignal?.aborted).toBe(false);
    expect(service.chat).toHaveBeenCalledOnce();
    expect(stream.at(-1)).toMatchObject({
      type: "chat-state",
      turnId: "turn-queued-steer",
      state: "cancelled",
      cancellation: {
        phase: "queued",
        unrelatedActivityContinues: true,
        workerTerminationAcknowledged: false
      }
    });

    finishOriginal(chatResult("predecessor complete"));
    await original;
  });

  it("honors exact cancellation that arrives before a delayed turn registers", async () => {
    const service = { chat: vi.fn().mockResolvedValue(chatResult("must not run")) };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );

    expect(controller.cancel("brain-race", "turn-not-started")).toBe(0);
    await expect(
      controller.send(
        "brain-race",
        "This delayed turn must remain cancelled.",
        undefined,
        "turn-not-started"
      )
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(service.chat).not.toHaveBeenCalled();
  });
});
