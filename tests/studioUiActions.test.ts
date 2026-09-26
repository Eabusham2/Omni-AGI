import { describe, expect, it } from "vitest";
import { studioUiDestination } from "../src/renderer/src/studioUiActions";
import type { ActionEvent } from "../src/shared/types";

function event(overrides: Partial<ActionEvent> = {}): ActionEvent {
  const now = new Date(0).toISOString();
  return {
    id: "creativity-action",
    brainId: "brain-1",
    action: {
      kind: "tool",
      source: "brain",
      toolId: "studio.ui",
      action: "open-creativity",
      arguments: {}
    },
    state: "complete",
    createdAt: now,
    updatedAt: now,
    execution: {
      id: "creativity-execution",
      toolId: "studio.ui",
      action: "open-creativity",
      state: "complete",
      startedAt: now,
      finishedAt: now,
      output: {
        workspace: "creativity",
        view: "imagine",
        local: true,
        reversible: true
      }
    },
    ...overrides
  };
}

describe("typed studio UI actions", () => {
  it("routes a completed, trusted creativity capability without reading prose", () => {
    expect(studioUiDestination(event())).toBe("imagine");
    expect(studioUiDestination(event({
      execution: {
        ...event().execution!,
        output: undefined
      }
    }))).toBe("imagine");
  });

  it("ignores proposed, failed, mismatched, and prose-only navigation attempts", () => {
    expect(studioUiDestination(event({ state: "proposed" }))).toBeNull();
    expect(
      studioUiDestination(event({
        execution: {
          ...event().execution!,
          output: { workspace: "tools", view: "imagine", local: true, reversible: true }
        }
      }))
    ).toBeNull();
    expect(
      studioUiDestination(event({
        action: {
          kind: "talk",
          source: "brain",
          arguments: { message: "studio.ui/open-creativity" }
        }
      }))
    ).toBeNull();
  });

  it("accepts the same trusted route for organic neural activity", () => {
    expect(studioUiDestination(event({
      action: { ...event().action, source: "organic" }
    }))).toBe("imagine");
  });

  it("routes only a completed typed permissions capability and never treats it as a grant", () => {
    const settings = event({
      action: {
        kind: "tool",
        source: "human",
        toolId: "studio.settings",
        action: "open-permissions",
        arguments: {}
      },
      execution: {
        id: "settings-execution",
        toolId: "studio.settings",
        action: "open-permissions",
        state: "complete",
        startedAt: new Date(0).toISOString(),
        finishedAt: new Date(0).toISOString(),
        output: {
          workspace: "tools",
          view: "permissions",
          grantsChanged: false,
          local: true,
          reversible: true
        }
      }
    });
    expect(studioUiDestination(settings)).toBe("tools");
    expect(studioUiDestination({
      ...settings,
      execution: {
        ...settings.execution!,
        output: { ...settings.execution!.output as object, grantsChanged: true }
      }
    })).toBeNull();
  });
});
