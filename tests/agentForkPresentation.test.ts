import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  AgentForkSubmissionGate,
  agentForkCompletionPresentation
} from "../src/renderer/src/agentForkPresentation";

describe("approved agent-fork UI state", () => {
  it("admits one approval submission until the exact operation settles", () => {
    const gate = new AgentForkSubmissionGate();
    expect(gate.begin()).toBe(true);
    expect(gate.begin()).toBe(false);
    gate.finish();
    expect(gate.begin()).toBe(true);
  });

  it("reports a created fork even when its follow-up subagent turn fails", () => {
    expect(agentForkCompletionPresentation({
      state: "failed",
      error: "worker restarted during the subagent reply"
    }, 1)).toEqual({
      tone: "partial",
      message:
        "1 isolated fork was created, but subagent work did not complete: " +
        "worker restarted during the subagent reply"
    });
  });

  it("disarms the one-use token, polls lineage, and disables repeat approval", () => {
    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    expect(app).toContain('if (approvalToken) setAgentApproval("")');
    expect(app).toContain("!agentSubmissionGate.current.begin()");
    expect(app).toContain("disabled={agentRunning || !objective.trim()}");
    expect(app).toContain("isolated fork${created.length === 1 ? \"\" : \"s\"} created");
    expect(app).toContain("const refreshed = await reloadBranches()");
  });
});
