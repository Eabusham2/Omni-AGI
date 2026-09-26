import type { ToolExecutionResult } from "../../shared/types";

export class AgentForkSubmissionGate {
  private running = false;

  begin(): boolean {
    if (this.running) return false;
    this.running = true;
    return true;
  }

  finish(): void {
    this.running = false;
  }
}

export interface AgentForkCompletionPresentation {
  tone: "complete" | "partial" | "failed";
  message: string;
}

export function agentForkCompletionPresentation(
  result: Pick<ToolExecutionResult, "state" | "error">,
  createdForks: number
): AgentForkCompletionPresentation {
  if (result.state === "complete") {
    return {
      tone: "complete",
      message: createdForks > 0
        ? `${createdForks} isolated fork${createdForks === 1 ? "" : "s"} created and subagent work completed.`
        : "Subagent work completed; lineage is up to date."
    };
  }
  if (createdForks > 0) {
    return {
      tone: "partial",
      message:
        `${createdForks} isolated fork${createdForks === 1 ? " was" : "s were"} created, ` +
        `but subagent work did not complete${result.error ? `: ${result.error}` : "."}`
    };
  }
  return {
    tone: "failed",
    message: result.error ?? "Subagent tool failed before creating a fork."
  };
}
