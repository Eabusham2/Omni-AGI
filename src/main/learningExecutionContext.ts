import type { ToolExecutionResult, ToolInvocation } from "../shared/types";
import type { WebLearningPolicy } from "../shared/learning";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";

/** Host-owned correlation. This is never read from generated tool arguments. */
export interface LearningExecutionContext {
  brainId: string;
  taskId: string;
  claimId?: string;
  turnId: string;
  originalTaskId?: string;
  policy(): Promise<WebLearningPolicy>;
  observe(invocation: ToolInvocation, result: ToolExecutionResult): Promise<void>;
}

export interface ToolExecutionContext {
  /** Stable controller event identity, separate from one approval attempt. */
  actionEventId: string;
  /** Controller-owned correlation; never copied from generated arguments. */
  chatTurnId?: string;
  /** Exact native emitter identity supplied by the main action controller. */
  nativeActionId?: string;
  /** Main-owned embedded App identity; never accepted from model operands. */
  mcpAppViewId?: string;
  /** Live main-owned extension lease; never generated operands or persisted grants. */
  originAuthority?: HostActionOriginAuthority;
  learning?: LearningExecutionContext;
}
