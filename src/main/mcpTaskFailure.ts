import type { McpTaskReceipt } from "../shared/mcpTasks";
import { checkedMcpTaskRecord } from "./mcpTaskProtocol";

/** Main-only failure with the actual committed custody. Remote JSON is never an instance of this class. */
export class McpTaskCustodyError extends Error {
  readonly receipt: McpTaskReceipt;
  constructor(message: string, receipt: McpTaskReceipt) {
    super(message); this.name = "McpTaskCustodyError";
    this.receipt = { ...structuredClone(receipt), record: checkedMcpTaskRecord(receipt.record) };
  }
}

export function mcpTaskReceiptFromFailure(error: unknown, brainId: string, toolId: string): McpTaskReceipt | undefined {
  if (!(error instanceof McpTaskCustodyError) || error.receipt.record.source.brainId !== brainId || error.receipt.record.source.toolId !== toolId) return undefined;
  return structuredClone(error.receipt);
}
