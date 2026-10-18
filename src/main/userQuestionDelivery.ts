import type { ToolExecutionResult, ToolInvocation } from "../shared/types";
import type { UserQuestionRecord } from "../shared/userQuestions";
import type { UserQuestionService } from "./userQuestions";

export class UserQuestionDelivery {
  private readonly pending = new Map<string, Promise<void>>();
  private readonly unsubscribe: () => void;
  constructor(private readonly questions: UserQuestionService,
    private readonly sink: { queueCompletedToolExecution(invocation: ToolInvocation, execution: ToolExecutionResult, turnId?: string): Promise<void> },
    private readonly onError: (error: unknown) => void) {
    this.unsubscribe = questions.subscribe(event => {
      if (event.type === "resolved" || event.type === "cancelled") {
        void this.deliver(event.question).catch(this.onError);
      }
    });
  }
  async recover(brainId: string): Promise<void> {
    for (const question of await this.questions.undelivered(brainId)) await this.deliver(question);
  }
  deliver(question: UserQuestionRecord): Promise<void> {
    const resolution = question.resolution;
    if (!resolution || question.deliveredAt || question.suspended) return Promise.resolve();
    const key = `${question.brainId}:${resolution.id}`;
    const existing = this.pending.get(key);
    if (existing) return existing;
    const task = (async () => {
      await this.sink.queueCompletedToolExecution({
        brainId: question.brainId, toolId: "studio.question", action: "ask",
        arguments: { prompt: question.prompt, mode: question.mode, options: question.options,
          minSelections: question.minSelections, maxSelections: question.maxSelections,
          allowOther: question.allowOther, requestId: question.requestId }
      }, {
        id: resolution.id, toolId: "studio.question", action: "ask", state: "complete",
        startedAt: question.createdAt, finishedAt: resolution.resolvedAt,
        output: { questionId: question.id, prompt: question.prompt, options: question.options,
          resolution, status: question.status }
      }, question.source?.turnId);
      // The exact evidence is durable before the question is marked delivered.
      await this.questions.acknowledge(question.brainId, question.id, resolution.id);
    })().finally(() => this.pending.delete(key));
    this.pending.set(key, task);
    return task;
  }
  async settle(brainId: string): Promise<void> { await Promise.all([...this.pending.entries()].filter(([key]) => key.startsWith(`${brainId}:`)).map(([, task]) => task)); }
  close(): void { this.unsubscribe(); }
}
