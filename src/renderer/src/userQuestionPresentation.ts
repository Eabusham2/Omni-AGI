import type { UserQuestionRecord, UserQuestionResponse } from "../../shared/userQuestions";

export interface UserQuestionDraft {
  optionIds: string[];
  text: string;
  other: boolean;
}

export function mergeUserQuestionRecords(
  saved: readonly UserQuestionRecord[],
  updates: readonly UserQuestionRecord[]
): UserQuestionRecord[] {
  const records = new Map(saved.map((record) => [record.id, record]));
  for (const record of updates) {
    const current = records.get(record.id);
    if (current) {
      const ordering = current.updatedAt.localeCompare(record.updatedAt);
      if (ordering > 0) continue;
      if (!ordering && current.status !== "pending" && record.status === "pending") continue;
      if (!ordering && current.resolution?.id === record.resolution?.id && current.deliveredAt && !record.deliveredAt) continue;
    }
    records.set(record.id, record);
  }
  return [...records.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
}

export function userQuestionDraftError(question: UserQuestionRecord, draft: UserQuestionDraft): string | undefined {
  if (question.status !== "pending") return "This question has already been resolved.";
  if (question.suspended) return "Resume this question before answering.";
  if (question.mode === "text" || draft.other) {
    if (draft.other && question.mode !== "text" && !question.allowOther) return "Choose one of the listed options.";
    return draft.text.trim() ? undefined : "Enter your answer.";
  }
  const selected = new Set(draft.optionIds);
  if (selected.size !== draft.optionIds.length || draft.optionIds.some((id) => !question.options.some((option) => option.id === id))) return "Choose from the listed options.";
  if (selected.size < question.minSelections) return `Choose at least ${question.minSelections} ${question.minSelections === 1 ? "option" : "options"}.`;
  if (selected.size > question.maxSelections) return `Choose no more than ${question.maxSelections} ${question.maxSelections === 1 ? "option" : "options"}.`;
  return undefined;
}

export function userQuestionDraftResponse(question: UserQuestionRecord, draft: UserQuestionDraft): UserQuestionResponse {
  const error = userQuestionDraftError(question, draft);
  if (error) throw new Error(error);
  return question.mode === "text" || draft.other
    ? { text: draft.text }
    : { optionIds: question.options.filter((option) => draft.optionIds.includes(option.id)).map((option) => option.id) };
}

export function userQuestionResolutionLabel(question: UserQuestionRecord): string {
  if (question.status === "assumed") return "Assumed after timeout";
  if (question.status === "answered") return "Answered by you";
  if (question.status === "cancelled") return "Skipped";
  return question.suspended ? "Paused copy" : "Waiting for you";
}

export function userQuestionAnswerText(question: UserQuestionRecord): string {
  const response = question.resolution?.response;
  if (response?.text !== undefined) return response.text;
  if (response?.optionIds) {
    const labels = new Map(question.options.map((option) => [option.id, option.label]));
    return response.optionIds.map((id) => labels.get(id) ?? id).join(" · ");
  }
  return question.resolution?.reason ?? "No answer was recorded.";
}

export function userQuestionCountdown(question: UserQuestionRecord, now: number): string | undefined {
  if (question.status !== "pending" || question.suspended || !question.timeout.deadlineAt) return undefined;
  const remaining = Math.max(0, Math.ceil((Date.parse(question.timeout.deadlineAt) - now) / 1_000));
  if (!remaining) return "Assumption pending";
  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;
  return `Auto choice in ${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function userQuestionTimeoutSeconds(minutes: string, seconds: string): number {
  if (!/^\d+$/.test(minutes) || !/^\d+$/.test(seconds)) throw new Error("Enter whole minutes and seconds.");
  const m = Number(minutes), s = Number(seconds);
  if (!Number.isSafeInteger(m) || !Number.isSafeInteger(s) || s > 59) throw new Error("Use seconds from 0 to 59.");
  const total = m * 60 + s;
  if (!Number.isSafeInteger(total) || total < 1 || total * 1_000 > Number.MAX_SAFE_INTEGER) throw new Error("Choose a wait longer than zero within the supported time range.");
  return total;
}
