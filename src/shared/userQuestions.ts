/** Questions collect task choices. They never authorize host capabilities. */
export type UserQuestionMode = "single" | "multiple" | "text";
export type UserQuestionStatus = "pending" | "answered" | "assumed" | "cancelled";

export interface UserQuestionOption {
  id: string;
  label: string;
  description?: string;
  /** Smaller numbers appear first, within the recommended/non-recommended group. */
  rank?: number;
  recommended?: boolean;
}

export interface UserQuestionResponse {
  optionIds?: string[];
  /** The answer to a text question, or an explicit Other answer when allowed. */
  text?: string;
}

export interface UserQuestionSource {
  kind: "brain" | "fact-check" | "learn" | "host";
  turnId?: string;
  actionId?: string;
}

export interface UserQuestionRequest {
  brainId: string;
  /** Reusing a request ID with the same contents returns the saved request. */
  requestId?: string;
  kind?: "task-choice";
  prompt: string;
  detail?: string;
  mode: UserQuestionMode;
  options?: UserQuestionOption[];
  minSelections?: number;
  maxSelections?: number;
  allowOther?: boolean;
  /** A proposed task assumption; it cannot enable the timeout policy. */
  suggestedResponse?: UserQuestionResponse;
  source?: UserQuestionSource;
}

export interface UserQuestionPreferences {
  timeoutEnabled: boolean;
  timeoutSeconds: number;
}

export const DEFAULT_USER_QUESTION_PREFERENCES: Readonly<UserQuestionPreferences> = Object.freeze({
  timeoutEnabled: false,
  timeoutSeconds: 20 * 60
});

export interface UserQuestionTimeout {
  enabled: boolean;
  seconds: number;
  policyRevision: string;
  deadlineAt?: string;
  assumption?: UserQuestionResponse;
  /** Text questions need a concrete suggestion before they can time out. */
  unavailableReason?: "no-suggested-response";
}

export interface UserQuestionResolution {
  id: string;
  source: "user" | "timeout" | "cancellation";
  resolvedAt: string;
  response?: UserQuestionResponse;
  reason?: string;
  deadlineAt?: string;
  /** Includes time the app was closed. It is not presented as an exact live duration. */
  overdueMilliseconds?: number;
  authority: "task-choice-only";
  permissionsGranted: false;
}

export interface UserQuestionRecord {
  format: "omni-user-question";
  version: 1;
  id: string;
  brainId: string;
  requestId: string;
  kind: "task-choice";
  prompt: string;
  detail?: string;
  mode: UserQuestionMode;
  /** Already sorted for presentation. */
  options: UserQuestionOption[];
  minSelections: number;
  maxSelections: number;
  allowOther: boolean;
  suggestedResponse?: UserQuestionResponse;
  source?: UserQuestionSource;
  requestSha256: string;
  status: UserQuestionStatus;
  createdAt: string;
  updatedAt: string;
  timeout: UserQuestionTimeout;
  resolution?: UserQuestionResolution;
  /** Set only after the host durably records the answer/assumption for continuation. */
  deliveredAt?: string;
  lineage?: Array<{ brainId: string; questionId: string; copiedAt: string }>;
  /** Copied/imported choices have no timer or delivery until explicitly resumed. */
  suspended?: { reason: "copied-instance"; fromBrainId: string; copiedAt: string };
  resumedAt?: string;
}

export interface UserQuestionRespondRequest {
  brainId: string;
  questionId: string;
  response: UserQuestionResponse;
}

export interface UserQuestionListQuery {
  status?: UserQuestionStatus | UserQuestionStatus[];
}

export type UserQuestionEvent = {
  type: "created" | "updated" | "resolved" | "cancelled" | "delivered";
  brainId: string;
  question: UserQuestionRecord;
} | {
  type: "preferences";
  brainId: string;
  preferences: UserQuestionPreferences;
} | {
  type: "error";
  brainId: string;
  message: string;
};

/** IPC contract; only host code may acknowledge a durable continuation. */
export interface UserQuestionsApi {
  request(request: UserQuestionRequest): Promise<UserQuestionRecord>;
  list(brainId: string, query?: UserQuestionListQuery): Promise<UserQuestionRecord[]>;
  respond(request: UserQuestionRespondRequest): Promise<UserQuestionRecord>;
  cancel(brainId: string, questionId: string, reason?: string): Promise<UserQuestionRecord>;
  resume(brainId: string, questionId?: string): Promise<UserQuestionRecord[]>;
  preferences(brainId: string): Promise<UserQuestionPreferences>;
  setPreferences(request: UserQuestionPreferences & { brainId: string }): Promise<UserQuestionPreferences>;
  onEvent(listener: (event: UserQuestionEvent) => void): () => void;
}
