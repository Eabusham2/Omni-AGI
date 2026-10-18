export const TOUCH_BAR_STATE_FORMAT = "omni-touch-bar-state" as const;
export const TOUCH_BAR_ACTION_FORMAT = "omni-touch-bar-action" as const;

export interface TouchBarState {
  format: typeof TOUCH_BAR_STATE_FORMAT;
  version: 1;
  revision: number;
  brainId: string;
  instanceName: string;
  phase: "conversation";
  submit?: { mode: "send" | "queue"; enabled: boolean };
  stop?: { turnId: string };
  call?: { mode: "start" | "end"; enabled: boolean };
}

export interface TouchBarActionEvent {
  format: typeof TOUCH_BAR_ACTION_FORMAT;
  version: 1;
  brainId: string;
  stateRevision: number;
  action: "submit" | "stop" | "call";
  submitMode?: "send" | "queue";
  turnId?: string;
  callMode?: "start" | "end";
}

export interface TouchBarUpdateResult {
  supported: boolean;
  accepted: boolean;
  revision?: number;
}

export interface TouchBarApi {
  update(state: TouchBarState | null): Promise<TouchBarUpdateResult>;
  onAction(listener: (event: TouchBarActionEvent) => void): () => void;
}
