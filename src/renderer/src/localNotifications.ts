export type LocalNotificationOutcome = "complete" | "error";

const COMPLETE_PATTERN = /\b(completed?|encoded|exported|installed|created|finished|learned)\b/i;
const ERROR_PATTERN = /\b(error|failed|failure|could not|couldn't|unavailable)\b/i;

export function notificationOutcome(message: string): LocalNotificationOutcome | null {
  if (ERROR_PATTERN.test(message)) return "error";
  if (COMPLETE_PATTERN.test(message)) return "complete";
  return null;
}

export interface LocalNotificationEnvironment {
  permission: NotificationPermission;
  requestPermission(): Promise<NotificationPermission>;
  show(title: string, options: NotificationOptions): void;
}

export function browserNotificationEnvironment(): LocalNotificationEnvironment | null {
  if (typeof window === "undefined" || typeof window.Notification !== "function") return null;
  return {
    permission: window.Notification.permission,
    requestPermission: () => window.Notification.requestPermission(),
    show: (title, options) => {
      new window.Notification(title, options);
    }
  };
}

export async function requestLocalNotifications(
  environment: LocalNotificationEnvironment | null
): Promise<boolean> {
  if (!environment) return false;
  if (environment.permission === "granted") return true;
  if (environment.permission === "denied") return false;
  return (await environment.requestPermission()) === "granted";
}

/**
 * Dispatches a quiet, privacy-preserving native alert only for backgrounded
 * completion/error outcomes. The body never copies chat, file names, paths, or
 * tool output onto the lock screen. Repeated events of one kind are throttled.
 */
export function dispatchLocalNotification({
  enabled,
  visible,
  message,
  now,
  lastDispatchedAt,
  environment
}: {
  enabled: boolean;
  visible: boolean;
  message: string;
  now: number;
  lastDispatchedAt: Partial<Record<LocalNotificationOutcome, number>>;
  environment: LocalNotificationEnvironment | null;
}): LocalNotificationOutcome | null {
  if (!enabled || visible || !environment || environment.permission !== "granted") return null;
  const outcome = notificationOutcome(message);
  if (!outcome || now - (lastDispatchedAt[outcome] ?? 0) < 10_000) return null;
  environment.show(outcome === "complete" ? "Omni task complete" : "Omni needs attention", {
    body: outcome === "complete"
      ? "A local task finished. Open Omni AGI Studio to review it."
      : "A local task could not finish. Open Omni AGI Studio for details.",
    tag: `omni-local-${outcome}`,
    silent: true
  });
  lastDispatchedAt[outcome] = now;
  return outcome;
}
