import { describe, expect, it, vi } from "vitest";
import {
  dispatchLocalNotification,
  notificationOutcome,
  requestLocalNotifications,
  type LocalNotificationEnvironment
} from "../src/renderer/src/localNotifications";

function environment(permission: NotificationPermission = "granted") {
  const show = vi.fn();
  const requestPermission = vi.fn(async () => "granted" as NotificationPermission);
  return {
    value: { permission, show, requestPermission } satisfies LocalNotificationEnvironment,
    show,
    requestPermission
  };
}

describe("local desktop notifications", () => {
  it("recognizes only meaningful completion and error outcomes", () => {
    expect(notificationOutcome("Training queued.")).toBeNull();
    expect(notificationOutcome("Crawler started.")).toBeNull();
    expect(notificationOutcome("Whole dataset training completed.")).toBe("complete");
    expect(notificationOutcome("The modality job could not finish.")).toBe("error");
  });

  it("requests permission only when needed", async () => {
    expect(await requestLocalNotifications(null)).toBe(false);
    expect(await requestLocalNotifications(environment("denied").value)).toBe(false);
    const pending = environment("default");
    expect(await requestLocalNotifications(pending.value)).toBe(true);
    expect(pending.requestPermission).toHaveBeenCalledOnce();
  });

  it("stays quiet in the foreground, deduplicates bursts, and never exposes task text", () => {
    const native = environment();
    const timestamps = {};
    const input = {
      enabled: true,
      visible: false,
      message: "Private file /secret/name.parquet completed.",
      now: 20_000,
      lastDispatchedAt: timestamps,
      environment: native.value
    };
    expect(dispatchLocalNotification({ ...input, visible: true })).toBeNull();
    expect(dispatchLocalNotification(input)).toBe("complete");
    expect(dispatchLocalNotification({ ...input, now: 21_000 })).toBeNull();
    expect(native.show).toHaveBeenCalledOnce();
    expect(JSON.stringify(native.show.mock.calls[0])).not.toContain("secret");
    expect(dispatchLocalNotification({ ...input, now: 31_000 })).toBe("complete");
  });
});
