import { describe, expect, it, vi } from "vitest";

vi.mock("electron", () => ({
  app: { isPackaged: false },
  BrowserWindow: { getAllWindows: vi.fn(() => []), fromWebContents: vi.fn() },
  dialog: {}, ipcMain: { handle: vi.fn(), removeHandler: vi.fn() }, nativeTheme: {}, shell: {},
  systemPreferences: {}, webContents: { fromId: vi.fn() }
}));
import { SparseUiSubscriptionRegistry } from "../src/main/ipc";

describe("sparse substrate main subscription ownership", () => {
  it("routes only the subscribed owner and rejects subscription identity reuse across brains", () => {
    const subscriptions = new SparseUiSubscriptionRegistry();
    subscriptions.subscribe({ senderId: 7, brainId: "brain-a", subscriptionId: "subscription-a" });
    subscriptions.subscribe({ senderId: 7, brainId: "brain-a", subscriptionId: "subscription-a" });
    subscriptions.subscribe({ senderId: 8, brainId: "brain-b", subscriptionId: "subscription-a" });
    expect(subscriptions.matching("brain-a")).toEqual([{ key: "7\0subscription-a",
      record: { senderId: 7, brainId: "brain-a", subscriptionId: "subscription-a" } }]);
    expect(() => subscriptions.subscribe({ senderId: 7, brainId: "brain-b", subscriptionId: "subscription-a" }))
      .toThrow(/another brain owner/i);
    expect(() => subscriptions.unsubscribe(7, "brain-b", "subscription-a")).toThrow(/another brain owner/i);
    expect(subscriptions.unsubscribe(7, "brain-a", "subscription-a")).toBe(true);
    expect(subscriptions.matching("brain-a")).toEqual([]);
  });
});
