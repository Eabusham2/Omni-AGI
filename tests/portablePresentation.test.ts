import { describe, expect, it, vi } from "vitest";
import { applyPortablePresentationOnce } from "../src/renderer/src/portablePresentation";
import type { PortableWorkspaceStatus } from "../src/shared/portableWorkspace";

const status: PortableWorkspaceStatus = { portable: true, state: "review-required", workspaceId: "portable-owner", settingsSha256: "a".repeat(64), presentation: {
  appearance: { schemaVersion: 1, mode: "system", palette: "aqua", layout: "glass" },
  workspace: { schemaVersion: 1, arrangement: "split", notificationsEnabled: false, profiles: [] }, theme: null
} };
describe("portable ordinary presentation restoration", () => {
  it("restores a verified copy once without overwriting later user choices", () => {
    const values = new Map<string, string>(); const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
    const apply = vi.fn();
    expect(applyPortablePresentationOnce(status, storage, apply)).toBe(true);
    expect(apply).toHaveBeenCalledWith(status.presentation);
    expect(applyPortablePresentationOnce(status, storage, apply)).toBe(false);
    expect(apply).toHaveBeenCalledOnce();
    expect(applyPortablePresentationOnce({ portable: false, state: "installed" }, storage, apply)).toBe(false);
  });
  it("refuses an unbound or credential-bearing presentation before applying it", () => {
    const apply = vi.fn(), storage = { getItem: () => null, setItem: vi.fn() };
    expect(() => applyPortablePresentationOnce({ ...status, settingsSha256: "unverified" }, storage, apply)).toThrow("binding");
    expect(() => applyPortablePresentationOnce({ ...status, presentation: { ...status.presentation, credentials: "forbidden" } as never }, storage, apply)).toThrow("unsupported fields");
    expect(apply).not.toHaveBeenCalled(); expect(storage.setItem).not.toHaveBeenCalled();
  });
});
