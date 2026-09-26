import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const app = readFileSync(
  resolve(process.cwd(), "src/renderer/src/App.tsx"),
  "utf8"
);

describe("system-access presentation", () => {
  it("keeps Build access simple while preserving the detailed authority matrix", () => {
    expect(app).toContain('label="System access"');
    expect(app).toContain('label="Full Authority"');
    expect(app).toContain("buildAccessLevels(enabled)");
    expect(app).toContain("buildAccessLevels(true, enabled)");
    expect(app).toContain("Fine-tune Off, Ask, Auto, or Full separately");
    expect(app).toContain("segmented--permissions");
  });

  it("maps device input into new builds and backfills it for existing brains", () => {
    expect(app).toContain('device: ["device.input"]');
    expect(app).toContain('"device.input": { label: "Device input"');
    expect(app).toContain("mergeToolPermissionDefaults(brain.toolPermissions");
    expect(app).toContain("mergeToolPermissionDefaults(records");
    expect(app).toContain("buildToolProtocolIds[toolId]");
  });
});
