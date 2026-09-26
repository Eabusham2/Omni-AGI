import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { nativeSystemToolExamples } from "../src/renderer/src/systemToolPresentation";

const root = resolve(import.meta.dirname, "..");

describe("platform-neutral System tool contract", () => {
  it("publishes only platform-neutral file and shell protocols for new brains", () => {
    const catalog = JSON.parse(
      readFileSync(resolve(root, "tools/catalog.json"), "utf8")
    ) as { tools: Array<{ id: string; label: string }> };
    const ids = catalog.tools.map(({ id }) => id);

    expect(ids).toEqual(expect.arrayContaining(["system.files", "system.shell"]));
    expect(ids.some((id) => id.startsWith("windows."))).toBe(false);
    expect(catalog.tools.find(({ id }) => id === "system.files")?.label)
      .toBe("System files");
    expect(catalog.tools.find(({ id }) => id === "system.shell")?.label)
      .toBe("System shell");
  });

  it("uses POSIX examples on macOS/Linux and native PowerShell examples only on Windows", () => {
    expect(nativeSystemToolExamples("MacIntel")).toEqual({
      windows: false,
      directory: "/tmp",
      shellCommand: "date",
      pythonEntryPath: "/tmp/script.py"
    });
    expect(nativeSystemToolExamples("Linux x86_64")).toEqual(
      nativeSystemToolExamples("MacIntel")
    );
    expect(nativeSystemToolExamples("Win32")).toEqual({
      windows: true,
      directory: "C:\\Users\\Public",
      shellCommand: "Get-Date",
      pythonEntryPath: "C:\\path\\to\\script.py"
    });
  });

  it("keeps legacy IDs only in compatibility projection, not Build or protocol metadata", () => {
    const renderer = readFileSync(
      resolve(root, "src/renderer/src/App.tsx"),
      "utf8"
    );
    const buildMapping = renderer
      .split("const buildToolProtocolIds")[1]
      ?.split("const standardBuildAccess")[0] ?? "";
    const protocolMetadata = renderer
      .split("const protocolMeta")[1]
      ?.split("function mergeToolPermissionDefaults")[0] ?? "";

    expect(buildMapping).toContain('files: ["system.files"]');
    expect(buildMapping).toContain('shell: ["system.shell"]');
    expect(buildMapping).not.toContain("windows.");
    expect(protocolMetadata).toContain('"system.files"');
    expect(protocolMetadata).toContain('"system.shell"');
    expect(protocolMetadata).not.toContain("windows.");
  });
});
