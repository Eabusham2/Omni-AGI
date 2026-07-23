import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");

function read(relativePath: string): string {
  return readFileSync(resolve(root, relativePath), "utf8");
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(read(relativePath)) as T;
}

describe("project integrity", () => {
  it("keeps the application independent from hosted or third-party chat runtimes", () => {
    const packageDocument = readJson<{
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    }>("package.json");
    const dependencies = {
      ...packageDocument.dependencies,
      ...packageDocument.devDependencies
    };

    expect(Object.keys(dependencies)).not.toContain("openai");
    expect(Object.keys(dependencies)).not.toContain("@anthropic-ai/sdk");
    expect(Object.keys(dependencies)).not.toContain("ollama");
    expect(read("src/shared/types.ts")).toContain(
      'export type InferenceRuntime = "adaptive-core";'
    );
    expect(read("docs/ARCHITECTURE.md")).toContain(
      "External models are not used to answer chat requests."
    );
  });

  it("ships the noncommercial terms separately from commercial permission", () => {
    const packageDocument = readJson<{ license: string }>("package.json");
    const noncommercial = read("LICENSE.md");
    const commercial = read("COMMERCIAL_LICENSE.md");

    expect(packageDocument.license).toBe("SEE LICENSE IN LICENSE.md");
    expect(noncommercial).toContain("PolyForm Noncommercial License 1.0.0");
    expect(noncommercial).toContain("Noncommercial Purpose");
    expect(commercial).toContain("Commercial and for-profit use");
    expect(read("THIRD_PARTY_NOTICES.md")).toContain("BitNet");
    expect(read("THIRD_PARTY_NOTICES.md")).toContain("snnTorch");
    expect(read("THIRD_PARTY_NOTICES.md")).toContain("NCPS");
  });

  it("requires provenance and license labels for every catalog entry", () => {
    const catalog = readJson<{
      schemaVersion: number;
      entries: Array<{
        id: string;
        kind: string;
        sourceUrl: string;
        license: string;
      }>;
    }>("catalog/catalog.json");

    expect(catalog.schemaVersion).toBe(1);
    expect(catalog.entries.length).toBeGreaterThanOrEqual(4);
    for (const entry of catalog.entries) {
      expect(entry.id).toMatch(/^[a-z0-9-]+$/);
      expect(["brain", "recipe", "dataset", "modality-pack"]).toContain(entry.kind);
      expect(entry.sourceUrl.trim().length).toBeGreaterThan(0);
      expect(entry.license.trim().length).toBeGreaterThan(0);
    }
  });

  it("defines structured protocols and explicit permission defaults for every tool", () => {
    const catalog = readJson<{
      schemaVersion: number;
      tools: Array<{
        id: string;
        defaultGrant: string;
        actions: Record<string, { input: object; output: object }>;
      }>;
    }>("tools/catalog.json");

    expect(catalog.schemaVersion).toBe(1);
    expect(catalog.tools.map((tool) => tool.id)).toEqual(
      expect.arrayContaining([
        "windows.files",
        "windows.powershell",
        "code.execute",
        "web.search",
        "web.fetch",
        "browser.automation",
        "modality.imagine",
        "agent.fork",
        "source.self-modify"
      ])
    );
    for (const tool of catalog.tools) {
      expect(["off", "ask", "auto", "full"]).toContain(tool.defaultGrant);
      expect(Object.keys(tool.actions).length).toBeGreaterThan(0);
      for (const action of Object.values(tool.actions)) {
        expect(action.input).toBeTypeOf("object");
        expect(action.output).toBeTypeOf("object");
      }
    }
  });

  it("keeps both Windows architectures in continuous packaging coverage", () => {
    const workflow = read(".github/workflows/windows.yml");
    const packageDocument = readJson<{
      scripts: Record<string, string>;
      build: { win: { target: string[] } };
    }>("package.json");

    expect(workflow).toContain('runs-on: ${{ matrix.runner }}');
    expect(workflow).toContain("arch: x64");
    expect(workflow).toContain("runner: windows-latest");
    expect(workflow).toContain("arch: arm64");
    expect(workflow).toContain("runner: windows-11-arm");
    expect(workflow).toContain('architecture: ${{ matrix.arch }}');
    expect(workflow).toContain("npm run test:python:portable");
    expect(workflow).toContain("npm run build:engine:win");
    expect(workflow).toContain("smoke-engine.ps1");
    expect(workflow).toContain("smoke-windows-package.ps1");
    expect(workflow).toContain("windows-package-smoke-*.json");
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "health" -Method "health"'
    );
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "create" -Method "create"'
    );
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "load" -Method "load"'
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      '-ArgumentList @("/S", "/D=$InstallRoot")'
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      '"Omni AGI Studio.exe"'
    );
    expect(packageDocument.scripts["package:win"]).toContain("-Arch x64");
    expect(packageDocument.scripts["package:win:arm64"]).toContain("-Arch arm64");
    expect(packageDocument.build.win.target).toEqual(expect.arrayContaining(["nsis", "zip"]));
  });
});
