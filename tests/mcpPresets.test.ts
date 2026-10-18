import { describe, expect, it } from "vitest";
import Ajv from "ajv";
import { MCP_PRESETS, mcpPresetDraft } from "../src/shared/mcpPresets";
import { declaredMapCapabilityPack } from "../src/main/mapCapabilityPack";
import catalog from "../tools/catalog.json";

describe("reviewed provider templates and declared map semantics", () => {
  it("uses documented exact endpoints or unresolved installed paths, never an auto installer, token or fabricated tool catalog", () => {
    expect(new Set(MCP_PRESETS.map(value => value.id)).size).toBe(MCP_PRESETS.length);
    for (const value of MCP_PRESETS) {
      expect(value.sources.length).toBeGreaterThan(0); expect(value.capabilitiesRequireLiveDiscovery).toBe(true);
      const draft = mcpPresetDraft(value.id); expect(draft.command).toBe(""); expect(draft.args).toEqual([]);
      expect(draft).not.toHaveProperty("bearerToken"); expect(draft).not.toHaveProperty("environment"); expect(draft).not.toHaveProperty("tools"); expect(draft).not.toHaveProperty("enabled");
      if (value.transport === "http") expect(new URL(draft.url).protocol).toBe("https:"); else expect(draft.url).toBe("");
    }
    expect(MCP_PRESETS.find(value => value.id === "adobe-express")?.scope).toContain("documentation");
    expect(MCP_PRESETS.find(value => value.id === "gmail")?.status).toBe("preview");
    expect(MCP_PRESETS.find(value => value.id === "gmail")?.scope).toContain("do not assume a send-mail");
    expect(() => mcpPresetDraft("invented-provider")).toThrow(/Unknown/);
  });
  it("declares real existing actions with valid typed training examples and no claimed execution or hidden persona", () => {
    const pack = declaredMapCapabilityPack(), ajv = new Ajv(); expect(pack.executedHostActions).toBe(false); expect(pack.rewardModel).toBe(false);
    expect(pack.id).toBe("private-photon-offline-coordinates-1"); expect(pack.sha256).toHaveLength(64);
    const ids = new Map(catalog.tools.map(value => [value.id, value]));
    for (const schema of pack.schemas) { expect(Object.keys(ids.get(schema.id)!.actions)).toEqual(schema.actions);
      for (const row of pack.records.filter(row => row.toolId === schema.id)) {
        if (row.action === null) throw new Error("A declared map tool example must name its real action.");
        expect(ajv.compile(schema.actionInputSchemas[row.action]!)(row.arguments)).toBe(true);
      }
    }
    expect(JSON.stringify(pack)).not.toContain("bearerToken"); expect(JSON.stringify(pack)).not.toContain("systemPrompt");
  });
});
