import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { describeRegisteredTools } from "../src/main/toolDescriptions";
import { defaultToolAction, ToolActionPicker } from "../src/renderer/src/ToolActionPicker";
import type { ToolPermissionRecord } from "../src/shared/types";

describe("registered tool descriptions never become a permission grant", () => {
  const permissions: ToolPermissionRecord[] = [
    { toolId: "runtime.vm", label: "Virtual machine", level: "off", updatedAt: "2026-10-13T00:00:00Z" },
    { toolId: "mcp.example.read", label: "Registered external read", level: "ask", updatedAt: "2026-10-13T00:00:00Z" },
    { toolId: "removed.tool", label: "Removed tool", level: "off", updatedAt: "2026-10-13T00:00:00Z" }
  ];
  it("describes Off builtins and registered external actions without mutating current authority", () => {
    const required: string[] = [], fields = { capabilities: { type: "object", properties: {}, required } };
    const external = { actions: ["call"], inputSchema: { type: "object", properties: { query: { type: "string" } } } };
    const page = describeRegisteredTools("brain-a", permissions, { "runtime.vm": ["create", "capabilities"] }, new Map([["runtime.vm", fields]]), new Map([["mcp.example.read", external]]));
    expect(page.brainId).toBe("brain-a"); expect(page.tools.map(tool => tool.source)).toEqual(["builtin", "external", "unregistered"]);
    expect(permissions[0].level).toBe("off"); expect(page.tools[2].actions).toEqual([]);
    (page.tools[0].actionInputSchemas!.capabilities.required as string[]).push("changed");
    expect(required).toEqual([]); page.tools[1].actions.push("extra"); expect(external.actions).toEqual(["call"]);
  });
  it("chooses a real no-operand inspection action, not an invented empty action or mutation", () => {
    const description = { id: "runtime.vm", label: "VM", source: "builtin" as const, actions: ["create", "status", "capabilities"], actionInputSchemas: {
      status: { required: ["environmentId"] }, capabilities: { required: [] } } };
    expect(defaultToolAction(description)).toBe("capabilities"); expect(defaultToolAction({ ...description, actions: [] })).toBe("");
  });
  it("renders declared choices but does not materialize a large schema until expansion", () => {
    const html = renderToStaticMarkup(<ToolActionPicker description={{ id: "runtime.vm", label: "VM", source: "builtin", actions: ["capabilities", "create"], actionInputSchemas: { create: { properties: { privateFixture: { description: "Only after expansion" } } } } }} action="create" disabled={false} onChange={() => { throw new Error("Unexpected action"); }} />);
    expect(html).toContain("<select"); expect(html).toContain("capabilities"); expect(html).toContain("Declared action fields"); expect(html).not.toContain("Only after expansion");
  });
});
