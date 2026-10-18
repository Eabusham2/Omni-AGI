import { createHash } from "node:crypto";
import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** Ordinary declared semantics only; examples never claim an artifact or host effect ran. */
export function declaredHtmlArtifactCapabilityPack(): DeclaredCapabilityPack {
  const tool = catalog.tools.find((candidate) => candidate.id === "studio.artifact");
  if (!tool) throw new Error("The HTML artifact tool schema is missing.");
  const schemas: DeclaredCapabilityPack["schemas"] = [{
    id: tool.id,
    actions: Object.keys(tool.actions),
    grant: "auto",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, value]) => {
          const raw = value.replace(/\?$/, ""), array = raw.endsWith("[]"), item = array ? raw.slice(0, -2) : raw;
          return [key, array ? { type: "array", items: { type: item } } : { type: item }];
        })), required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    }))
  }];
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Present this comparison as a local interactive chart in the chat.", kind: "tool",
      toolId: "studio.artifact", action: "create", arguments: {
        title: "Comparison chart", html: "<canvas id=\"chart\"></canvas>",
        script: "const c=document.querySelector('#chart');const x=c.getContext('2d');x.fillRect(20,20,80,40);",
        tools: [], height: 360
      } },
    { utterance: "Create a saved local viewer whose Read button may request only the existing file-read tool through its current permission.",
      kind: "tool", toolId: "studio.artifact", action: "create", arguments: {
        title: "Reviewed file viewer", html: "<button id=\"read\">Read selected file</button><pre id=\"output\"></pre>",
        script: "document.querySelector('#read').addEventListener('click',async()=>{const value=await omniArtifact.call('system.files','read',{path:'selected-file.txt'});document.querySelector('#output').textContent=String(value?.content??'');});",
        tools: [{ toolId: "system.files", actions: ["read"] }], height: 420
      } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "durable-chat-html-artifacts-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false,
    rewardModel: false, license: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
    provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
