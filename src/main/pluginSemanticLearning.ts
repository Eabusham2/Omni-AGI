import { createHash } from "node:crypto";
import type { OmniPluginPackage } from "../shared/plugins";

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, nested]) => [key, canonical(nested)]));
  }
  return value;
}

export function pluginSemanticLearningExperience(
  packageValue: OmniPluginPackage,
  contentSha256: string
): { content: string; sha256: string } {
  const content = JSON.stringify(canonical({
    format: "omni-plugin-semantic-declaration",
    version: 1,
    plugin: {
      id: packageValue.manifest.id,
      version: packageValue.manifest.version,
      contentSha256,
      name: packageValue.manifest.name,
      description: packageValue.manifest.description,
      author: packageValue.manifest.author,
      license: packageValue.manifest.license,
      documentation: packageValue.manifest.documentation
    },
    capabilities: packageValue.manifest.capabilities,
    tools: packageValue.manifest.tools,
    ...(packageValue.manifest.interactivePanels === undefined ? {} : {
      interactivePanels: packageValue.manifest.interactivePanels.map((panel) => ({
        id: panel.id,
        title: panel.title,
        tools: panel.tools
      }))
    }),
    ...(packageValue.manifest.runtimeModules === undefined ? {} : {
      runtimeModules: packageValue.manifest.runtimeModules.map((module) => ({
        id: module.id, title: module.title, description: module.description,
        events: module.events, ui: module.ui, dynamicTools: module.dynamicTools,
        activation: "explicit-human-exact-version", hostEffects: "declared-current-tool-permissions"
      }))
    }),
    ...(packageValue.manifest.trainingModules === undefined ? {} : {
      trainingModules: packageValue.manifest.trainingModules.map((module) => ({
        id: module.id, kind: module.kind, abi: module.abi, wasmSha256: module.wasmSha256
      }))
    }),
    trainingFormats: packageValue.manifest.trainingFormats,
    executedHostActions: false,
    fabricatedToolResults: false,
    hiddenPrompt: false,
    behavioralPersona: false
  }));
  return { content, sha256: createHash("sha256").update(content).digest("hex") };
}
