import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { environmentCanonical, environmentHash } from "./environmentFormat";

type ScalarSchema = { type: string; enum: string[] } | { type: string; enum?: undefined };
type InputSchema = ScalarSchema | { type: string; items: ScalarSchema };
function schemaDescriptor(value: string): InputSchema {
  const optional = value.endsWith("?"), raw = optional ? value.slice(0, -1) : value, array = raw.endsWith("[]"), descriptor = array ? raw.slice(0, -2) : raw;
  const item: ScalarSchema = descriptor.includes("|") ? { type: "string", enum: descriptor.split("|") }
    : ["string", "number", "boolean", "object", "integer"].includes(descriptor) ? { type: descriptor }
      : { type: "string", enum: [descriptor] };
  return array ? { type: "array", items: item } : item;
}

/** Ordinary declared examples only; no sandbox/VM process is started here. */
export function declaredEnvironmentCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => ["runtime.sandbox", "runtime.vm"].includes(tool.id)).map(tool => ({
    id: tool.id,
    actions: Object.keys(tool.actions),
    grant: "ask" as const,
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, value]) => [key, schemaDescriptor(value)])),
        required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    }))
  }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Inspect installed sandbox and VM adapters without installing, downloading, starting, or adopting anything.", kind: "tool", toolId: "runtime.sandbox", action: "capabilities", arguments: {} },
    { utterance: "Create one private OS-backed sandbox workspace with network disabled; fail closed if this platform has no supported adapter.", kind: "tool", toolId: "runtime.sandbox", action: "create", arguments: { name: "isolated-work" } },
    { utterance: "List managed sandbox identities through a bounded page without loading or running them.", kind: "tool", toolId: "runtime.sandbox", action: "list", arguments: { limit: 50 } },
    { utterance: "Inspect this exact sandbox lifecycle and retained operation receipts without replaying work.", kind: "tool", toolId: "runtime.sandbox", action: "status", arguments: { environmentId: "a".repeat(32) } },
    { utterance: "Load this exact reviewed sandbox identity before using it. On Windows Sandbox, each Load starts a clean disposable guest and must not adopt another current-user session.", kind: "tool", toolId: "runtime.sandbox", action: "load", arguments: { environmentId: "a".repeat(32) } },
    { utterance: "Run one exact installed guest program inside the loaded sandbox with no guest network or host mounts. Retain only output the adapter actually exposes: Windows Sandbox reports an exit code but no guest stdout or stderr.", kind: "tool", toolId: "runtime.sandbox", action: "run", arguments: { environmentId: "a".repeat(32), program: "/usr/bin/true", arguments: [], timeoutMs: 60000 } },
    { utterance: "Stop this exact active sandbox owner; on Windows this deletes the disposable guest. Keep committed receipts and never claim already completed effects were undone.", kind: "tool", toolId: "runtime.sandbox", action: "stop", arguments: { environmentId: "a".repeat(32) } },
    { utterance: "Inspect whether a Lima VM adapter and the reviewed local-image route are actually available without installing or downloading anything.", kind: "tool", toolId: "runtime.vm", action: "capabilities", arguments: {} },
    { utterance: "Create a Lima VM only from this current-task local image checksum and explicit guest outbound-network review; never use a remote URL, downloaded template, user provisioning script, host mount, credential forwarding, or existing user VM. Lima still performs its documented base guest and SSH setup.", kind: "tool", toolId: "runtime.vm", action: "create",
      arguments: { name: "reviewed-local-vm", imageAttachmentId: "b".repeat(32), imageSha256: "c".repeat(64), architecture: "aarch64", cpus: 2, memoryGiB: 2, diskGiB: 20, network: "guest-outbound" } },
    { utterance: "List managed VM records without starting a guest.", kind: "tool", toolId: "runtime.vm", action: "list", arguments: { limit: 50 } },
    { utterance: "Reconcile this VM record with the exact private Lima runtime and retain unknown or broken states for review.", kind: "tool", toolId: "runtime.vm", action: "status", arguments: { environmentId: "d".repeat(32) } },
    { utterance: "Load this exact stopped managed VM after current resource admission; never adopt another user VM.", kind: "tool", toolId: "runtime.vm", action: "load", arguments: { environmentId: "d".repeat(32) } },
    { utterance: "Run this exact absolute guest program without preserving host environment variables or mounting host files.", kind: "tool", toolId: "runtime.vm", action: "run", arguments: { environmentId: "d".repeat(32), program: "/usr/bin/true", arguments: [], timeoutMs: 60000 } },
    { utterance: "Stop this exact managed environment; retain cancelled/interrupted operation evidence and do not claim external effects were undone.", kind: "tool", toolId: "runtime.vm", action: "stop", arguments: { environmentId: "d".repeat(32) } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "managed-sandbox-vm-tools-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: environmentHash(environmentCanonical(body)) } as DeclaredCapabilityPack;
}
