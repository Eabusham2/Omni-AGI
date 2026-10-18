import { createHash } from "node:crypto";
import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** Ordinary declarations only, never a filesystem action or successful-run claim. */
export function declaredFileCapabilityPack(): DeclaredCapabilityPack {
  const tool = catalog.tools.find(value => value.id === "system.files")!;
  const descriptor = (value: string) => {
    const type = value.replace(/\?$/, "");
    return type.includes("|") ? { type: "string", enum: type.split("|") } : { type };
  };
  const schemas: DeclaredCapabilityPack["schemas"] = [{ id: tool.id, grant: "ask", actions: Object.keys(tool.actions),
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([name, value]) => {
      const input = (value as { input: Record<string, string> }).input;
      return [name, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, type]) => [key, descriptor(type)])),
        required: Object.entries(input).filter(([, type]) => !type.endsWith("?")).map(([key]) => key) }];
    })) }];
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "List a reviewed directory in bounded host-order pages. A partial page does not know the full entry count; only totalKnown at the end is a complete count.", kind: "tool", toolId: "system.files", action: "list", arguments: { path: "/absolute/reviewed/directory", pageSize: 64 } },
    { utterance: "Continue a directory listing using exactly the cursor returned by its previous real page. Each page needs current permission; a placeholder or expired cursor is not authority.", kind: "tool", toolId: "system.files", action: "list-next", arguments: { cursor: "host-returned-current-list-cursor", path: "/absolute/reviewed/directory" } },
    { utterance: "Read the first bounded page of a reviewed file. This page alone is not a full-file read when hasMore is true.", kind: "tool", toolId: "system.files", action: "read", arguments: { path: "/absolute/reviewed/file", maxBytes: 65536 } },
    { utterance: "Inspect a bounded exact segment of a reviewed absolute file without reading its remainder. Keep the returned source identity for later ranges.", kind: "tool", toolId: "system.files", action: "read-segment", arguments: { path: "/absolute/reviewed/file", offset: 0, maxBytes: 65536 } },
    { utterance: "Read every byte of a reviewed file using bounded pages. Continue its opaque cursor until complete; only a final wholeSha256 covers the entire file.", kind: "tool", toolId: "system.files", action: "read-all", arguments: { path: "/absolute/reviewed/file", maxBytes: 65536 } },
    { utterance: "Continue an exact checked file read with its real nextCursor. Current task, policy and source identity must still match; the earlier action's approval is not reused.", kind: "tool", toolId: "system.files", action: "read-next", arguments: { cursor: "host-returned-current-read-cursor", path: "/absolute/reviewed/file" } },
    { utterance: "Search reviewed directory content for exact literal UTF-8 bytes with bounded scan and result windows. Continue even an empty page while hasMore is true. No regex, case-folded-content or atomic tree snapshot is claimed.", kind: "tool", toolId: "system.files", action: "search", arguments: { path: "/absolute/reviewed/directory", mode: "content", query: "literal text", pageSize: 64, scanBytes: 1048576 } },
    { utterance: "Search file names and metadata without loading file contents. Directory traversal does not follow symbolic links.", kind: "tool", toolId: "system.files", action: "search", arguments: { path: "/absolute/reviewed/directory", mode: "name", query: "report", pageSize: 64 } },
    { utterance: "Walk reviewed file metadata without opening file contents. Results describe checked entries, not a single atomic snapshot of the whole directory tree.", kind: "tool", toolId: "system.files", action: "search", arguments: { path: "/absolute/reviewed/directory", mode: "metadata", pageSize: 64 } },
    { utterance: "Continue the same exact search using its real nextCursor, without changing its path, pattern or filters. An empty bounded scan page may still have more source bytes to visit.", kind: "tool", toolId: "system.files", action: "search-next", arguments: { cursor: "host-returned-current-search-cursor", path: "/absolute/reviewed/directory" } },
    { utterance: "Read an explicitly selected current-task attachment as exact paged bytes, not a dataset ingestion. Original warm/inherited scope and current attachment policy remain required.", kind: "tool", toolId: "system.files", action: "read-all", arguments: { attachmentId: "selected-current-task-attachment", maxBytes: 65536 } },
    { utterance: "Prepare a selected current-task file for a separately permissioned shell/code operation. Use its returned main-owned path, never infer a path from an attachment ID. Handoff copies only and does not execute or grant shell access.", kind: "tool", toolId: "system.files", action: "materialize-task", arguments: { attachmentId: "selected-current-task-attachment" } },
    { utterance: "Close the current file read, search or listing cursor when no longer needed. Closing retires its host handle; restart or copied history cannot revive it.", kind: "tool", toolId: "system.files", action: "close", arguments: { cursor: "host-returned-current-file-cursor" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "bounded-file-tools-1", sourceKind: "project-authored-declared-examples",
    executedHostActions: false, preferenceLabels: false, rewardModel: false, license: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
    provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const sorted = (value: unknown): unknown => Array.isArray(value) ? value.map(sorted) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => [key, sorted(item)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(sorted(body))).digest("hex") } as DeclaredCapabilityPack;
}
