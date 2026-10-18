import { randomUUID, createHash } from "node:crypto";
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { McpAccessOverride, McpRegistryUpdateRequest, McpToolSummary, McpTransport } from "../shared/types";
import type { McpNegotiatedCapabilities } from "../shared/mcpCapabilities";
import { negotiatedMcpCapabilities } from "./mcpCapabilities";

export interface McpRegistryServer {
  id: string;
  label: string;
  transport: McpTransport;
  url?: string;
  command?: string;
  args?: string[];
  tools: McpToolSummary[];
  capabilities?: McpNegotiatedCapabilities;
  enabled: boolean;
  defaultEnabled: boolean;
  onlyBrainId?: string;
  credentialId: string;
  environmentKeys?: string[];
  /** Keeps old native tool identities stable when two former local ids collide. */
  namespaceByBrain: Record<string, string>;
}

export interface McpInstanceAccess {
  brainId: string;
  enabled: boolean;
  overrides: Record<string, Exclude<McpAccessOverride, "inherit">>;
  learnedSchemas: Record<string, string>;
}

export interface McpRegistryDocument {
  format: "omni-mcp-registry";
  version: 2;
  enabled: boolean;
  servers: McpRegistryServer[];
  instances: McpInstanceAccess[];
  lastConfiguredBrainId?: string;
}

export interface LegacyMcpServer {
  brainId: string;
  id: string;
  label: string;
  transport: McpTransport;
  url?: string;
  command?: string;
  args?: string[];
  tools: McpToolSummary[];
}

const MAX_REGISTRY_BYTES = 64 * 1024 * 1024;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;

export function requireMcpId(value: unknown, label: string): string {
  if (typeof value !== "string" || !SAFE_ID.test(value)) throw new Error(`Invalid MCP ${label}.`);
  return value;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid MCP registry object.");
  return value as Record<string, unknown>;
}

function requireBoolean(value: unknown): boolean {
  if (typeof value !== "boolean") throw new Error("MCP switches must be booleans.");
  return value;
}

function emptyDocument(): McpRegistryDocument {
  return { format: "omni-mcp-registry", version: 2, enabled: true, servers: [], instances: [] };
}

/** Registry writes are atomic; the client serializes read/change/publish transactions. */
export class McpRegistryStore {
  constructor(private readonly path: string) {}

  async load(validateLegacy: (value: unknown, template?: boolean) => LegacyMcpServer): Promise<McpRegistryDocument> {
    let raw: unknown;
    try {
      const info = await stat(this.path);
      if (!info.isFile() || info.size > MAX_REGISTRY_BYTES) throw new Error("The MCP registry is invalid or exceeds 64 MiB.");
      raw = JSON.parse(await readFile(this.path, "utf8"));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyDocument();
      throw error;
    }
    const source = asRecord(raw);
    if (source.format !== "omni-mcp-registry" || !Array.isArray(source.servers)) throw new Error("The MCP registry is invalid.");
    if (source.version === 1) {
      // A local credential is not silently shared with other instances during migration.
      const next = emptyDocument();
      const ids = new Set<string>();
      const oldKeys = new Set<string>();
      for (const value of source.servers) {
        const config = validateLegacy(value);
        const oldKey = `${config.brainId}:${config.id}`;
        if (oldKeys.has(oldKey)) throw new Error("The MCP registry contains a duplicate server id.");
        oldKeys.add(oldKey);
        let id = config.id;
        if (ids.has(id)) id = `${id.slice(0, 50)}-${createHash("sha256").update(oldKey).digest("hex").slice(0, 12)}`;
        if (ids.has(id)) throw new Error("The MCP registry contains a duplicate server id.");
        ids.add(id);
        const { brainId, ...server } = config;
        next.servers.push({ ...server, id, enabled: true, defaultEnabled: false, onlyBrainId: brainId,
          credentialId: `mcp:${brainId}:${config.id}`, namespaceByBrain: { [brainId]: config.id } });
        let instance = next.instances.find((entry) => entry.brainId === brainId);
        if (!instance) {
          instance = { brainId, enabled: true, overrides: {}, learnedSchemas: {} };
          next.instances.push(instance);
        }
        instance.overrides[id] = "on";
      }
      return next;
    }
    if (source.version !== 2 || !Array.isArray(source.instances)) throw new Error("The MCP registry is invalid.");
    const ids = new Set<string>();
    const servers = source.servers.map((value): McpRegistryServer => {
      const server = asRecord(value);
      const id = requireMcpId(server.id, "server id");
      if (ids.has(id)) throw new Error("The MCP registry contains a duplicate server id.");
      ids.add(id);
      if (typeof server.credentialId !== "string" || !/^mcp:[a-zA-Z0-9:_.-]{1,240}$/.test(server.credentialId)) {
        throw new Error("Invalid MCP credential reference.");
      }
      const namespaceByBrain: Record<string, string> = Object.create(null) as Record<string, string>;
      for (const [brainId, namespace] of Object.entries(asRecord(server.namespaceByBrain))) {
        requireMcpId(brainId, "brain id");
        namespaceByBrain[brainId] = requireMcpId(namespace, "server namespace");
      }
      // Validate endpoint and stored schema contents against one deterministic namespace.
      const templateBrain = Object.keys(namespaceByBrain)[0] ?? "registry";
      if (server.environmentKeys !== undefined && (!Array.isArray(server.environmentKeys) ||
        server.environmentKeys.some(name => typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name)) ||
        new Set(server.environmentKeys).size !== server.environmentKeys.length || server.transport !== "stdio")) throw new Error("Invalid MCP environment key metadata.");
      const base = validateLegacy({ ...server, brainId: templateBrain,
        id: namespaceByBrain[templateBrain] ?? id,
        tools: Array.isArray(server.tools) ? server.tools.map((tool) => ({ ...asRecord(tool), id: undefined })) : server.tools,
        }, true);
      return { id, label: base.label, transport: base.transport,
        ...(base.url ? { url: base.url } : {}), ...(base.command ? { command: base.command } : {}),
        ...(base.args ? { args: base.args } : {}), tools: base.tools,
        ...(server.capabilities !== undefined ? { capabilities: negotiatedMcpCapabilities(server.capabilities) } : {}),
        enabled: requireBoolean(server.enabled), defaultEnabled: requireBoolean(server.defaultEnabled),
        ...(server.onlyBrainId !== undefined ? { onlyBrainId: requireMcpId(server.onlyBrainId, "brain id") } : {}),
        credentialId: server.credentialId, namespaceByBrain,
        ...(server.environmentKeys ? { environmentKeys: [...server.environmentKeys] as string[] } : {}) };
    });
    const brains = new Set<string>();
    const instances = source.instances.map((value): McpInstanceAccess => {
      const instance = asRecord(value);
      const brainId = requireMcpId(instance.brainId, "brain id");
      if (brains.has(brainId)) throw new Error("Duplicate MCP instance preferences.");
      brains.add(brainId);
      const overrides: McpInstanceAccess["overrides"] = Object.create(null) as McpInstanceAccess["overrides"];
      const learnedSchemas: Record<string, string> = Object.create(null) as Record<string, string>;
      for (const [id, override] of Object.entries(asRecord(instance.overrides))) {
        if (!ids.has(id) || (override !== "on" && override !== "off")) throw new Error("Invalid MCP instance override.");
        overrides[id] = override;
      }
      for (const [id, hash] of Object.entries(asRecord(instance.learnedSchemas))) {
        if (!ids.has(id) || typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) throw new Error("Invalid MCP learned schema receipt.");
        learnedSchemas[id] = hash;
      }
      return { brainId, enabled: requireBoolean(instance.enabled), overrides, learnedSchemas };
    });
    const lastConfiguredBrainId = source.lastConfiguredBrainId === undefined ? undefined : requireMcpId(source.lastConfiguredBrainId, "brain id");
    if (lastConfiguredBrainId && !brains.has(lastConfiguredBrainId)) throw new Error("Invalid MCP last instance reference.");
    return { format: "omni-mcp-registry", version: 2, enabled: requireBoolean(source.enabled), servers, instances,
      ...(lastConfiguredBrainId ? { lastConfiguredBrainId } : {}) };
  }

  async save(document: McpRegistryDocument): Promise<void> {
    const contents = JSON.stringify(document, null, 2);
    if (Buffer.byteLength(contents) > MAX_REGISTRY_BYTES) throw new Error("The MCP registry exceeds 64 MiB.");
    await mkdir(dirname(this.path), { recursive: true });
    const temporary = `${this.path}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, contents, { encoding: "utf8", mode: 0o600 });
      await chmod(temporary, 0o600).catch(() => undefined);
      await rename(temporary, this.path);
    } finally { await rm(temporary, { force: true }).catch(() => undefined); }
  }
}

export function attachMcpInstance(document: McpRegistryDocument, brainId: string, inheritFromBrainId?: string): McpInstanceAccess {
  requireMcpId(brainId, "brain id");
  const existing = document.instances.find((instance) => instance.brainId === brainId);
  if (existing) return existing;
  const sourceId = inheritFromBrainId ?? document.lastConfiguredBrainId;
  if (sourceId) requireMcpId(sourceId, "brain id");
  const inherited = document.instances.find((instance) => instance.brainId === sourceId);
  const overrides = Object.fromEntries(Object.entries(inherited?.overrides ?? {}).filter(([id]) => {
    const server = document.servers.find((candidate) => candidate.id === id);
    return server && !server.onlyBrainId;
  }));
  const instance: McpInstanceAccess = { brainId, enabled: inherited?.enabled ?? true, overrides, learnedSchemas: {} };
  document.instances.push(instance);
  return instance;
}

export function mcpAccess(document: McpRegistryDocument, brainId: string, server: McpRegistryServer): {
  effectiveEnabled: boolean; disabledReason?: string; accessOverride: McpAccessOverride;
} {
  const instance = document.instances.find((entry) => entry.brainId === brainId);
  const override = instance && Object.hasOwn(instance.overrides, server.id) ? instance.overrides[server.id]! : "inherit";
  const disabledReason = !document.enabled ? "MCP is switched off for this installation."
    : !server.enabled ? "This server is switched off for all instances."
      : server.onlyBrainId && server.onlyBrainId !== brainId ? "This server belongs to another instance."
        : instance?.enabled === false ? "MCP is switched off for this instance."
          : !(override === "on" || (override === "inherit" && server.defaultEnabled)) ? "This server is switched off for this instance."
            : undefined;
  return { effectiveEnabled: !disabledReason, accessOverride: override, ...(disabledReason ? { disabledReason } : {}) };
}

export function updateMcpAccess(document: McpRegistryDocument, request: McpRegistryUpdateRequest): void {
  if (!request || typeof request !== "object" || Array.isArray(request)) throw new Error("Invalid MCP access update.");
  if (request.enabled !== undefined) requireBoolean(request.enabled);
  if (request.defaultEnabled !== undefined) requireBoolean(request.defaultEnabled);
  if (request.brainId !== undefined) requireMcpId(request.brainId, "brain id");
  if (request.registryId !== undefined) requireMcpId(request.registryId, "server id");
  const server = request.registryId !== undefined ? document.servers.find((entry) => entry.id === request.registryId) : undefined;
  if (request.registryId !== undefined && !server) throw new Error("MCP server was not found.");
  if (request.brainId !== undefined) {
    if (request.defaultEnabled !== undefined) throw new Error("Server defaults are installation-wide settings.");
    const instance = attachMcpInstance(document, request.brainId);
    if (server) {
      if (request.enabled !== undefined || !["inherit", "on", "off"].includes(request.accessOverride ?? "")) throw new Error("Use an instance access override for this server.");
      if (server.onlyBrainId && server.onlyBrainId !== request.brainId) throw new Error("This MCP server belongs to another instance.");
      if (request.accessOverride === "inherit") delete instance.overrides[server.id];
      else instance.overrides[server.id] = request.accessOverride as "on" | "off";
    } else {
      if (request.enabled === undefined || request.accessOverride !== undefined) throw new Error("An instance MCP switch is required.");
      instance.enabled = request.enabled;
    }
    document.lastConfiguredBrainId = request.brainId;
  } else if (server) {
    if (request.accessOverride !== undefined || (request.enabled === undefined && request.defaultEnabled === undefined)) throw new Error("A server switch or default is required.");
    if (request.enabled !== undefined) server.enabled = request.enabled;
    if (request.defaultEnabled !== undefined) server.defaultEnabled = request.defaultEnabled;
  } else {
    if (request.enabled === undefined || request.defaultEnabled !== undefined || request.accessOverride !== undefined) throw new Error("The installation MCP switch is required.");
    document.enabled = request.enabled;
  }
}
