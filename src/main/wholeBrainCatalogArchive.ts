import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { realpath } from "node:fs/promises";
import { collectWholeBrainCatalog, collectWholeBrainCatalogLineage, safeWholeBrainCatalogPath, wholeBrainCatalogZipSources,
  wholeBrainCatalogCanonical, wholeBrainCatalogDigest, wholeBrainCatalogBinding, readWholeBrainCatalogMetadata,
  admitWholeBrainCatalogMemory,
  type WholeBrainCatalogFile, type WholeBrainCatalogReadOptions } from "./wholeBrainCatalogPortability";
import type { ExtractedZipArchive, StreamingZipSource } from "./streamingZip";
import { MCP_TASK_CUSTODY_REQUIRED_FEATURE } from "./mcpTaskArchive";
import { API_TEACHER_MEDIA_REQUIRED_FEATURE } from "./apiTeacherMedia";

export const OMNI_BUNDLE_VERSION = 2;
export const WHOLE_BRAIN_CATALOG_REQUIRED_FEATURE = "whole-brain-catalog-closure-v1";
export const OMNI_BUNDLE_REQUIRED_FEATURES = Object.freeze([WHOLE_BRAIN_CATALOG_REQUIRED_FEATURE]);
export const OMNI_BUNDLE_SUPPORTED_FEATURES = Object.freeze([...OMNI_BUNDLE_REQUIRED_FEATURES, MCP_TASK_CUSTODY_REQUIRED_FEATURE, API_TEACHER_MEDIA_REQUIRED_FEATURE]);
type Metadata = Record<string, unknown>;
export interface WholeBrainArchiveScope { engineRoot: string; metadata: unknown; }
export interface WholeBrainCatalogArchive {
  files: Map<string, WholeBrainCatalogFile>;
  sources: StreamingZipSource[];
}

/** Validate before any identity reservation, local-reference adoption or effects.
 * Format1 is read-only legacy import, never a new export format. */
export function assertOmniBundleFeatures(value: unknown): 1 | 2 {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid .omni manifest.");
  const manifest = value as Metadata;
  if (manifest.format !== "omni-brain" || manifest.formatVersion !== 1 && manifest.formatVersion !== OMNI_BUNDLE_VERSION) throw new Error("Unsupported .omni format version; this application reads supported native format1 and format2 archives.");
  if (manifest.requiredFeatures !== undefined && (!Array.isArray(manifest.requiredFeatures) || manifest.requiredFeatures.some(feature => typeof feature !== "string") ||
      new Set(manifest.requiredFeatures).size !== manifest.requiredFeatures.length)) throw new Error("The .omni required-feature declaration is invalid.");
  const features = (manifest.requiredFeatures ?? []) as string[];
  for (const feature of features) if (!OMNI_BUNDLE_SUPPORTED_FEATURES.includes(feature)) throw new Error(`Unsupported required .omni feature: ${feature}. No instance was adopted.`);
  if (manifest.formatVersion === 1 && features.length) throw new Error("Required catalog features need .omni format2; format1 is accepted only without catalog state.");
  if (manifest.formatVersion === OMNI_BUNDLE_VERSION && !features.includes(WHOLE_BRAIN_CATALOG_REQUIRED_FEATURE)) throw new Error("Format2 .omni archives must require complete whole-brain catalog closure support.");
  return manifest.formatVersion;
}

/** Reuse the authoritative pointer/index/lineage validators; no glob history,
 * synthetic registry, model Load, metadata rewrite or source mutation. */
export async function collectWholeBrainCatalogArchive(scopes: { current: WholeBrainArchiveScope; origin: WholeBrainArchiveScope },
  options: WholeBrainCatalogReadOptions = {}): Promise<WholeBrainCatalogArchive> {
  const files = new Map<string, WholeBrainCatalogFile>(), sources: StreamingZipSource[] = [];
  for (const name of ["current", "origin"] as const) {
    options.signal?.throwIfAborted(); const scope = scopes[name];
    const catalog = await collectWholeBrainCatalog(scope.engineRoot, scope.metadata, options);
    const lineage = await collectWholeBrainCatalogLineage(scope.engineRoot, scope.metadata, options), prefix = `catalog/${name}`;
    if (catalog) {
      const counts = catalog.pointer.counts as Record<string, unknown>;
      if (catalog.pointer.records === 0 || counts.regions === 0) throw new Error(`The saved ${name} whole-brain catalog is empty; it cannot satisfy format2 closure.`);
    } else {
      // A verified copied checkpoint can retain the exact source registry only
      // in lineage. Export it without inventing destination record identities,
      // mutating source metadata or requiring a neural Load/Save.
      const binding = wholeBrainCatalogBinding(scope.metadata), { brainId: _owner, ...state } = binding;
      const candidates = [...lineage.entries.values()].filter(entry => {
        const { brainId: _source, ...saved } = entry.catalogPointer.binding as Record<string, unknown>;
        return wholeBrainCatalogCanonical(saved) === wholeBrainCatalogCanonical(state);
      });
      const pointers = new Set(candidates.map(entry => wholeBrainCatalogCanonical(entry.catalogPointer)));
      if (!candidates.length) throw new Error(`Format2 export/import requires a real saved ${name} whole-brain catalog binding or exact corresponding retained lineage. No registry was invented or implicitly Loaded.`);
      if (pointers.size !== 1) throw new Error(`Ambiguous ${name} catalog lineage: more than one distinct registry matches the saved state; an explicit source mapping is required.`);
      candidates.sort((a, b) => a.entryKeySha256 < b.entryKeySha256 ? -1 : a.entryKeySha256 > b.entryKeySha256 ? 1 : 0);
      const selected = candidates[0]!, counts = selected.catalogPointer.counts as Record<string, unknown>;
      if (selected.catalogPointer.records === 0 || counts.regions === 0) throw new Error(`The retained ${name} catalog is empty; it cannot satisfy format2.`);
      const projection = { format: "omni-whole-brain-catalog-archive-projection", version: 1, scope: name,
        relationship: "preserved-current-state", destinationRegistry: false,
        stateBinding: binding, stateBindingSha256: wholeBrainCatalogDigest(binding), sourceEntryKeySha256: selected.entryKeySha256,
        sourceEntrySha256: selected.entrySha256, sourceBrainId: selected.sourceBrainId,
        catalogBindingSha256: selected.catalogBindingSha256, catalogPointerSha256: wholeBrainCatalogDigest(selected.catalogPointer) };
      await admitWholeBrainCatalogMemory(options, 65536 + Buffer.byteLength(wholeBrainCatalogCanonical(binding)) * 8,
        "archive-only current-state projection");
      const contents = Buffer.from(wholeBrainCatalogCanonical(projection)), archiveName = `${prefix}/projection.json`;
      files.set(archiveName, { relativePath: archiveName, contents, bytes: contents.length, sha256: createHash("sha256").update(contents).digest("hex") });
      sources.push({ name: archiveName, contents });
    }
    for (const closure of [...(catalog ? [catalog] : []), lineage]) {
      for (const file of closure.files.values()) {
        if (!safeWholeBrainCatalogPath(file.relativePath)) throw new Error("Unsafe catalog archive component.");
        const archiveName = `${prefix}/${file.relativePath}`, previous = files.get(archiveName);
        if (previous && (previous.sha256 !== file.sha256 || previous.bytes !== file.bytes)) throw new Error("Conflicting exact catalog archive component.");
        files.set(archiveName, { ...file, relativePath: archiveName });
      }
      sources.push(...wholeBrainCatalogZipSources(closure, prefix));
    }
  }
  // Identical current+lineage names are a single exact content binding.
  return { files, sources: [...new Map(sources.map(source => [source.name, source])).values()] };
}

/** Complete exact namespace validation follows general ZIP/manifest checksums,
 * but still precedes any installed-instance mutation. */
export async function validateWholeBrainCatalogArchive(archive: ExtractedZipArchive, version: 1 | 2,
  metadata: { current: unknown; origin: unknown }, options: WholeBrainCatalogReadOptions = {}): Promise<WholeBrainCatalogArchive> {
  const names = [...archive.entries.keys()];
  const pointer = (value: unknown) => Boolean(value && typeof value === "object" && !Array.isArray(value) &&
    (Object.hasOwn(value, "whole_brain_catalog") || Object.hasOwn(value, "whole_brain_catalog_lineage")));
  if (version === 1) {
    if (pointer(metadata.current) || pointer(metadata.origin) || names.some(name => name.startsWith("catalog/") || /(?:^|\/)whole-brain-catalog\//.test(name))) throw new Error("This catalog-bearing or dangling format1 .omni archive requires format2. Its files were not installed or modified; re-export the source with complete format2 catalog custody.");
    return { files: new Map(), sources: [] };
  }
  const closure = await collectWholeBrainCatalogArchive({ current: { engineRoot: join(archive.root, "catalog", "current"), metadata: metadata.current },
    origin: { engineRoot: join(archive.root, "catalog", "origin"), metadata: metadata.origin } }, options);
  const declared = names.filter(name => name.startsWith("catalog/")), actualRoot = await realpath(archive.root);
  if (declared.length !== closure.files.size || declared.some(name => !closure.files.has(name))) throw new Error("The .omni catalog namespace differs from its exact current/origin/lineage closure; orphan payloads are not adopted.");
  for (const [name, file] of closure.files) {
    options.signal?.throwIfAborted(); const entry = archive.entries.get(name);
    if (!entry || entry.name !== name || entry.uncompressedBytes !== file.bytes ||
      await realpath(entry.path) !== resolve(actualRoot, ...name.split("/")) ||
      file.sourcePath && resolve(file.sourcePath) !== await realpath(entry.path)) throw new Error("The .omni catalog payload differs from its authenticated extracted entry.");
    if (file.contents) {
      const actual = await readWholeBrainCatalogMetadata(entry.path, options, file);
      if (!actual.raw.equals(Buffer.from(file.contents))) throw new Error("The archive-only catalog projection differs from its exact rederived binding.");
    }
  }
  return closure;
}

/** Replace ONLY two owned top-level catalog fields on collision. All other
 * native JSON lexemes (including sealed 1.0 descriptors) remain byte-identical.
 * The caller uses rewriteNativeIdentity first for strict native identity parsing. */
export function rewriteNativeCatalogFields(raw: string, target: Metadata): string {
  const parsed: unknown = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Native catalog metadata must be an object.");
  const owned = new Set(["whole_brain_catalog", "whole_brain_catalog_lineage"]), pieces: string[] = [];
  let cursor = 0;
  const space = () => { while (cursor < raw.length && /\s/.test(raw[cursor]!)) cursor++; };
  const string = () => { const start = cursor++; while (cursor < raw.length) { const char = raw[cursor++]; if (char === "\\") cursor++; else if (char === '"') return JSON.parse(raw.slice(start, cursor)) as string; } throw new Error("Invalid native JSON string."); };
  const value = () => {
    space(); const start = cursor;
    if (raw[cursor] === '"') string();
    else if (raw[cursor] === "{" || raw[cursor] === "[") {
      const stack = [raw[cursor++]];
      while (stack.length && cursor < raw.length) { const char = raw[cursor]; if (char === '"') string(); else { cursor++; if (char === "{" || char === "[") stack.push(char); else if (char === "}" || char === "]") stack.pop(); } }
      if (stack.length) throw new Error("Incomplete native catalog JSON value.");
    } else while (cursor < raw.length && !/[\s,}]/.test(raw[cursor]!)) cursor++;
    if (cursor === start) throw new Error("Missing native catalog JSON value.");
  };
  space(); if (raw[cursor++] !== "{") throw new Error("Invalid native metadata root."); space(); const seen = new Set<string>();
  while (raw[cursor] !== "}") {
    const start = cursor; if (raw[cursor] !== '"') throw new Error("Invalid native metadata key."); const name = string();
    if (seen.has(name)) throw new Error("Duplicate native metadata field."); seen.add(name); space(); if (raw[cursor++] !== ":") throw new Error("Invalid native metadata separator.");
    value(); if (!owned.has(name)) pieces.push(raw.slice(start, cursor)); space(); if (raw[cursor] !== ",") break; cursor++; space();
  }
  if (raw[cursor++] !== "}") throw new Error("Invalid native metadata terminator."); space(); if (cursor !== raw.length) throw new Error("Unexpected trailing native metadata.");
  for (const name of owned) if (Object.hasOwn(target, name)) pieces.push(`${JSON.stringify(name)}:${wholeBrainCatalogCanonical(target[name])}`);
  return `{${pieces.join(",\n")}}`;
}
