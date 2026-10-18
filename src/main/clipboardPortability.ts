import { lstat } from "node:fs/promises";
import { join } from "node:path";
import { clipboardHash, clipboardId } from "./clipboardFormat";
import { ClipboardStore, CLIPBOARD_WORKSPACE_PATH } from "./clipboardStore";
import { withClipboardMutation, withClipboardSnapshot } from "./clipboardWorkspaceLease";
import { copyMutableFileIsolated } from "./mutableFileIsolation";

export function safeClipboardWorkspacePath(value: string): boolean {
  return value === CLIPBOARD_WORKSPACE_PATH || /^clipboard\/blobs\/[a-f0-9]{64}\.data$/.test(value);
}
async function exists(path: string): Promise<boolean> { return lstat(path).then(() => true, (error: NodeJS.ErrnoException) => error.code === "ENOENT" ? false : Promise.reject(error)); }

export async function validateClipboardWorkspace(root: string, brainId: string): Promise<void> {
  if (!await exists(join(root, CLIPBOARD_WORKSPACE_PATH))) {
    if (await exists(join(root, "clipboard"))) throw new Error("Clipboard workspace subtree has no owned database."); return;
  }
  const store = await ClipboardStore.open(root, brainId, true); try { await store.validate(); } finally { store.close(); }
}

export async function rekeyClipboardWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  clipboardId(sourceBrainId, "source clipboard owner"); clipboardId(targetBrainId, "target clipboard owner");
  if (!await exists(join(root, CLIPBOARD_WORKSPACE_PATH))) return;
  await withClipboardMutation(root, async () => {
    try { await validateClipboardWorkspace(root, sourceBrainId); }
    catch (sourceError) { try { await validateClipboardWorkspace(root, targetBrainId); const target = await ClipboardStore.open(root, targetBrainId);
      try { target.recoverInterruptedHostEffects("The copied clipboard workspace changed owners before this mutation reached a terminal acknowledgement.", sourceBrainId); } finally { target.close(); }
      await validateClipboardWorkspace(root, targetBrainId); return; } catch { throw sourceError; } }
    const store = await ClipboardStore.open(root, sourceBrainId); try { store.rekey(targetBrainId); } finally { store.close(); }
    await validateClipboardWorkspace(root, targetBrainId);
  });
}

export async function mergeClipboardWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentBrainId: string,
  reserve: (bytes: number) => Promise<void>): Promise<void> {
  clipboardId(owner, "merged clipboard owner"); clipboardId(parentBrainId, "clipboard parent owner");
  if (!await exists(join(sourceRoot, CLIPBOARD_WORKSPACE_PATH))) return;
  await withClipboardSnapshot(sourceRoot, async () => {
    await validateClipboardWorkspace(sourceRoot, owner);
    await withClipboardMutation(targetRoot, async () => {
    const source = await ClipboardStore.open(sourceRoot, owner, true), target = await ClipboardStore.open(targetRoot, owner);
    try {
      const versions = source.versions("", 200), aliases = new Map<string, string>(); let after = versions.nextAfter, page = versions;
      const all = [...page.entries]; while (after) { page = source.versions(after, 200); all.push(...page.entries); after = page.nextAfter; }
      const sourceCurrent = source.version(); let importedCurrent: { id: string; revision: number } | undefined;
      for (const version of all) {
        if (version.action === "initial") { aliases.set(version.id, target.version().id); continue; }
        await reserve(version.bytes + 65536); const id = clipboardHash(`clipboard-version\0${parentBrainId}\0${version.id}`), revision = target.nextRevision();
        await copyMutableFileIsolated(source.blobPath(version.contentSha256), target.blobPath(version.contentSha256), { expectedSha256: version.contentSha256 });
        target.importVersion(version, source.blobPath(version.contentSha256), id, revision,
          { kind: "inherited", capturedAt: new Date().toISOString(), brainId: owner, parentBrainId, parentVersionId: version.id });
        const imported = target.version(id); aliases.set(version.id, id); if (version.id === sourceCurrent.id) importedCurrent = { id, revision: imported.revision };
      }
      if (target.version().action === "initial" && importedCurrent) target.setCurrent(importedCurrent.id, importedCurrent.revision);
      for (const effect of source.allEffects()) target.importEffect(effect, clipboardHash(`clipboard-effect\0${parentBrainId}\0${effect.requestId}`), aliases, parentBrainId);
    } finally { source.close(); target.close(); }
    await validateClipboardWorkspace(targetRoot, owner);
    });
  });
}
