/** Real tiny SQLite custody only; no renderer, tools, model or neural worker. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { afterEach, expect, it } from "vitest";
import { configurePluginCustodyJournal } from "../src/main/pluginCustodyJournal";
import { PluginRuntimeStore, PLUGIN_RUNTIME_WORKSPACE } from "../src/main/pluginRuntimeStore";
import { PluginPanelReceiptStore, PANEL_RECEIPT_PATH } from "../src/main/pluginPanelReceiptStore";
import { snapshotMutableSqliteIsolated } from "../src/main/mutableFileIsolation";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

it("keeps FULL per-commit durability and falls back safely on SQLite versions without the WAL-reset fix", () => {
  for (const [version, mode] of [["3.53.3", "WAL"], ["3.51.3", "WAL"], ["3.50.7", "WAL"], ["3.44.6", "WAL"],
    ["3.51.2", "PERSIST"], ["3.50.6", "PERSIST"], ["3.45.0", "PERSIST"], ["unknown", "PERSIST"]]) {
    const commands: string[] = [];
    configurePluginCustodyJournal({ prepare: () => ({ get: () => ({ version }) }), exec: (sql: string) => { commands.push(sql); } } as unknown as DatabaseSync);
    expect(commands).toEqual([`PRAGMA journal_mode=${mode}; PRAGMA synchronous=FULL;`]);
  }
});

it("keeps live plugin commits visible to independent readers and consistent portable snapshots", async () => {
  const directory = await mkdtemp(join(tmpdir(), "omni-plugin-journal-")); roots.push(directory);
  const runtime = (await PluginRuntimeStore.open(directory, "owner", { create: true, checkDisk: async () => undefined }))!;
  const panel = (await PluginPanelReceiptStore.open(directory, "owner", { checkDisk: async () => undefined }, true))!;
  try {
    for (const store of [runtime, panel]) {
      expect(store.db.prepare("PRAGMA journal_mode").get()?.journal_mode).toBe("wal");
      expect(store.db.prepare("PRAGMA synchronous").get()?.synchronous).toBe(2);
      expect(Number(store.db.prepare("PRAGMA wal_autocheckpoint").get()?.wal_autocheckpoint)).toBeGreaterThan(0);
    }
    await runtime.put("a".repeat(64), "owned-module", "counter", 42, 0);
    await panel.record("owned-view", "request", "actual-host-result", { complete: true });
    const reader = (await PluginRuntimeStore.open(directory, "owner", { readOnly: true }))!;
    try { expect(reader.value("a".repeat(64), "owned-module", "counter")).toEqual({ revision: 1, value: 42 }); }
    finally { reader.close(); }
    const runtimeCopy = join(directory, "runtime-copy.sqlite3"), panelCopy = join(directory, "panel-copy.sqlite3");
    await snapshotMutableSqliteIsolated(join(directory, PLUGIN_RUNTIME_WORKSPACE), runtimeCopy);
    await snapshotMutableSqliteIsolated(join(directory, PANEL_RECEIPT_PATH), panelCopy);
    const runtimeSnapshot = await PluginRuntimeStore.openFile(runtimeCopy, "owner"), panelSnapshot = await PluginPanelReceiptStore.captured(panelCopy, "owner");
    try {
      expect(runtimeSnapshot.value("a".repeat(64), "owned-module", "counter")).toEqual({ revision: 1, value: 42 });
      expect(panelSnapshot.history().records[0]).toMatchObject({ kind: "actual-host-result", value: { complete: true } });
      for (const store of [runtimeSnapshot, panelSnapshot]) expect(store.db.prepare("PRAGMA journal_mode").get()?.journal_mode).toBe("delete");
      expect(runtime.db.prepare("PRAGMA journal_mode").get()?.journal_mode).toBe("wal");
    } finally { runtimeSnapshot.close(); panelSnapshot.close(); }
  } finally { runtime.close(); panel.close(); }
});
