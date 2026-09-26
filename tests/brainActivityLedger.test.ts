import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import {
  BrainActivityLedger,
  trainingSourceEvidenceFingerprint
} from "../src/main/brainActivityLedger";
import { BrainRepository } from "../src/main/brainRepository";
import { DEFAULT_CONFIG, type JournalEntry, type TrainingSource } from "../src/shared/types";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function root(): Promise<string> {
  const value = await mkdtemp(join(tmpdir(), "omni-activity-ledger-"));
  roots.push(value);
  return value;
}

function journal(index: number): JournalEntry {
  return {
    id: `journal-${String(index).padStart(6, "0")}`,
    createdAt: new Date(1_700_000_000_000 + index).toISOString(),
    kind: index % 7 === 0 ? "tool" : "learning",
    summary: `journal ${index}`,
    detail: `detail ${index}`
  };
}

function source(index: number): TrainingSource {
  return {
    id: `source-${String(index).padStart(6, "0")}`,
    name: `source ${index}`,
    path: `/Users/example/private/source-${index}.txt`,
    kind: "text",
    bytes: index + 1,
    learnedIdeas: index % 5,
    learnedConcepts: index % 7,
    learnedSynapses: index % 11,
    learnedRecords: index + 1,
    learnedParameterSteps: index % 3,
    parametersChanged: index % 2 === 0,
    importedAt: new Date(1_700_000_000_000 + index).toISOString(),
    rawTextRetained: false,
    contentHash: createHash("sha256").update(`source-${index}`).digest("hex"),
    policy: "pretrain"
  };
}

describe("brain activity ledger", () => {
  it("keyset-pages 10k+ journal and source rows with exact coverage and bounded reads", async () => {
    const directory = await root();
    const count = 10_050;
    const journals = Array.from({ length: count }, (_, index) => journal(index));
    const sources = Array.from({ length: count }, (_, index) => source(index));
    await BrainActivityLedger.replace(directory, "brain-a", journals, sources);
    const ledger = await BrainActivityLedger.open(directory, "brain-a");
    const state = ledger.getSummary();
    expect(state).toMatchObject({
      journalCount: count,
      trainingSourceCount: count,
      trainingSourceVersionCount: count
    });

    const journalIds = new Set<string>();
    let journalCursor: string | undefined;
    do {
      const page = ledger.journalPage(journalCursor, 100);
      expect(page.rowsRead).toBeLessThanOrEqual(101);
      page.entries.forEach((entry) => journalIds.add(entry.entry.id));
      journalCursor = page.nextCursor;
    } while (journalCursor);
    expect(journalIds.size).toBe(count);
    expect([...journalIds].sort()).toEqual(journals.map((entry) => entry.id).sort());

    const sourceIds = new Set<string>();
    let sourceCursor: string | undefined;
    do {
      const page = ledger.trainingSourcePage(sourceCursor, 100);
      expect(page.rowsRead).toBeLessThanOrEqual(101);
      page.entries.forEach((entry) => sourceIds.add(entry.source.id));
      sourceCursor = page.nextCursor;
    } while (sourceCursor);
    expect(sourceIds.size).toBe(count);
    expect([...sourceIds].sort()).toEqual(sources.map((entry) => entry.id).sort());

    const updated = {
      ...sources[17]!,
      learnedRecords: 999_999,
      learnedConcepts: 333
    };
    ledger.upsertTrainingSources([updated]);
    expect(ledger.trainingSourceById(updated.id)).toEqual(updated);
    expect(ledger.getSummary()).toMatchObject({
      trainingSourceCount: count,
      trainingSourceVersionCount: count + 1,
      topAdaptation: {
        sourceId: updated.id,
        learnedRecords: 999_999
      }
    });
    ledger.close();
    const cloneDirectory = await root();
    await BrainActivityLedger.clone(directory, cloneDirectory, "brain-a", "brain-b");
    const cloned = await BrainActivityLedger.open(cloneDirectory, "brain-b");
    expect(cloned.integrity()).toMatchObject({
      journalCount: count,
      trainingSourceCount: count,
      trainingSourceVersionCount: count + 1
    });
    cloned.close();
  }, 30_000);

  it("atomically clones/rekeys and detects row or cursor tampering", async () => {
    const directory = await root();
    const target = await root();
    await BrainActivityLedger.replace(
      directory,
      "brain-a",
      [journal(0), journal(1), journal(2)],
      [source(0), source(1), source(2)]
    );
    await BrainActivityLedger.clone(directory, target, "brain-a", "brain-b");
    const cloned = await BrainActivityLedger.open(target, "brain-b");
    expect(cloned.integrity()).toMatchObject({ journalCount: 3, trainingSourceCount: 3 });
    const first = cloned.journalPage(undefined, 1);
    const badCursor = `${first.nextCursor!.slice(0, -1)}${
      first.nextCursor!.endsWith("0") ? "1" : "0"
    }`;
    expect(() => cloned.journalPage(badCursor, 1)).toThrow(/cursor checksum/i);
    cloned.close();

    const database = new DatabaseSync(BrainActivityLedger.databasePath(target));
    database.prepare("UPDATE journal_entries SET payload_json=? WHERE sequence=3").run(
      JSON.stringify({ ...journal(2), summary: "tampered" })
    );
    database.close();
    const tampered = await BrainActivityLedger.open(target, "brain-b");
    try {
      expect(() => tampered.journalPage(undefined, 1)).toThrow(/row checksum/i);
    } finally {
      tampered.close();
    }
  });

  it("additively migrates the pre-fingerprint SQLite projection without losing history", async () => {
    const directory = await root();
    const originalSources = [source(0), source(1), source(2)];
    await BrainActivityLedger.replace(
      directory,
      "brain-a",
      [journal(0), journal(1)],
      originalSources
    );
    const path = BrainActivityLedger.databasePath(directory);
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      DROP INDEX training_source_current_fingerprint;
      ALTER TABLE training_source_current DROP COLUMN evidence_fingerprint;
      PRAGMA user_version=1;
    `);
    legacy.close();

    const migrated = await BrainActivityLedger.open(directory, "brain-a");
    expect(migrated.integrity()).toMatchObject({
      journalCount: 2,
      trainingSourceCount: 3,
      trainingSourceVersionCount: 3
    });
    expect(
      originalSources.map((entry) => migrated.trainingSourceById(entry.id))
    ).toEqual(originalSources);
    migrated.upsertTrainingSources([source(3)]);
    migrated.close();

    const inspected = new DatabaseSync(path, { readOnly: true });
    const columns = inspected.prepare(
      "PRAGMA table_info(training_source_current)"
    ).all() as unknown as Array<{ name: string }>;
    const indices = inspected.prepare(
      "PRAGMA index_list(training_source_current)"
    ).all() as unknown as Array<{ name: string }>;
    const version = inspected.prepare("PRAGMA user_version").get() as {
      user_version: number;
    };
    expect(columns.map((entry) => entry.name)).toContain("evidence_fingerprint");
    expect(indices.map((entry) => entry.name)).toContain(
      "training_source_current_fingerprint"
    );
    expect(version.user_version).toBe(2);
    expect(Number((inspected.prepare(
      "SELECT COUNT(*) count FROM training_source_current WHERE evidence_fingerprint=''"
    ).get() as { count: number }).count)).toBe(0);
    inspected.close();
  });

  it("migrates the legacy current-source projection before an upsert", async () => {
    const directory = await root();
    await BrainActivityLedger.replace(directory, "brain-a", [], [source(0)]);
    const path = BrainActivityLedger.databasePath(directory);
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      DROP INDEX training_source_current_fingerprint;
      ALTER TABLE training_source_current DROP COLUMN evidence_fingerprint;
    `);
    legacy.close();

    const ledger = await BrainActivityLedger.open(directory, "brain-a");
    const updated = { ...source(0), learnedRecords: 44 };
    expect(() => ledger.upsertTrainingSources([updated])).not.toThrow();
    expect(
      ledger.trainingSourceByFingerprint(
        trainingSourceEvidenceFingerprint(updated)
      )
    ).toEqual(updated);
    expect(ledger.integrity()).toMatchObject({
      trainingSourceCount: 1,
      trainingSourceVersionCount: 2
    });
    ledger.close();
  });

  it("builds a fresh export projection without raw paths, text, or tool details", async () => {
    const directory = await root();
    const projectedDirectory = await root();
    const secret = `sk-${"x".repeat(32)}`;
    const path = "/Users/example/private/secret.txt";
    await BrainActivityLedger.replace(
      directory,
      "brain-a",
      [{ ...journal(0), kind: "tool", detail: `output=${secret}; path=${path}` }],
      [{ ...source(0), path, rawTextRetained: true, rawText: secret }]
    );
    await BrainActivityLedger.project(
      directory,
      projectedDirectory,
      "brain-a",
      "brain-a",
      (entry) => ({ ...entry, detail: "Private operational detail omitted." }),
      (entry) => {
        const safe = { ...entry, rawTextRetained: false };
        delete safe.path;
        delete safe.rawText;
        return safe;
      }
    );
    const bytes = await readFile(BrainActivityLedger.databasePath(projectedDirectory));
    const exposed = bytes.toString("utf8");
    expect(exposed).not.toContain(secret);
    expect(exposed).not.toContain(path);
    expect(exposed).not.toContain("output=");
  });

  it("keeps brain.json compact while returning only a fixed compatibility window", async () => {
    const directory = await root();
    const repository = new BrainRepository(join(directory, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Bounded activity" });
    brain.journal = [
      ...(brain.journal ?? []),
      ...Array.from({ length: 10_050 }, (_, index) => journal(index))
    ];
    brain.trainingSources = Array.from({ length: 10_050 }, (_, index) => source(index));
    delete brain.activity;
    await rm(join(repository.brainDirectory(brain.id), "activity"), {
      recursive: true,
      force: true
    });
    await writeFile(
      join(repository.brainDirectory(brain.id), "brain.json"),
      JSON.stringify(brain, null, 2)
    );
    const saved = await repository.get(brain.id);
    expect(saved.journal).toHaveLength(100);
    expect(saved.trainingSources).toHaveLength(100);
    expect(saved.trainingSources.some((entry) => entry.id === source(0).id)).toBe(false);
    await expect(
      repository.trainingSourceByContentHash(brain.id, source(0).contentHash!)
    ).resolves.toMatchObject({ id: source(0).id });
    expect(saved.activity).toMatchObject({
      journalCount: 10_051,
      trainingSourceCount: 10_050
    });
    const persisted = JSON.parse(
      await readFile(join(repository.brainDirectory(brain.id), "brain.json"), "utf8")
    ) as { journal: unknown[]; trainingSources: unknown[]; activity: { journalCount: number } };
    expect(persisted.journal).toEqual([]);
    expect(persisted.trainingSources).toEqual([]);
    expect(persisted.activity.journalCount).toBe(10_051);
    expect(Buffer.byteLength(JSON.stringify(persisted))).toBeLessThan(128 * 1024);
    expect(await repository.list()).toEqual([
      expect.objectContaining({ id: brain.id, trainingSources: 10_050 })
    ]);
    expect(await repository.journalPage(brain.id, undefined, 60)).toMatchObject({
      totalEntries: 10_051,
      entries: expect.any(Array)
    });
    expect((await repository.trainingSourcePage(brain.id, undefined, 60)).entries).toHaveLength(60);
  }, 30_000);

  it("wires fixed-size Journal, Data, and Run views to paged or latent APIs", async () => {
    const [appSource, evolutionSource, preloadSource] = await Promise.all([
      readFile(join(process.cwd(), "src", "renderer", "src", "App.tsx"), "utf8"),
      readFile(
        join(process.cwd(), "src", "renderer", "src", "EvolutionWorkspace.tsx"),
        "utf8"
      ),
      readFile(join(process.cwd(), "src", "preload", "index.ts"), "utf8")
    ]);
    expect(appSource).toContain("window.omni.data.sources(brain.id, cursor, 60)");
    expect(appSource).toContain("window.omni.brain.journalPage(brain.id, nextCursor, 60)");
    expect(appSource).toContain("displayedSources.map((source)");
    expect(evolutionSource).toContain("sourceIds: []");
    expect(preloadSource).toContain("invoke(IPC.brain.journalPage");
    expect(preloadSource).toContain("invoke(IPC.data.sources");
  });
});
