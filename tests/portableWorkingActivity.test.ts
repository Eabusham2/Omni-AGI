import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { expect, it } from "vitest";
import { validatePortableWorkingActivity, validatePortableWorkingActivityMetadata } from "../src/main/portableWorkingActivity";

const schema = "CREATE TABLE working_activity_clock(singleton INTEGER PRIMARY KEY CHECK(singleton=1),format TEXT NOT NULL,version INTEGER NOT NULL,activity_at REAL NOT NULL,wall_at REAL)";
function baseline(database: DatabaseSync, wall: number | null = 1700000000.25): void {
  database.exec(schema);
  database.prepare("INSERT INTO working_activity_clock VALUES(1,?,1,?,?)").run("omni-cold-working-activity", 12.25, wall);
}
function memory(work: (database: DatabaseSync) => void): void {
  const database = new DatabaseSync(":memory:");
  try { work(database); } finally { database.close(); }
}

it("keeps old native absence and private page content unchanged during read-only custody validation", async () => {
  const directory = await mkdtemp(join(tmpdir(), "omni-portable-cold-clock-")), path = join(directory, "working.sqlite3");
  try {
    const writer = new DatabaseSync(path);
    try { writer.exec("CREATE TABLE working_pages(metadata_json TEXT NOT NULL)"); writer.prepare("INSERT INTO working_pages VALUES(?)").run(JSON.stringify({ source: "private unsanitized source", salience: .7 })); } finally { writer.close(); }
    const digest = () => readFile(path).then(bytes => createHash("sha256").update(bytes).digest("hex")), before = await digest();
    const reader = new DatabaseSync(path, { readOnly: true });
    try {
      validatePortableWorkingActivity(reader);
      for (const row of reader.prepare("SELECT metadata_json FROM working_pages").iterate()) validatePortableWorkingActivityMetadata(JSON.parse(String(row.metadata_json)));
      expect(reader.prepare("SELECT name FROM sqlite_master WHERE name='working_activity_clock'").get()).toBeUndefined();
    } finally { reader.close(); }
    expect(await digest()).toBe(before);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

it("accepts exact finite baselines, nullable wall provenance, negative rebased page time and explicit unknown age", () => {
  for (const wall of [null, 0, 1700000000.25]) memory(database => {
    baseline(database, wall); validatePortableWorkingActivity(database);
    validatePortableWorkingActivityMetadata({ activityUpdatedAt: -150.125, source: "unchanged" });
    validatePortableWorkingActivityMetadata({ activityUpdatedAt: 0 });
    validatePortableWorkingActivityMetadata({ activityRecencyUnknown: true });
    validatePortableWorkingActivityMetadata({});
  });
});

it("rejects a substituted object, changed field schema and executable baseline triggers without modifying state", () => {
  memory(database => { database.exec("CREATE VIEW working_activity_clock AS SELECT 1 AS singleton"); expect(() => validatePortableWorkingActivity(database)).toThrow(/ordinary SQLite table/); });
  for (const altered of [schema.replace("activity_at REAL", "activity_at TEXT"), schema.replace("wall_at REAL", "wall_at REAL NOT NULL"), schema.replace("wall_at REAL)", "wall_at REAL,extra TEXT)"), schema.replace("version INTEGER NOT NULL", "version INTEGER")]) {
    memory(database => { database.exec(altered); expect(() => validatePortableWorkingActivity(database)).toThrow(/schema/); });
  }
  memory(database => {
    baseline(database); database.exec("CREATE TRIGGER unexpected_baseline AFTER UPDATE ON working_activity_clock BEGIN DELETE FROM working_activity_clock; END");
    expect(() => validatePortableWorkingActivity(database)).toThrow(/schema/);
    expect(database.prepare("SELECT COUNT(*) AS n FROM working_activity_clock").get()?.n).toBe(1);
  });
});

it("requires one exact baseline row, fixed format/version and finite nonnegative active and known wall coordinates", () => {
  memory(database => { database.exec(schema); expect(() => validatePortableWorkingActivity(database)).toThrow(/row/); });
  memory(database => {
    database.exec(schema.replace(" CHECK(singleton=1)", ""));
    database.exec("INSERT INTO working_activity_clock VALUES(1,'omni-cold-working-activity',1,0,NULL),(2,'omni-cold-working-activity',1,0,NULL)");
    expect(() => validatePortableWorkingActivity(database)).toThrow(/row/);
  });
  for (const [column, value] of [["format", "other"], ["version", 2], ["activity_at", -1], ["activity_at", Infinity], ["activity_at", "bad"], ["wall_at", -1], ["wall_at", Infinity], ["wall_at", "bad"]] as const) memory(database => {
    baseline(database); database.prepare(`UPDATE working_activity_clock SET ${column}=?`).run(value);
    expect(() => validatePortableWorkingActivity(database)).toThrow(/row/);
  });
  memory(database => {
    database.exec(schema.replace(" CHECK(singleton=1)", "")); database.exec("INSERT INTO working_activity_clock VALUES(2,'omni-cold-working-activity',1,0,NULL)");
    expect(() => validatePortableWorkingActivity(database)).toThrow(/row/);
  });
});

it("mirrors Python finite-stamp and exact unknown-marker semantics without inventing a missing clock", () => {
  for (const value of [NaN, Infinity, -Infinity, null, true, "4", undefined]) {
    expect(() => validatePortableWorkingActivityMetadata({ activityUpdatedAt: value })).toThrow(/finite/);
  }
  for (const value of [false, null, 0, 1, "true", undefined]) {
    expect(() => validatePortableWorkingActivityMetadata({ activityRecencyUnknown: value })).toThrow(/unknown recency/);
  }
  expect(() => validatePortableWorkingActivityMetadata({ activityRecencyUnknown: true, activityUpdatedAt: -1 })).toThrow(/unknown recency/);
  expect(() => validatePortableWorkingActivityMetadata({ activityRecencyUnknown: true, activityUpdatedAt: undefined })).toThrow(/unknown recency/);
  expect(() => validatePortableWorkingActivityMetadata(null)).toThrow(/object/);
  expect(() => validatePortableWorkingActivityMetadata([])).toThrow(/object/);
});
