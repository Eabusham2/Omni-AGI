import type { DatabaseSync } from "node:sqlite";

const BASELINE_FORMAT = "omni-cold-working-activity";

/** Read-only custody validation. Absence is an old native state, not fresh age. */
export function validatePortableWorkingActivity(database: DatabaseSync): void {
  const objects = database.prepare("SELECT name,type FROM sqlite_master WHERE name='working_activity_clock' COLLATE NOCASE LIMIT 2").all();
  if (objects.length === 0) return;
  if (objects.length !== 1 || objects[0]?.name !== "working_activity_clock" || objects[0]?.type !== "table") {
    throw new Error("Saved cold working activity baseline must be its exact ordinary SQLite table.");
  }
  const columns = database.prepare("SELECT cid,name,type,\"notnull\",dflt_value,pk,hidden FROM pragma_table_xinfo('working_activity_clock') ORDER BY cid LIMIT 6").all();
  const expected = [
    ["singleton", "INTEGER", 0, 1], ["format", "TEXT", 1, 0], ["version", "INTEGER", 1, 0],
    ["activity_at", "REAL", 1, 0], ["wall_at", "REAL", 0, 0]
  ] as const;
  if (columns.length !== expected.length || expected.some(([name, type, notNull, primary], index) => {
    const column = columns[index];
    return column?.cid !== index || column.name !== name || column.type !== type || column.notnull !== notNull ||
      column.pk !== primary || column.dflt_value !== null || column.hidden !== 0;
  }) || database.prepare("SELECT 1 FROM sqlite_master WHERE tbl_name='working_activity_clock' COLLATE NOCASE AND type='trigger' LIMIT 1").get()) {
    throw new Error("Saved cold working activity baseline schema is invalid.");
  }
  const rows = database.prepare("SELECT singleton,format,version,activity_at,wall_at FROM working_activity_clock LIMIT 2").all();
  const row = rows[0];
  if (rows.length !== 1 || row?.singleton !== 1 || row.format !== BASELINE_FORMAT || row.version !== 1 ||
    typeof row.activity_at !== "number" || !Number.isFinite(row.activity_at) || row.activity_at < 0 ||
    row.wall_at !== null && (typeof row.wall_at !== "number" || !Number.isFinite(row.wall_at) || row.wall_at < 0)) {
    throw new Error("Saved cold working activity baseline row is invalid.");
  }
}

/** Mirrors Python page_activity_at; negative rebased coordinates are valid. */
export function validatePortableWorkingActivityMetadata(value: unknown): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Saved cold working activity metadata must be an object.");
  }
  if (Object.hasOwn(value, "activityRecencyUnknown")) {
    if (Reflect.get(value, "activityRecencyUnknown") !== true || Object.hasOwn(value, "activityUpdatedAt")) {
      throw new Error("Saved cold working unknown recency declaration is invalid.");
    }
    return;
  }
  if (Object.hasOwn(value, "activityUpdatedAt")) {
    const stamp: unknown = Reflect.get(value, "activityUpdatedAt");
    if (typeof stamp !== "number" || !Number.isFinite(stamp)) {
      throw new Error("Saved cold working activity timestamp must be finite.");
    }
  }
}
