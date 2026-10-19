import { win32 } from "node:path";

function filesystemFilename(path: string): string {
  // Never accept URI grants such as immutable=1, nolock=1, mode=memory, or
  // a caller-selected VFS at the checked-filesystem-path boundary.
  if (/^file:/i.test(path)) throw new TypeError("SQLite expects a filesystem path, not a URI filename.");
  return path;
}

/** Adapt a filesystem filename at SQLite's boundary, never stored custody paths. */
export function sqliteFilename(path: string, platform: NodeJS.Platform = process.platform): string {
  filesystemFilename(path);
  // These are SQLite database names, not relative filesystem paths.
  if (path === ":memory:" || path === "") return path;
  if (platform !== "win32") return path;
  // The longpath VFS retains normal locking but raises SQLite's pathname buffer
  // limit. The extended-length prefix also bypasses Win32 MAX_PATH. Percent-
  // encode the entire path so #, %, ? and UNC separators cannot become URI options.
  return `file:${encodeURIComponent(win32.toNamespacedPath(path))}?vfs=win32-longpath`;
}

/** Node 22 backup() does not enable URI parsing or inherit its source's VFS. */
export function sqliteBackupFilename(path: string, platform: NodeJS.Platform = process.platform,
  nodeVersion = process.versions.node): string {
  // Validate raw input independently of the native API's URI behavior.
  const raw = filesystemFilename(path);
  if (platform !== "win32" || raw === ":memory:" || raw === "") return raw;
  if (Number(nodeVersion.split(".")[0]) >= 24) return sqliteFilename(raw, platform);
  const filename = win32.toNamespacedPath(raw);
  // Node 22's default win32 VFS allows 260*4 UTF-8 path bytes,
  // including the eight bytes SQLite reserves for a journal suffix. Public
  // backup() exposes no destination-VFS option on that runtime. Refuse rather
  // than pass a URI that would create a literal filename/NTFS alternate stream.
  if (Buffer.byteLength(filename, "utf8") + 8 > 260 * 4) {
    throw new Error("Node 22 incremental SQLite backup cannot select the Windows long-path VFS; its destination exceeds 1032 UTF-8 bytes. Use Node 24 or newer.");
  }
  return filename;
}
