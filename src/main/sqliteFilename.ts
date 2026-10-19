import { win32 } from "node:path";

/** Adapt a filesystem filename at SQLite's boundary, never stored custody paths. */
export function sqliteFilename(path: string, platform: NodeJS.Platform = process.platform): string {
  // These are SQLite database names, not relative filesystem paths.
  if (path === ":memory:" || path === "") return path;
  // Callers supply checked filesystem paths. Do not pass through URI grants
  // such as immutable=1, nolock=1, mode=memory, or an unreviewed VFS selection.
  if (path.startsWith("file:")) throw new TypeError("SQLite expects a filesystem path, not a URI filename.");
  if (platform !== "win32") return path;
  // The longpath VFS retains normal locking but raises SQLite's pathname buffer
  // limit. The extended-length prefix also bypasses Win32 MAX_PATH. Percent-
  // encode the entire path so #, %, ? and UNC separators cannot become URI options.
  return `file:${encodeURIComponent(win32.toNamespacedPath(path))}?vfs=win32-longpath`;
}
