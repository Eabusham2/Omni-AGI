import { createHash } from "node:crypto";
import { closeSync, constants, fstatSync, lstatSync, openSync, readFileSync, realpathSync, type BigIntStats } from "node:fs";
import { dirname, join } from "node:path";
import { PORTABLE_MARKER, PORTABLE_MANIFEST } from "../shared/portableWorkspace";
export interface PortableBootstrap {
  root: string; userDataRoot: string; brainRoot: string; deviceDataRoot: string;
  workspaceId: string; manifestSha256: string; locationSha256: string;
}
/** Before app.ready/session creation; no environment override, lookup scan or executable invocation. */
export function readPortableBootstrap(input: { platform: string; packaged: boolean; executablePath: string; installedUserDataRoot: string }): PortableBootstrap | undefined {
  if (input.platform !== "win32" || !input.packaged) return undefined;
  const root = realpathSync(dirname(input.executablePath)), path = join(root, PORTABLE_MARKER);
  let info: BigIntStats; try { info = lstatSync(path, { bigint: true }); } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  if (!info.isFile() || info.isSymbolicLink() || info.size > 4096n) throw new Error("The portable marker must be a small regular file.");
  const fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  let marker: Record<string, unknown>;
  try { const opened = fstatSync(fd, { bigint: true }); if (opened.ino !== info.ino || opened.dev !== info.dev || opened.size !== info.size) throw new Error("Portable marker changed during opening.");
    marker = JSON.parse(readFileSync(fd, "utf8")) as Record<string, unknown>;
    const current = lstatSync(path, { bigint: true }), after = fstatSync(fd, { bigint: true }); if (after.ino !== info.ino || after.dev !== info.dev || after.size !== info.size || after.mtimeNs !== info.mtimeNs || after.ctimeNs !== info.ctimeNs || current.ino !== info.ino || current.dev !== info.dev) throw new Error("Portable marker changed during reading.");
  } finally { closeSync(fd); }
  if (!marker || Object.keys(marker).sort().join(",") !== "format,id,manifest,manifestSha256,version" || marker.format !== "omni-portable-launcher" || marker.version !== 1 || marker.manifest !== PORTABLE_MANIFEST || typeof marker.id !== "string" || /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.exec(marker.id)?.[0] !== marker.id || typeof marker.manifestSha256 !== "string" || /^[a-f0-9]{64}$/.exec(marker.manifestSha256)?.[0] !== marker.manifestSha256) throw new Error("The portable marker has an unsupported shape.");
  const locationSha256 = createHash("sha256").update(root.toLocaleLowerCase()).digest("hex");
  return { root, userDataRoot: join(root, "data"), brainRoot: join(root, "data", "brains"),
    deviceDataRoot: join(input.installedUserDataRoot, "portable-devices", marker.id, locationSha256),
    workspaceId: marker.id, manifestSha256: marker.manifestSha256, locationSha256 };
}
