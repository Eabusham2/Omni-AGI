import { randomUUID } from "node:crypto";
import { lstat, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { automationId } from "./automationFormat";
import { writeMutableFileIsolated } from "./mutableFileIsolation";

/** Installation-local activation generation, never imported as an action-file grant.
 * Normal restart retains it. Recovery/replacement rotates it before restoring
 * saved operational data, so an older enabled row cannot resurrect authority.
 */
export class AutomationAuthorityRegistry {
  private readonly operations = new Map<string, Promise<string>>();
  constructor(private readonly installationRoot: string) {}
  epoch(brainId: string): Promise<string> { return this.operation(brainId, false); }
  rotate(brainId: string): Promise<string> { return this.operation(brainId, true); }
  private operation(brainId: string, rotate: boolean): Promise<string> {
    automationId(brainId);
    const prior = this.operations.get(brainId) ?? Promise.resolve("");
    const current = prior.catch(() => "").then(async () => {
      const directory = join(this.installationRoot, ".automation-authority"), path = join(directory, `${brainId}.json`);
      await mkdir(directory, { recursive: true, mode: 0o700 });
      const parent = await lstat(directory); if (!parent.isDirectory() || parent.isSymbolicLink()) throw new Error("Automation authority directory is unsafe.");
      const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
      if (info) {
        if (!info.isFile() || info.isSymbolicLink() || info.size > 4096) throw new Error("Automation authority record is unsafe.");
        const row = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
        if (row.format !== "omni-automation-authority-1" || row.brainId !== brainId || typeof row.epoch !== "string" || !/^[a-f0-9-]{36}$/.test(row.epoch)) throw new Error("Automation authority owner is invalid.");
        if (!rotate) return row.epoch;
      }
      const epoch = randomUUID();
      await writeMutableFileIsolated(path, Buffer.from(JSON.stringify({ format: "omni-automation-authority-1", brainId, epoch })));
      return epoch;
    });
    this.operations.set(brainId, current);
    void current.finally(() => { if (this.operations.get(brainId) === current) this.operations.delete(brainId); }).catch(() => undefined);
    return current;
  }
}
