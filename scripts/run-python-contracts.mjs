import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

if (process.argv.slice(2).some(value => value !== "--list") || process.argv.length > 3) {
  throw new Error("Bounded CI contracts accept only the fixed selection or --list.");
}
const root = fileURLToPath(new URL("../", import.meta.url));
const python = process.env.OMNI_PYTHON || (process.platform === "win32" ? "python" : "python3");
const result = spawnSync(python, ["-I", fileURLToPath(new URL("run-python-contracts.py", import.meta.url)), ...process.argv.slice(2)], {
  cwd: root, stdio: "inherit", shell: false, timeout: 120_000,
  env: { ...process.env, PYTHONUNBUFFERED: "1" }
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
