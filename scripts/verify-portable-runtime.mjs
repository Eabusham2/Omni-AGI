import { resolve } from "node:path";
import { readFile } from "node:fs/promises";
import { verifyWindowsPortableRuntimeReceipt } from "./portable-runtime-receipt.mjs";
const index = process.argv.indexOf("--directory");
if (index < 0 || !process.argv[index + 1]) throw new Error("--directory must be an explicitly selected Windows package root.");
const sourceIndex = process.argv.indexOf("--source-commit");
if (sourceIndex >= 0 && !/^[a-f0-9]{40}$/u.test(process.argv[sourceIndex + 1] ?? "")) throw new Error("--source-commit requires the verified full Git commit.");
const packageValue = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
process.stdout.write(`${JSON.stringify(await verifyWindowsPortableRuntimeReceipt({ directory: resolve(process.argv[index + 1]), version: packageValue.version,
  ...(sourceIndex >= 0 ? { sourceCommit: process.argv[sourceIndex + 1] } : {}) }))}\n`);
