import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync, mkdirSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const executableArgument = option("--executable");
if (!executableArgument) {
  throw new Error("Usage: node scripts/smoke-engine.mjs --executable <path> [--brain-root <path>]");
}

const executable = resolve(executableArgument);
const brainRoot = resolve(
  option("--brain-root") ?? mkdtempSync(join(tmpdir(), "omni-engine-smoke-"))
);
mkdirSync(brainRoot, { recursive: true });

const worker = spawn(executable, [], {
  cwd: dirname(executable),
  env: {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    OMNI_PROTOCOL_VERSION: "1"
  },
  stdio: ["pipe", "pipe", "pipe"]
});
const lines = createInterface({ input: worker.stdout });
const iterator = lines[Symbol.asyncIterator]();
let stderr = "";
worker.stderr.on("data", (chunk) => {
  stderr = `${stderr}${chunk.toString()}`.slice(-12_000);
});

async function nextLine(method) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`Timed out waiting for packaged worker RPC ${method}`)),
      180_000
    );
    timer.unref();
  });
  try {
    return await Promise.race([iterator.next(), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function waitForExit() {
  if (worker.exitCode !== null || worker.signalCode !== null) return;
  await new Promise((resolveExit, rejectExit) => {
    const timer = setTimeout(() => {
      rejectExit(new Error("Packaged worker did not exit after shutdown."));
    }, 30_000);
    const onExit = () => {
      clearTimeout(timer);
      resolveExit();
    };
    worker.once("exit", onExit);
  });
}

async function rpc(id, method, params = {}) {
  worker.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
  while (true) {
    const result = await nextLine(method);
    if (result.done) {
      throw new Error(`Packaged worker closed stdout during ${method}.\n${stderr}`);
    }
    let message;
    try {
      message = JSON.parse(result.value);
    } catch {
      continue;
    }
    if (message.id !== id) continue;
    if (message.error) {
      throw new Error(`Packaged worker RPC ${method} failed: ${message.error.message}`);
    }
    return message.result;
  }
}

let stopped = false;
try {
  const health = await rpc("health", "health");
  if (
    health?.ready !== true ||
    health?.worker !== "python" ||
    health?.protocolVersion !== 1 ||
    typeof health?.operatingSystem !== "string"
  ) {
    throw new Error("Packaged worker returned an invalid health response.");
  }

  const storagePath = join(brainRoot, "packaged-smoke-brain");
  const created = await rpc("create", "create", {
    brainId: "packaged-smoke-brain",
    storagePath,
    hardwareTier: "micro",
    config: {
      name: "Packaged worker smoke",
      hardwareTier: "micro",
      parallelThoughts: 1
    }
  });
  if (created?.brainId !== "packaged-smoke-brain") {
    throw new Error("Packaged worker did not create the requested brain identity.");
  }

  const unloaded = await rpc("unload", "unload", {
    brainId: "packaged-smoke-brain",
    storagePath
  });
  if (unloaded?.unloaded !== true) {
    throw new Error("Packaged worker did not unload its created brain.");
  }

  const loaded = await rpc("load", "load", {
    brainId: "packaged-smoke-brain",
    storagePath
  });
  if (loaded?.brainId !== "packaged-smoke-brain") {
    throw new Error("Packaged worker could not reload its safe-tensor checkpoint.");
  }
  const state = await rpc("state", "state", {
    brainId: "packaged-smoke-brain",
    storagePath
  });
  if (state?.brainId !== "packaged-smoke-brain") {
    throw new Error("Packaged worker could not inspect its reloaded brain.");
  }

  for (const name of [
    "brain.json",
    "core.safetensors",
    "plasticity.safetensors",
    "events.sqlite3"
  ]) {
    if (!existsSync(join(storagePath, "engine", name))) {
      throw new Error(`Packaged worker smoke is missing engine/${name}`);
    }
  }

  const shutdown = await rpc("shutdown", "shutdown");
  if (shutdown?.stopping !== true) {
    throw new Error("Packaged worker did not acknowledge shutdown.");
  }
  await waitForExit();
  stopped = true;

  process.stdout.write(
    `${JSON.stringify(
      {
        executable,
        executableName: basename(executable),
        engineVersion: health.engineVersion,
        protocolVersion: health.protocolVersion,
        pythonVersion: health.pythonVersion,
        torchVersion: health.torchVersion,
        platform: health.platform,
        operatingSystem: health.operatingSystem,
        persistedBrain: true,
        safeTensorCheckpoint: true,
        sqliteEventLog: true
      },
      null,
      2
    )}\n`
  );
} finally {
  lines.close();
  if (!stopped && worker.exitCode === null) worker.kill("SIGKILL");
}
