import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { _electron as electron, expect, test, type ElectronApplication } from "@playwright/test";

test("packaged desktop opens the library and Build screen without training", async () => {
  test.setTimeout(120_000);
  const executable = process.env.OMNI_E2E_EXECUTABLE?.trim();
  test.skip(!executable, "Only packaged artifact smoke supplies OMNI_E2E_EXECUTABLE.");

  const dataDirectory = await mkdtemp(join(tmpdir(), "omni-packaged-shell-"));
  const environment = Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
    )
  );
  let application: ElectronApplication | undefined;
  try {
    application = await electron.launch({
      executablePath: resolve(executable!),
      args: [`--user-data-dir=${join(dataDirectory, "electron-profile")}`, "--disable-gpu"],
      cwd: dirname(resolve(executable!)),
      env: {
        ...environment,
        NODE_ENV: "test",
        OMNI_AGI_DATA_DIR: dataDirectory,
        OMNI_PACKAGED_ENGINE_REQUIRED: "1"
      },
      chromiumSandbox: false,
      timeout: 60_000
    });
    const page = await application.firstWindow({ timeout: 45_000 });
    await expect(page.getByText("Brain Library").first()).toBeVisible({ timeout: 45_000 });
    await page.getByRole("button", { name: "Build a new brain" }).click();
    await expect(page.getByRole("heading", { name: "Build a brain" })).toBeVisible();
    // Deliberately stop before Create: CI verifies packaging and navigation,
    // while real neural training belongs to a separate acceptance run.
  } finally {
    if (application) await application.close().catch(() => undefined);
    await rm(dataDirectory, { recursive: true, force: true });
  }
});
