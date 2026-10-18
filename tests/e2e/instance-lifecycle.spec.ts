import { _electron as electron, expect, test } from "@playwright/test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

test("unsupported exact frontier is disclosed and can be explicitly stopped without loading", async () => {
  const data = await mkdtemp(join(tmpdir(), "omni-lifecycle-unsupported-ui-"));
  const env = Object.fromEntries(Object.entries(process.env).filter(([name, value]) => name !== "ELECTRON_RUN_AS_NODE" && value !== undefined)) as Record<string, string>;
  const application = await electron.launch({ args: [resolve("tests/e2e/instance-lifecycle-main.cjs")], env: { ...env, OMNI_LIFECYCLE_UI_DATA: data, OMNI_LIFECYCLE_UI_REPO: process.cwd() } });
  try {
    const page = await application.firstWindow();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.evaluate(() => (window as any).__lifecycleFixture.unsupported());
    await expect(page.getByText(/This runtime cannot continue that exact checkpoint/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Load and continue same turn", exact: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Stop saved work", exact: true })).toBeEnabled();
    await expect(page.getByText("The selected settings currently fit.", { exact: true })).toBeVisible();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => { document.documentElement.dataset.colorScheme = "light"; });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBe(0);
    await page.screenshot({ path: test.info().outputPath("instance-unsupported-light-narrow.png"), fullPage: true });
    await page.getByRole("button", { name: "Stop saved work", exact: true }).click();
    await expect(page.getByLabel("Lifecycle receipt")).toHaveText("unloaded · resident=false");
    const calls = await page.evaluate(() => (window as any).__lifecycleFixture.calls as string[]);
    expect(calls.filter(call => call === "stop-saved-turn")).toHaveLength(1); expect(calls).not.toContain("load");
  } finally { await application.close(); await rm(data, { recursive: true, force: true }); }
});

test("explicit load review waits for fit, pauses for edits and waits for the real unload receipt", async () => {
  const data = await mkdtemp(join(tmpdir(), "omni-lifecycle-ui-"));
  const env = Object.fromEntries(Object.entries(process.env).filter(([name, value]) => name !== "ELECTRON_RUN_AS_NODE" && value !== undefined)) as Record<string, string>;
  const application = await electron.launch({ args: [resolve("tests/e2e/instance-lifecycle-main.cjs")], env: { ...env, OMNI_LIFECYCLE_UI_DATA: data, OMNI_LIFECYCLE_UI_REPO: process.cwd() } });
  try {
    const page = await application.firstWindow(), errors: string[] = []; page.on("pageerror", (error) => errors.push(error.message));
    const calls = () => page.evaluate(() => (window as unknown as { __lifecycleFixture: { calls: string[] } }).__lifecycleFixture.calls);
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByLabel("Auto-load when opening this instance (off by default)")).not.toBeChecked();
    expect(await calls()).not.toContain("load");
    await expect(page.getByText("Current free RAM is below the selected envelope.")).toBeVisible();
    await page.screenshot({ path: test.info().outputPath("instance-load-fit-dark.png"), fullPage: true });
    await page.getByRole("button", { name: "Wait for fit and load", exact: true }).click();
    await expect(page.getByText("Waiting for resources. Fit is checked again while this review stays open.")).toBeVisible();
    await page.getByRole("button", { name: "Edit resource settings", exact: true }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    await expect(page.getByRole("heading", { name: "Offline settings" })).toBeVisible();
    await page.evaluate(() => (window as any).__lifecycleFixture.fit(true));
    await page.waitForTimeout(2800); expect(await calls()).not.toContain("load");
    await page.getByRole("button", { name: "Load / review fit", exact: true }).click();
    await expect(page.getByRole("button", { name: "Load instance", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Load instance", exact: true }).click();
    await expect(page.getByLabel("Lifecycle receipt")).toHaveText("loaded · resident=true");
    await page.getByRole("button", { name: "Unload", exact: true }).click();
    await expect(page.getByLabel("Lifecycle receipt")).toHaveText("suspending · resident=true");
    expect(await calls()).not.toContain("unload-ack");
    await page.evaluate(() => (window as any).__lifecycleFixture.release());
    await expect(page.getByLabel("Lifecycle receipt")).toHaveText("suspended · resident=false");
    await page.getByRole("button", { name: "Load / review fit", exact: true }).click();
    await expect(page.getByText(/5 published tokens/)).toBeVisible();
    await expect(page.getByRole("heading", { name: "Reconnect explicitly after Load" })).toBeVisible();
    await page.getByLabel("Auto-load when opening this instance (off by default)").check();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => { document.documentElement.dataset.colorScheme = "light"; });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBe(0);
    await page.screenshot({ path: test.info().outputPath("instance-resume-light-narrow.png"), fullPage: true });
    await page.getByRole("button", { name: "Load and continue same turn", exact: true }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    expect((await calls()).filter((call) => call === "load")).toHaveLength(2);
    await page.evaluate(() => { (window as any).__lifecycleFixture.reset(); (window as any).__lifecycleFixture.fit(false); });
    await page.getByRole("button", { name: "Reopen fixture instance", exact: true }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByText("Waiting for resources. Fit is checked again while this review stays open.")).toBeVisible();
    expect((await calls()).filter((call) => call === "load")).toHaveLength(2);
    await page.evaluate(() => (window as any).__lifecycleFixture.fit(true));
    await expect.poll(async () => (await calls()).filter((call) => call === "load").length).toBe(3);
    await expect(page.getByRole("dialog")).not.toBeVisible();
    expect(errors).toEqual([]);
  } finally { await application.close(); await rm(data, { recursive: true, force: true }); }
});
