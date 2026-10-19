import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import react from "@vitejs/plugin-react";
import { build } from "vite";
import { _electron as electron, expect, test } from "@playwright/test";

const repository = resolve(process.cwd());
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
  )
);

test("offline preview keeps imagination recording in chat without claiming persisted media", async () => {
  const profile = await mkdtemp(join(tmpdir(), "omni-chat-media-preview-"));
  let application: Awaited<ReturnType<typeof electron.launch>> | undefined;
  try {
    application = await electron.launch({
      args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
      env: {
        ...inheritedEnvironment,
        NODE_ENV: "test",
        OMNI_RESPONSIVE_REPOSITORY: repository,
        OMNI_RESPONSIVE_USER_DATA_DIR: profile
      }
    });

    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    await page.setViewportSize({ width: 390, height: 844 });
    await page
      .locator("article.brain-card")
      .filter({ has: page.getByRole("heading", { name: "Aster", exact: true }) })
      .click();
    const composer = page.getByLabel("Message Aster");
    const draft = "Keep this draft while inspecting chat recording controls";
    await composer.fill(draft);
    const navigation = page.getByRole("navigation", { name: "Workspace", exact: true });
    await expect(navigation.getByRole("button")).toHaveCount(5);
    for (const label of ["Conversation", "Training & Evolution", "Brain map", "Tools & permissions", "Settings"]) {
      await expect(navigation.getByRole("button", { name: label, exact: true })).toBeVisible();
    }
    await expect(navigation.getByRole("button", { name: "Imagination", exact: true })).toHaveCount(0);
    const record = page.getByRole("button", { name: "Record", exact: true });
    await record.click();
    const panel = page.locator("#chat-capture-panel");
    await expect(record).toHaveAttribute("aria-expanded", "true");
    await expect(panel.getByText("Recording is available in the desktop app.", { exact: true })).toBeVisible();
    await expect(page.locator(".chat-imagination-media")).toHaveCount(0);
    await expect(panel.locator("img, video, audio")).toHaveCount(0);
    expect(await page.evaluate(() => typeof (window as Window & { omni?: unknown }).omni)).toBe("undefined");
    await record.click();
    await expect(record).toHaveAttribute("aria-expanded", "false");
    await expect(panel).toBeHidden();
    await navigation.getByRole("button", { name: "Training & Evolution", exact: true }).click();
    await page.getByRole("button", { name: "Back to conversation", exact: true }).click();
    await expect(composer).toHaveValue(draft);
    await expect(page.getByLabel("Send message")).toBeDisabled();
  } finally {
    try { await application?.close(); }
    finally { await rm(profile, { recursive: true, force: true }); }
  }
});

test("actual chat media presentation stays visible beside its collapsed synthetic action audit", async () => {
  // The existing fixture mounts ChatImaginationMedia with an explicitly
  // synthetic event. This is presentation proof, never decoder/file evidence.
  const profile = await mkdtemp(join(tmpdir(), "omni-inline-media-fixture-"));
  let application: Awaited<ReturnType<typeof electron.launch>> | undefined;
  try {
    const site = join(profile, "site");
    // File-based output needs no Chromium network connection. All generated
    // bytes are confined to this owned profile, not the product build/cache.
    await build({ configFile: false, envDir: false, root: repository, base: "./", plugins: [react()],
      cacheDir: join(profile, "vite-cache"), logLevel: "warn",
      resolve: { alias: { "@shared": resolve(repository, "src/shared"), "@renderer": resolve(repository, "src/renderer/src") } },
      build: { outDir: site, emptyOutDir: true, minify: false,
        rollupOptions: { input: resolve(repository, "tests/e2e/chat-imagination-media-fixture.html") } } });
    application = await electron.launch({ args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
      env: { ...inheritedEnvironment, NODE_ENV: "test", OMNI_RESPONSIVE_REPOSITORY: repository,
        OMNI_RESPONSIVE_USER_DATA_DIR: profile } });
    const page = await application.firstWindow();
    await page.goto(pathToFileURL(join(site, "tests/e2e/chat-imagination-media-fixture.html")).href);
    await expect(page.getByText(/synthetic validated event only/)).toBeVisible();
    expect(await page.evaluate(() => typeof (window as Window & { omni?: unknown }).omni)).toBe("undefined");
    const media = page.getByRole("region", { name: "Image imagination", exact: true });
    const image = media.getByAltText("Image created by the local imagination action", { exact: true });
    const audit = page.locator("details.chat-action-card");
    await expect(audit).not.toHaveAttribute("open", "");
    await expect(image).toBeVisible();
    await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    await expect(media.getByText("Final artifact available", { exact: true })).toBeVisible();
    await expect(media).toHaveAttribute("data-imagination-state", "final");
    await audit.locator("summary").click();
    await expect(audit.getByText("Fixture-only action audit.", { exact: true })).toBeVisible();
    await audit.locator("summary").click();
    await expect(image).toBeVisible();
    await page.getByRole("button", { name: "320px surface", exact: true }).click();
    await page.getByRole("button", { name: "Switch theme", exact: true }).click();
    await expect(image).toBeVisible();
    const bounds = await image.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.width).toBeLessThanOrEqual(320);
    await expect(audit).not.toHaveAttribute("open", "");
  } finally {
    try { await application?.close(); }
    finally { await rm(profile, { recursive: true, force: true }); }
  }
});
