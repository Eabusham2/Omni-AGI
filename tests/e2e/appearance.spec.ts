import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication
} from "@playwright/test";

const repository = resolve(process.cwd());

test("appearance follows the OS, persists, and keeps palette and layout independent", async () => {
  const userDataDirectory = await mkdtemp(join(tmpdir(), "omni-appearance-e2e-"));
  let application: ElectronApplication | undefined;

  try {
    application = await electron.launch({
      args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
      env: {
        ...Object.fromEntries(
          Object.entries(process.env).filter(([key]) => key !== "ELECTRON_RUN_AS_NODE")
        ),
        NODE_ENV: "test",
        OMNI_RESPONSIVE_REPOSITORY: repository,
        OMNI_RESPONSIVE_USER_DATA_DIR: userDataDirectory
      }
    });
    const page = await application.firstWindow();
    await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
    await page.waitForLoadState("domcontentloaded");
    const root = page.locator("html");

    await expect(root).toHaveAttribute("data-appearance-mode", "system");
    await expect(root).toHaveAttribute("data-color-scheme", "dark");

    await page.getByRole("button", { name: "Appearance settings" }).click();
    const dialog = page.getByRole("dialog", { name: "Appearance settings" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Auto", exact: true })).toHaveAttribute(
      "aria-pressed",
      "true"
    );

    await dialog.getByRole("button", { name: "Light", exact: true }).click();
    await expect(root).toHaveAttribute("data-appearance-mode", "light");
    await expect(root).toHaveAttribute("data-color-scheme", "light");

    await dialog.getByRole("button", { name: "Auto", exact: true }).click();
    await expect(root).toHaveAttribute("data-color-scheme", "dark");
    await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
    await expect(root).toHaveAttribute("data-color-scheme", "light");

    await dialog.getByRole("button", { name: "Colorful", exact: true }).click();
    await expect(root).toHaveAttribute("data-palette", "spectrum");
    await expect(root).toHaveAttribute("data-layout", "expressive");
    await expect(root).toHaveAttribute("data-appearance-pack", "colorful");

    const layoutGroup = dialog.getByRole("group", { name: "Shape and density" });
    await layoutGroup.getByRole("button", { name: "Blocky", exact: true }).click();
    await expect(root).toHaveAttribute("data-palette", "spectrum");
    await expect(root).toHaveAttribute("data-layout", "classic");
    await expect(root).toHaveAttribute("data-appearance-pack", "custom");

    await page.reload();
    await expect(root).toHaveAttribute("data-appearance-mode", "system");
    await expect(root).toHaveAttribute("data-color-scheme", "light");
    await expect(root).toHaveAttribute("data-palette", "spectrum");
    await expect(root).toHaveAttribute("data-layout", "classic");

    const contrast = await page.evaluate(() => {
      const parse = (color: string): [number, number, number] => {
        const channels = color.match(/[\d.]+/g)?.slice(0, 3).map(Number) ?? [0, 0, 0];
        return [channels[0] ?? 0, channels[1] ?? 0, channels[2] ?? 0];
      };
      const luminance = (color: [number, number, number]): number => {
        const channels = color.map((channel) => {
          const value = channel / 255;
          return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0]! + 0.7152 * channels[1]! + 0.0722 * channels[2]!;
      };
      const style = getComputedStyle(document.body);
      const foreground = luminance(parse(style.color));
      const background = luminance(parse(style.backgroundColor));
      return (Math.max(foreground, background) + 0.05) /
        (Math.min(foreground, background) + 0.05);
    });
    expect(contrast).toBeGreaterThanOrEqual(7);
  } finally {
    await application?.close().catch(() => undefined);
    await rm(userDataDirectory, { recursive: true, force: true });
  }
});
