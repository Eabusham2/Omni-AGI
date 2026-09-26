import { resolve } from "node:path";
import { _electron as electron, expect, test, type Page } from "@playwright/test";

const repository = resolve(process.cwd());
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
  )
);

const packs = [
  { id: "standard", palette: "violet", layout: "standard" },
  { id: "classic", palette: "graphite", layout: "classic" },
  { id: "colorful", palette: "spectrum", layout: "expressive" },
  { id: "liquid-glass", palette: "aqua", layout: "glass" }
] as const;

const viewports = [
  { id: "desktop", width: 1440, height: 900 },
  { id: "mobile", width: 430, height: 820 }
] as const;

type ContrastResult = {
  id: string;
  selector: string;
  ratio: number;
  minimum: number;
  foreground: string;
  background: string;
  opacity: number;
};

type ContrastTarget = {
  id: string;
  selector: string;
  minimum: number;
  backgroundSelector?: string;
  foregroundProperty?: "color" | "fill" | "stroke";
};

async function launchAppearanceApplication() {
  return electron.launch({
    args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
    env: { ...inheritedEnvironment, NODE_ENV: "test", OMNI_RESPONSIVE_REPOSITORY: repository }
  });
}

async function selectAppearance(
  page: Page,
  mode: "light" | "dark",
  palette: string,
  layout: string
): Promise<void> {
  await page.evaluate(
    ({ modeValue, paletteValue, layoutValue }) => {
      window.localStorage.setItem(
        "omni.appearance.v1",
        JSON.stringify({
          schemaVersion: 1,
          mode: modeValue,
          palette: paletteValue,
          layout: layoutValue
        })
      );
    },
    { modeValue: mode, paletteValue: palette, layoutValue: layout }
  );
  await page.reload();
  await page.waitForLoadState("domcontentloaded");
  await expect(page.locator("html")).toHaveAttribute("data-color-scheme", mode);
  await expect(page.locator("html")).toHaveAttribute("data-palette", palette);
  await expect(page.locator("html")).toHaveAttribute("data-layout", layout);
  await page.locator(".brain-card", { hasText: "Aster" }).click();
  await expect(page.locator(".conversation")).toBeVisible();
}

async function installAuditFixtures(page: Page): Promise<void> {
  await page.evaluate(() => {
    document.querySelector("[data-theme-audit-fixture]")?.remove();
    const host = document.createElement("div");
    host.dataset.themeAuditFixture = "true";
    host.className = "chat-action-stream";
    host.innerHTML = `
      <article class="chat-action-card chat-action-card--complete">
        <span class="chat-action-card__icon" aria-hidden="true">✓</span>
        <div class="chat-action-card__body">
          <div class="chat-action-card__head">
            <span><small>CHAT ACTION</small><strong>files / inspect</strong></span>
            <em><i></i> complete</em>
          </div>
          <code>{"path":"theme-audit.txt"}</code>
        </div>
      </article>
      <button class="button" disabled>Disabled action</button>
      <label class="simple-name-field">
        <span>Brain name</span>
        <input value="Readable cortex name" aria-label="Theme audit name" />
        <small>Names remain local to this identity.</small>
      </label>
      <section class="simple-review-card">
        <div><span><small>READY TO BUILD</small><strong>Readable review</strong></span></div>
        <dl>
          <div><dt>Native core</dt><dd>Locally initialized</dd></div>
          <div><dt>Working memory</dt><dd>Automatic</dd></div>
        </dl>
      </section>
      <label class="toggle-row is-disabled">
        <span><strong>Unavailable option</strong><small>Disabled settings must remain legible.</small></span>
        <input type="checkbox" disabled /><i><b></b></i>
      </label>
      <div class="workspace-meter">
        <div><span>Workspace memory</span><strong>4.8 GB available</strong></div>
        <i><b style="width: 62%"></b></i>
        <small>Automatic allocation remains inside the live-safe pool.</small>
      </div>
      <details class="live-perception-panel" open>
        <summary>
          <span class="live-perception-panel__icon"></span>
          <span><strong>Live Perception</strong><small>Camera · 1920×1080 @ 24 FPS</small></span>
          <em class="is-active"><i></i> Observing</em>
        </summary>
        <div class="live-perception-panel__body">
          <div class="live-perception-facts">
            <span><small>Applied capture</small><strong>1920×1080 @ 24 FPS</strong></span>
            <span><small>Resource class</small><strong>High</strong></span>
            <span><small>Accepted</small><strong>1,248</strong></span>
            <span><small>Backpressure</small><strong>2</strong></span>
          </div>
          <p class="live-perception-control-status"><span>Brain configure · applied · r4</span></p>
          <p class="live-perception-privacy">Raw packets are not stored and dataset coverage is not committed.</p>
        </div>
      </details>
    `;
    document.querySelector(".message-stream")?.append(host);
  });
}

async function measureContrast(
  page: Page,
  requestedTargets?: ContrastTarget[]
): Promise<ContrastResult[]> {
  return page.evaluate((providedTargets) => {
    type Rgba = { r: number; g: number; b: number; a: number };

    const targets: ContrastTarget[] = providedTargets ?? [
      { id: "body", selector: "body", minimum: 7 },
      { id: "nav-icon", selector: ".workspace-rail nav button:not(.is-active)", minimum: 3 },
      { id: "nav-active-icon", selector: ".workspace-rail nav button.is-active", minimum: 3 },
      { id: "secondary-text", selector: ".conversation__date time", minimum: 4.5 },
      { id: "input-text", selector: ".composer textarea", minimum: 4.5 },
      { id: "brain-message", selector: ".message--brain .message__content", minimum: 4.5 },
      { id: "human-message", selector: ".message--human .message__content", minimum: 4.5 },
      { id: "action-label", selector: "[data-theme-audit-fixture] .chat-action-card__head small", minimum: 4.5 },
      { id: "action-title", selector: "[data-theme-audit-fixture] .chat-action-card__head strong", minimum: 4.5 },
      { id: "action-detail", selector: "[data-theme-audit-fixture] .chat-action-card__body > code", minimum: 4.5 },
      { id: "disabled-text", selector: "[data-theme-audit-fixture] button:disabled", minimum: 3 },
      { id: "appearance-help", selector: ".appearance-panel > header small", minimum: 4.5 },
      { id: "appearance-legend", selector: ".appearance-fieldset legend", minimum: 4.5 },
      { id: "appearance-mode", selector: ".appearance-segments button:not(.is-active)", minimum: 4.5 },
      { id: "appearance-pack-title", selector: ".appearance-packs > button:not(.is-active) strong", minimum: 4.5 },
      { id: "appearance-pack-description", selector: ".appearance-packs > button:not(.is-active) small", minimum: 4.5 },
      { id: "builder-name", selector: "[data-theme-audit-fixture] .simple-name-field input", minimum: 4.5 },
      { id: "builder-help", selector: "[data-theme-audit-fixture] .simple-name-field small", minimum: 4.5 },
      { id: "builder-review-label", selector: "[data-theme-audit-fixture] .simple-review-card dt", minimum: 4.5 },
      { id: "builder-review-value", selector: "[data-theme-audit-fixture] .simple-review-card dd", minimum: 4.5 },
      { id: "disabled-setting", selector: "[data-theme-audit-fixture] .toggle-row.is-disabled small", minimum: 3 },
      { id: "workspace-meter-label", selector: "[data-theme-audit-fixture] .workspace-meter span", minimum: 4.5 },
      { id: "workspace-meter-value", selector: "[data-theme-audit-fixture] .workspace-meter strong", minimum: 4.5 },
      { id: "workspace-meter-detail", selector: "[data-theme-audit-fixture] .workspace-meter small", minimum: 4.5 },
      { id: "perception-title", selector: "[data-theme-audit-fixture] .live-perception-panel > summary strong", minimum: 4.5 },
      { id: "perception-detail", selector: "[data-theme-audit-fixture] .live-perception-panel > summary small", minimum: 4.5 },
      { id: "perception-status", selector: "[data-theme-audit-fixture] .live-perception-panel > summary em", minimum: 4.5 },
      { id: "perception-fact-label", selector: "[data-theme-audit-fixture] .live-perception-facts small", minimum: 4.5 },
      { id: "perception-fact-value", selector: "[data-theme-audit-fixture] .live-perception-facts strong", minimum: 4.5 },
      { id: "perception-control", selector: "[data-theme-audit-fixture] .live-perception-control-status", minimum: 4.5 },
      { id: "perception-privacy", selector: "[data-theme-audit-fixture] .live-perception-privacy", minimum: 4.5 }
    ];

    const parseColor = (value: string): Rgba => {
      const normalized = value.trim().toLowerCase();
      if (normalized === "transparent") return { r: 0, g: 0, b: 0, a: 0 };
      if (normalized.startsWith("color(srgb")) {
        const channels = normalized.match(/[+-]?(?:\d*\.)?\d+/g)?.map(Number) ?? [];
        return {
          r: Math.max(0, Math.min(255, (channels[0] ?? 0) * 255)),
          g: Math.max(0, Math.min(255, (channels[1] ?? 0) * 255)),
          b: Math.max(0, Math.min(255, (channels[2] ?? 0) * 255)),
          a: Math.max(0, Math.min(1, channels[3] ?? 1))
        };
      }
      const channels = normalized.match(/[+-]?(?:\d*\.)?\d+/g)?.map(Number) ?? [];
      return {
        r: Math.max(0, Math.min(255, channels[0] ?? 0)),
        g: Math.max(0, Math.min(255, channels[1] ?? 0)),
        b: Math.max(0, Math.min(255, channels[2] ?? 0)),
        a: Math.max(0, Math.min(1, channels[3] ?? 1))
      };
    };

    const composite = (front: Rgba, back: Rgba): Rgba => {
      const alpha = front.a + back.a * (1 - front.a);
      if (alpha <= 0) return { r: 0, g: 0, b: 0, a: 0 };
      return {
        r: (front.r * front.a + back.r * back.a * (1 - front.a)) / alpha,
        g: (front.g * front.a + back.g * back.a * (1 - front.a)) / alpha,
        b: (front.b * front.a + back.b * back.a * (1 - front.a)) / alpha,
        a: alpha
      };
    };

    const backdrop = (element: Element): Rgba => {
      const ancestry: Element[] = [];
      for (let current: Element | null = element; current; current = current.parentElement) {
        ancestry.unshift(current);
      }
      let color: Rgba = document.documentElement.dataset.colorScheme === "dark"
        ? { r: 0, g: 0, b: 0, a: 1 }
        : { r: 255, g: 255, b: 255, a: 1 };
      for (const current of ancestry) {
        const style = getComputedStyle(current);
        const layer = parseColor(style.backgroundColor);
        layer.a *= Number(style.opacity || "1");
        color = composite(layer, color);
      }
      return color;
    };

    const luminance = (color: Rgba): number => {
      const channel = (value: number): number => {
        const unit = value / 255;
        return unit <= 0.04045 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
    };

    const format = (color: Rgba): string =>
      `rgb(${Math.round(color.r)} ${Math.round(color.g)} ${Math.round(color.b)})`;

    return targets.map((target) => {
      const element = document.querySelector(target.selector);
      if (!element) throw new Error(`Theme audit target is missing: ${target.selector}`);
      const style = getComputedStyle(element);
      const background = backdrop(
        target.backgroundSelector
          ? document.querySelector(target.backgroundSelector) ?? element
          : element
      );
      let opacity = 1;
      for (let current: Element | null = element; current; current = current.parentElement) {
        opacity *= Number(getComputedStyle(current).opacity || "1");
      }
      const rawForeground = parseColor(style[target.foregroundProperty ?? "color"]);
      rawForeground.a *= opacity;
      const foreground = composite(rawForeground, background);
      const foregroundLuminance = luminance(foreground);
      const backgroundLuminance = luminance(background);
      const ratio = (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
        (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
      return {
        id: target.id,
        selector: target.selector,
        ratio: Number(ratio.toFixed(2)),
        minimum: target.minimum,
        foreground: format(foreground),
        background: format(background),
        opacity: Number(opacity.toFixed(2))
      };
    });
  }, requestedTargets);
}

async function layoutSignature(page: Page): Promise<{
  railWidth: number;
  chatColumns: string;
  messageInset: number;
  messageWidth: number;
  surfaceRadius: string;
  surfaceFont: string;
  cardColumns: string;
  cardGap: string;
}> {
  return page.evaluate(() => {
    const bounds = (selector: string): DOMRect => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Layout audit target is missing: ${selector}`);
      return element.getBoundingClientRect();
    };
    const rail = bounds(".workspace-rail");
    const messageStream = bounds(".message-stream");
    const firstMessage = bounds(".message");
    const surface = getComputedStyle(document.querySelector(".composer")!);
    const chat = getComputedStyle(document.querySelector(".chat-layout")!);
    const cards = getComputedStyle(document.querySelector(".chat-action-stream")!);
    return {
      railWidth: Number(rail.width.toFixed(2)),
      chatColumns: chat.gridTemplateColumns,
      messageInset: Number((firstMessage.left - messageStream.left).toFixed(2)),
      messageWidth: Number(firstMessage.width.toFixed(2)),
      surfaceRadius: surface.borderRadius,
      surfaceFont: surface.fontFamily,
      cardColumns: cards.gridTemplateColumns,
      cardGap: cards.gap || `${cards.rowGap} ${cards.columnGap}`
    };
  });
}

test("System appearance follows live OS light and dark changes", async () => {
  const application = await launchAppearanceApplication();
  try {
    const page = await application.firstWindow();
    await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
    await page.waitForLoadState("domcontentloaded");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "omni.appearance.v1",
        JSON.stringify({ schemaVersion: 1, mode: "system", palette: "violet", layout: "standard" })
      );
    });
    await page.reload();
    const root = page.locator("html");
    await expect(root).toHaveAttribute("data-appearance-mode", "system");
    await expect(root).toHaveAttribute("data-color-scheme", "light");

    await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
    await expect(root).toHaveAttribute("data-color-scheme", "dark");
    await page.getByRole("button", { name: "Appearance settings" }).click();
    await expect(page.getByText("Following the OS · currently dark", { exact: true })).toBeVisible();

    await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
    await expect(root).toHaveAttribute("data-color-scheme", "light");
    await expect(page.getByText("Following the OS · currently light", { exact: true })).toBeVisible();
  } finally {
    await application.close();
  }
});

test("fresh platform defaults are native while a saved preference always wins", async () => {
  const appleApplication = await launchAppearanceApplication();
  try {
    const page = await appleApplication.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    await page.addInitScript(() => {
      Object.defineProperty(Navigator.prototype, "platform", {
        configurable: true,
        get: () => "MacIntel"
      });
      Object.defineProperty(Navigator.prototype, "userAgent", {
        configurable: true,
        get: () => "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7)"
      });
    });
    await page.evaluate(() => window.localStorage.removeItem("omni.appearance.v1"));
    await page.reload();
    const root = page.locator("html");
    await expect(root).toHaveAttribute("data-appearance-mode", "system");
    await expect(root).toHaveAttribute("data-palette", "aqua");
    await expect(root).toHaveAttribute("data-layout", "glass");
    await expect(root).toHaveAttribute("data-appearance-pack", "liquid-glass");

    await page.evaluate(() => {
      window.localStorage.setItem(
        "omni.appearance.v1",
        JSON.stringify({ schemaVersion: 1, mode: "light", palette: "violet", layout: "standard" })
      );
    });
    await page.reload();
    await expect(root).toHaveAttribute("data-appearance-mode", "light");
    await expect(root).toHaveAttribute("data-palette", "violet");
    await expect(root).toHaveAttribute("data-layout", "standard");
    await expect(root).toHaveAttribute("data-appearance-pack", "standard");
  } finally {
    await appleApplication.close();
  }

  const nonAppleApplication = await launchAppearanceApplication();
  try {
    const page = await nonAppleApplication.firstWindow();
    await page.addInitScript(() => {
      Object.defineProperty(Navigator.prototype, "platform", {
        configurable: true,
        get: () => "Win32"
      });
      Object.defineProperty(Navigator.prototype, "userAgent", {
        configurable: true,
        get: () => "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
      });
    });
    await page.evaluate(() => window.localStorage.removeItem("omni.appearance.v1"));
    await page.reload();
    const root = page.locator("html");
    await expect(root).toHaveAttribute("data-appearance-mode", "system");
    await expect(root).toHaveAttribute("data-palette", "violet");
    await expect(root).toHaveAttribute("data-layout", "standard");
    await expect(root).toHaveAttribute("data-appearance-pack", "standard");
  } finally {
    await nonAppleApplication.close();
  }
});

test("all appearance packs meet representative contrast at desktop and mobile sizes", async () => {
  const application = await launchAppearanceApplication();
  const failures: Array<{ matrix: string; result: ContrastResult }> = [];
  const report: Record<string, ContrastResult[]> = {};
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const matrix = `${viewport.id}/${mode}/${pack.id}`;
          await selectAppearance(page, mode, pack.palette, pack.layout);
          await installAuditFixtures(page);
          await page.getByRole("button", { name: "Appearance settings" }).click();
          const panel = page.getByRole("dialog", { name: "Appearance settings" });
          await expect(panel).toBeVisible();
          const bounds = await panel.boundingBox();
          expect(bounds, `${matrix} appearance panel must have bounds`).not.toBeNull();
          expect(bounds!.x, `${matrix} appearance panel left edge`).toBeGreaterThanOrEqual(0);
          expect(bounds!.y, `${matrix} appearance panel top edge`).toBeGreaterThanOrEqual(0);
          expect(bounds!.x + bounds!.width, `${matrix} appearance panel right edge`).toBeLessThanOrEqual(viewport.width + 1);
          expect(bounds!.y + bounds!.height, `${matrix} appearance panel bottom edge`).toBeLessThanOrEqual(viewport.height + 1);
          const results = await measureContrast(page);
          report[matrix] = results;
          for (const result of results) {
            if (result.ratio < result.minimum) failures.push({ matrix, result });
          }
        }
      }
    }
  } finally {
    await application.close();
  }

  console.log(`THEME_CONTRAST_REPORT=${JSON.stringify(report)}`);
  expect(
    failures,
    failures
      .map(({ matrix, result }) =>
        `${matrix} ${result.id}: ${result.ratio}:1 < ${result.minimum}:1 ` +
        `(${result.foreground} on ${result.background}; opacity ${result.opacity})`
      )
      .join("\n")
  ).toEqual([]);
});

test("Brain Map labels, inspector copy, status, and controls meet contrast in every pack", async () => {
  const application = await launchAppearanceApplication();
  const failures: Array<{ matrix: string; result: ContrastResult }> = [];
  const report: Record<string, ContrastResult[]> = {};
  const targets: ContrastTarget[] = [
    {
      id: "graph-label",
      selector: ".graph-node text:last-of-type",
      minimum: 4.5,
      backgroundSelector: ".brain-graph",
      foregroundProperty: "fill"
    },
    {
      id: "graph-legend",
      selector: ".graph-legend span",
      minimum: 4.5,
      backgroundSelector: ".graph-surface"
    },
    {
      id: "graph-inspector-title",
      selector: ".map-inspector h2",
      minimum: 4.5,
      backgroundSelector: ".map-inspector"
    },
    {
      id: "graph-inspector-section",
      selector: ".map-inspector .panel-section__head span",
      minimum: 4.5,
      backgroundSelector: ".map-inspector"
    },
    {
      id: "graph-status",
      selector: ".graph-status span",
      minimum: 4.5,
      backgroundSelector: ".graph-surface"
    },
    {
      id: "graph-controls",
      selector: ".map-viewport-controls span",
      minimum: 4.5,
      backgroundSelector: ".map-viewport-controls"
    }
  ];

  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const matrix = `${viewport.id}/${mode}/${pack.id}`;
          await selectAppearance(page, mode, pack.palette, pack.layout);
          await page.getByRole("button", { name: "Brain map", exact: true }).click();
          await expect(page.locator(".map-page")).toBeVisible();
          const results = await measureContrast(page, targets);
          report[matrix] = results;
          for (const result of results) {
            if (result.ratio < result.minimum) failures.push({ matrix, result });
          }
        }
      }
    }
  } finally {
    await application.close();
  }

  console.log(`BRAIN_MAP_CONTRAST_REPORT=${JSON.stringify(report)}`);
  expect(
    failures,
    failures
      .map(({ matrix, result }) =>
        `${matrix} ${result.id}: ${result.ratio}:1 < ${result.minimum}:1 ` +
        `(${result.foreground} on ${result.background}; opacity ${result.opacity})`
      )
      .join("\n")
  ).toEqual([]);
});

test("layout packs produce distinct workspace topology, proportions, spacing, and typography", async () => {
  const application = await launchAppearanceApplication();
  try {
    const page = await application.firstWindow();
    await page.setViewportSize({ width: 1440, height: 900 });
    const signatures = new Map<string, Awaited<ReturnType<typeof layoutSignature>>>();
    for (const pack of packs) {
      await selectAppearance(page, "dark", pack.palette, pack.layout);
      await installAuditFixtures(page);
      signatures.set(pack.id, await layoutSignature(page));
    }

    const uniqueSignatures = new Set(
      [...signatures.values()].map((signature) => JSON.stringify(signature))
    );
    expect(
      uniqueSignatures.size,
      `Every advertised layout must be structurally distinct: ${JSON.stringify(Object.fromEntries(signatures))}`
    ).toBe(packs.length);

    const standard = signatures.get("standard")!;
    const classic = signatures.get("classic")!;
    const colorful = signatures.get("colorful")!;
    const glass = signatures.get("liquid-glass")!;

    expect(classic.surfaceRadius).not.toBe(standard.surfaceRadius);
    expect(colorful.surfaceRadius).not.toBe(standard.surfaceRadius);
    expect(glass.surfaceRadius).not.toBe(standard.surfaceRadius);
    expect(
      new Set([standard.railWidth, classic.railWidth, colorful.railWidth, glass.railWidth]).size,
      "Layout packs must change navigation/workspace proportions, not only paint"
    ).toBeGreaterThanOrEqual(3);
    expect(
      new Set([standard.messageInset, classic.messageInset, colorful.messageInset, glass.messageInset]).size,
      "Layout packs must change content placement/inset"
    ).toBeGreaterThanOrEqual(3);
    expect(
      new Set([standard.chatColumns, classic.chatColumns, colorful.chatColumns, glass.chatColumns]).size,
      "At least one layout must alter the conversation/inspector area topology"
    ).toBeGreaterThanOrEqual(2);
    expect(
      new Set([standard.cardGap, classic.cardGap, colorful.cardGap, glass.cardGap]).size,
      "Layout packs must change spatial rhythm"
    ).toBeGreaterThanOrEqual(3);
    expect(
      new Set([standard.surfaceFont, classic.surfaceFont, colorful.surfaceFont, glass.surfaceFont]).size,
      "At least one alternate layout must establish a distinct typography treatment"
    ).toBeGreaterThanOrEqual(2);
  } finally {
    await application.close();
  }
});

test("workspace layout gallery changes real chat and cortex placement", async () => {
  const application = await launchAppearanceApplication();
  try {
    const page = await application.firstWindow();
    await page.setViewportSize({ width: 1440, height: 900 });
    await selectAppearance(page, "dark", "violet", "standard");
    await page.getByRole("button", { name: "Appearance settings" }).click();

    const readPlacement = () => page.evaluate(() => {
      const conversation = document.querySelector(".conversation")!.getBoundingClientRect();
      const cortex = document.querySelector(".cortex-panel")!.getBoundingClientRect();
      const chat = getComputedStyle(document.querySelector(".chat-layout")!);
      return {
        columns: chat.gridTemplateColumns,
        rows: chat.gridTemplateRows,
        conversation: { x: conversation.x, y: conversation.y, width: conversation.width, height: conversation.height },
        cortex: { x: cortex.x, y: cortex.y, width: cortex.width, height: cortex.height },
        cortexDisplay: getComputedStyle(document.querySelector(".cortex-panel")!).display
      };
    });

    const choose = async (label: string, value: string) => {
      await page.locator(".appearance-layout-gallery button", { hasText: label }).click();
      await expect(page.locator("html")).toHaveAttribute("data-workspace-arrangement", value);
      return readPlacement();
    };

    const split = await choose("Side by side", "split");
    const focus = await choose("Full focus", "focus");
    const stacked = await choose("Top + below", "stacked");
    const mixed = await choose("Mixed canvas", "mixed");

    expect(split.conversation.x).toBeLessThan(split.cortex.x);
    expect(focus.cortexDisplay).toBe("none");
    expect(focus.conversation.width).toBeGreaterThan(split.conversation.width);
    expect(stacked.cortex.y).toBeGreaterThan(stacked.conversation.y);
    expect(stacked.rows).not.toBe(split.rows);
    expect(mixed.cortex.x).toBeLessThan(mixed.conversation.x);
    expect(mixed.columns).not.toBe(split.columns);
  } finally {
    await application.close();
  }
});
