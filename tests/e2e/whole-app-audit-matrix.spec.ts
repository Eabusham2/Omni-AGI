import { appendFileSync, createReadStream, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, readlinkSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createInterface } from "node:readline";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { _electron as electron, expect, test, type ElectronApplication, type Locator, type Page } from "@playwright/test";
import type { WholeAppCompositionControl } from "./whole-app-audit-contract";
// The Playwright ESM loader does not transform JSON imports. Read the exact
// manifest-bound source instead of relying on runtime import attributes.
const auditScope = JSON.parse(readFileSync(new URL("./whole-app-audit-scope.json", import.meta.url), "utf8")) as {
  id: string; routes: Array<{ id: string; label: string; selector: string }>;
  designStates: string[]; compositionStates: string[];
};
import { assertAuditResources, AuditResourcePauseError, type AuditResourceSnapshot } from "./whole-app-audit-resources.mjs";
import { appearanceAuditFailureMessage, createAppearanceAuditObservation, type AppearanceAuditObservations, type AppearanceAuditSnapshot } from "./appearance-audit-observation.mjs";
import { adjudicateLowParentPaint, measureVisibleControlInk, type ControlContentsPaint, type ControlPaintAdjudication } from "./control-paint-evidence";

/**
 * Missing U1 composition coverage, not a replacement for the closed 11 UI tests.
 * This mounts the production App through its explicit, no-preload preview seam.
 * Design-preview identities/trace/counters are synthetic. No brain, worker,
 * Load, training, provider, capture, permission or external effect is started.
 *
 * Default traversal is the full Cartesian product, not only the four paired
 * appearance packs: 4 palettes * 4 surfaces * 2 schemes * 4 arrangements *
 * 3 sizes = 384 cells. Five routes plus 12 design states give 6,528 core
 * visits; 25 typed event/composition states give 9,600 additional visits.
 * A shard/range is useful for bounded runs but never reports full coverage.
 *
 * Opt in with OMNI_WHOLE_APP_AUDIT=1; it is not added to existing CI here.
 * OMNI_WHOLE_APP_AUDIT_RANGE=0:48 (zero-based, end-exclusive cell range), or
 * OMNI_WHOLE_APP_AUDIT_SHARD=1/8 (one-based shard), optionally combined.
 * OMNI_WHOLE_APP_AUDIT_TIMEOUT_MS sets a finite execution budget, default 90m.
 * JSONL is flushed after each visit. An interrupted/failed/sharded traversal
 * remains incomplete. Screenshots are captures awaiting human inspection,
 * never automatically labelled accepted visual or accessibility evidence.
 */

const repository = resolve(process.cwd());
const palettes = ["violet", "graphite", "spectrum", "aqua"] as const;
const surfaces = ["standard", "classic", "expressive", "glass"] as const;
const schemes = ["light", "dark"] as const;
const arrangements = ["split", "focus", "stacked", "mixed"] as const;
const sizes = [
  { id: "compact", width: 390, height: 844 },
  { id: "normal", width: 1280, height: 800 },
  { id: "large", width: 1920, height: 1080 }
] as const;
const routes = auditScope.routes, designStates = auditScope.designStates;
const coreStates = [...routes.map(route => route.id), ...designStates];
type Cell = { index: number; key: string; palette: typeof palettes[number]; surface: typeof surfaces[number]; scheme: typeof schemes[number]; arrangement: typeof arrangements[number]; size: typeof sizes[number] };
type Issue = { kind: string; control?: string; detail: string };
type CaptureScrollPosition = { owner: string; top: number; maximum: number; reachedBottom: boolean };
type Capture = { path: string; purpose: string; visualReview: "not-yet-reviewed"; scrollPositions?: CaptureScrollPosition[] };
type StaticArtifactFile = { path: string; bytes: number; sha256: string };
type StaticFixtureManifest = {
  format: "omni-u1-static-fixture"; version: 1; repositoryRoot: string; sourceCommit: string;
  sourceFingerprint: string; sourceFiles: StaticArtifactFile[]; buildInputs: StaticArtifactFile[]; auditSpecSha256: string; fixtureSha256: string;
  productRendererBytesSha256: string; artifactBytesSha256: string; entryPath: string; files: StaticArtifactFile[]; noHmr: true;
  buildBase: "./";
  boundaries: { ipc: false; models: 0; weights: 0; create: 0; load: 0; training: 0; tools: 0; providers: 0; devices: 0; externalEffects: 0 };
};
type NoticeGeometry = {
  notice: { left: number; right: number; top: number; bottom: number };
  conversation: { left: number; right: number; top: number; bottom: number };
  composer: { left: number; right: number; top: number; bottom: number; scrollTop: number; scrollHeight: number; clientHeight: number; marginBottom: string; paddingBottom: string };
  reservation: string | null; measuredSpace: string;
};
type ControlSample = {
  id: string; name: string; tag: string; disabled: boolean; inert: boolean;
  width: number; height: number; clippedX: number; initialClippedX?: number; visible: boolean;
  color: string; background: string; opacity: string;
  contrast: number | null; contrastScope: "estimated-solid-composite" | "unmeasured-image-or-filter";
  contentsPaint?: ControlContentsPaint; contentsAdjudication?: ControlPaintAdjudication;
  basePaint?: ControlPaint; hoverPaint?: ControlPaint; focusPaint?: ControlPaint;
  focused: boolean; hovered: boolean; invocation: "not-invoked-no-host-authority";
};
type ControlPaint = Pick<ControlSample, "color" | "background" | "opacity" | "contrast" | "contrastScope" | "contentsPaint" | "contentsAdjudication">;
type Visit = {
  key: string; cellIndex: number; route: string; state: string; status: "visited" | "failed";
  startedAt: string; elapsedMs: number; controls: ControlSample[];
  scroll: Array<{ owner: string; top: number; bottom: number; maximum: number; sampledOffsets: number[]; reachedBottom: boolean;
    postKeyboardTop?: number; captureTop?: number; captureMaximum?: number; captureReachedBottom?: boolean }>;
  captures: Capture[]; issues: Issue[];
  noticeGeometry?: { before?: NoticeGeometry; after?: NoticeGeometry };
  appearanceLifecycle?: AppearanceAuditObservations;
};

const cells: Cell[] = [];
for (const palette of palettes) for (const surface of surfaces) for (const scheme of schemes)
  for (const arrangement of arrangements) for (const size of sizes) cells.push({
    index: cells.length, key: `${palette}-${surface}-${scheme}-${arrangement}-${size.id}`,
    palette, surface, scheme, arrangement, size
  });

function boundedInteger(value: string | undefined, fallback: number, label: string): number {
  if (value === undefined || value === "") return fallback;
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) throw new Error(`Invalid ${label}: ${value}`);
  return Number(value);
}

function selectedCells(): Cell[] {
  const rawRange = process.env.OMNI_WHOLE_APP_AUDIT_RANGE;
  const range = rawRange?.match(/^(\d+):(\d+)$/);
  if (rawRange && !range) throw new Error("OMNI_WHOLE_APP_AUDIT_RANGE must be start:end (end-exclusive).");
  const start = range ? Number(range[1]) : 0, end = range ? Number(range[2]) : cells.length;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start >= end || end > cells.length) throw new Error("Invalid whole-App audit range.");
  const rawShard = process.env.OMNI_WHOLE_APP_AUDIT_SHARD, shard = rawShard?.match(/^(\d+)\/(\d+)$/);
  if (rawShard && !shard) throw new Error("OMNI_WHOLE_APP_AUDIT_SHARD must be one-based index/total.");
  const index = shard ? Number(shard[1]) : 1, count = shard ? Number(shard[2]) : 1;
  if (![index, count].every(Number.isSafeInteger) || index < 1 || count < index || count > cells.length) throw new Error("Invalid whole-App audit shard.");
  const selected = cells.filter(cell => cell.index >= start && cell.index < end && cell.index % count === index - 1);
  if (!selected.length) throw new Error("The requested whole-App audit selection is empty.");
  return selected;
}

async function settledFrame(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>(resolveFrame => requestAnimationFrame(() => requestAnimationFrame(() => resolveFrame()))));
}

async function appearanceObservations(page: Page): Promise<AppearanceAuditObservations | undefined> {
  return page.evaluate(() => (window as Window & { __omniAppearanceAuditObservation?: { read(): AppearanceAuditObservations } })
    .__omniAppearanceAuditObservation?.read());
}

async function requireInspectedAppearance(page: Page, record: Visit, phase: string): Promise<void> {
  if (!record.state.startsWith("appearance-over-training-")) return;
  // Read only current geometry during each control step; serialize the full
  // bounded ring once at the end or failure, not on every paint/scroll sample.
  const snapshot = await page.evaluate(() => (window as Window & { __omniAppearanceAuditObservation?: { snapshot(): AppearanceAuditSnapshot | null } })
    .__omniAppearanceAuditObservation?.snapshot());
  if (!snapshot) throw new Error(`Appearance lifecycle observer is missing during ${phase}; this state cannot be qualified.`);
  const failure = appearanceAuditFailureMessage(snapshot, phase);
  if (failure) { record.appearanceLifecycle = await appearanceObservations(page); throw new Error(failure); }
}

async function configureCell(page: Page, cell: Cell): Promise<void> {
  await page.setViewportSize({ width: cell.size.width, height: cell.size.height });
  await expect.poll(() => page.evaluate(() => ({ width: innerWidth, height: innerHeight }))).toEqual({ width: cell.size.width, height: cell.size.height });
  await page.evaluate(value => {
    localStorage.setItem("omni.appearance.v1", JSON.stringify({ schemaVersion: 1, mode: value.scheme, palette: value.palette, layout: value.surface }));
    localStorage.setItem("omni.appearance.workspace.v1", JSON.stringify({ schemaVersion: 1, arrangement: value.arrangement, notificationsEnabled: false, profiles: [] }));
  }, cell);
  await page.reload();
  await expect(page.locator(".app-shell")).toBeVisible();
  for (const [attribute, value] of Object.entries({ "data-color-scheme": cell.scheme, "data-palette": cell.palette, "data-layout": cell.surface, "data-workspace-arrangement": cell.arrangement })) {
    await expect(page.locator("html")).toHaveAttribute(attribute, value);
  }
  expect(await page.evaluate(() => typeof (window as Window & { readonly omni?: unknown }).omni)).toBe("undefined");
  await expect(page.locator(".library-page")).toBeVisible();
}

/** Real computed styles, not inserted look-alike controls. No CSS is changed. */
async function sampleControls(root: Locator, onlyId?: string): Promise<ControlSample[]> {
  const samples: ControlSample[] = await root.evaluate((element, selectedId) => {
    const query = 'button, input:not([type="hidden"]), select, textarea, summary, a[href], [role="button"], [role="switch"], [role="tab"], [role="checkbox"]';
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 1;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const rgba = (color: string): number[] => {
      if (!context) return [0, 0, 0, 0];
      context.clearRect(0, 0, 1, 1); context.fillStyle = color; context.fillRect(0, 0, 1, 1);
      return [...context.getImageData(0, 0, 1, 1).data].map((value, index) => index === 3 ? value / 255 : value);
    };
    const over = (front: number[], back: number[]): number[] => {
      const alpha = front[3]! + back[3]! * (1 - front[3]!);
      return [0, 1, 2].map(index => alpha === 0 ? 0 : (front[index]! * front[3]! + back[index]! * back[3]! * (1 - front[3]!)) / alpha).concat(alpha);
    };
    const luminance = (rgb: number[]): number => rgb.slice(0, 3).map(value => {
      const unit = value / 255; return unit <= .04045 ? unit / 12.92 : ((unit + .055) / 1.055) ** 2.4;
    }).reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index]!, 0);
    const nodes = [...element.querySelectorAll<HTMLElement>(query)].map(node => {
      node.dataset.wholeAppAuditControl ??= crypto.randomUUID(); return node;
    }).filter(node => selectedId === undefined || node.dataset.wholeAppAuditControl === selectedId);
    return nodes.map(node => {
      const bounds = node.getBoundingClientRect(), style = getComputedStyle(node);
      const ancestors: HTMLElement[] = []; let current: HTMLElement | null = node;
      while (current) { ancestors.unshift(current); current = current.parentElement; }
      let background = [255, 255, 255, 1], opacity = 1, uncertainPaint = false, uncertainEffect = false, hiddenByDisclosure = false;
      let left = 0, right = innerWidth, top = 0, bottom = innerHeight;
      for (const ancestor of ancestors) {
        const ancestorStyle = getComputedStyle(ancestor), box = ancestor.getBoundingClientRect();
        if (ancestor instanceof HTMLDetailsElement && !ancestor.open && !ancestor.querySelector(":scope > summary")?.contains(node)) hiddenByDisclosure = true;
        if (ancestorStyle.contentVisibility === "hidden") hiddenByDisclosure = true;
        const paint = rgba(ancestorStyle.backgroundColor);
        // An opaque solid child actually covers a background image behind it.
        // Filters/backdrop effects remain explicitly unmeasured.
        if (paint[3] === 1 && ancestorStyle.backgroundImage === "none") uncertainPaint = false;
        background = over(paint, background);
        opacity *= Number(ancestorStyle.opacity);
        uncertainPaint ||= ancestorStyle.backgroundImage !== "none";
        uncertainEffect ||= ancestorStyle.backdropFilter !== "none" || ancestorStyle.filter !== "none" || Number(ancestorStyle.opacity) !== 1;
        if (ancestor !== node) {
          if (/hidden|clip|auto|scroll/.test(ancestorStyle.overflowX)) { left = Math.max(left, box.left); right = Math.min(right, box.right); }
          if (/hidden|clip|auto|scroll/.test(ancestorStyle.overflowY)) { top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom); }
        }
      }
      const ink = rgba(style.color); ink[3] = ink[3]! * opacity;
      const foreground = over(ink, background), fg = luminance(foreground), bg = luminance(background);
      const labels = "labels" in node ? [...((node as HTMLInputElement).labels ?? [])].map(label => label.textContent ?? "").join(" ") : "";
      const labelledBy = node.getAttribute("aria-labelledby")?.split(/\s+/).map(id => document.getElementById(id)?.textContent ?? "").join(" ");
      const name = (node.getAttribute("aria-label") || labelledBy || labels || node.getAttribute("title") || node.textContent || "").replace(/\s+/g, " ").trim();
      const visible = !hiddenByDisclosure && bounds.width > 0 && bounds.height > 0 && style.visibility !== "hidden" && style.display !== "none" && opacity > 0 && bounds.right > left && bounds.left < right && bounds.bottom > top && bounds.top < bottom;
      return { id: node.dataset.wholeAppAuditControl!, name, tag: node.tagName.toLowerCase(), disabled: node.matches(":disabled") || node.getAttribute("aria-disabled") === "true",
        inert: Boolean(node.closest("[inert]")), width: bounds.width, height: bounds.height,
        clippedX: Math.max(0, left - bounds.left) + Math.max(0, bounds.right - right), visible,
        color: style.color, background: `rgba(${background.join(",")})`, opacity: String(opacity),
        contrast: uncertainPaint || uncertainEffect ? null : Math.round(((Math.max(fg, bg) + .05) / (Math.min(fg, bg) + .05)) * 1000) / 1000,
        contrastScope: uncertainPaint || uncertainEffect ? "unmeasured-image-or-filter" as const : "estimated-solid-composite" as const,
        focused: document.activeElement === node, hovered: node.matches(":hover"), invocation: "not-invoked-no-host-authority" as const };
    });
  }, onlyId);
  // Successful old measurements take precisely the old semantic path. Only
  // a low parent-color estimate needs actual descendant text/icon evidence.
  for (const sample of samples) if (sample.visible && !sample.disabled && !sample.inert && sample.contrast !== null && sample.contrast < 4.5) {
    try {
      sample.contentsPaint = await root.locator(`[data-whole-app-audit-control="${sample.id}"]`).evaluate(measureVisibleControlInk);
    } catch (error) {
      sample.contentsPaint = { format: "visible-control-ink-v1", complete: false, textRuns: [], graphics: [],
        unmeasured: [`Actual visible paint measurement failed: ${error instanceof Error ? error.message : String(error)}`] };
    }
  }
  return samples;
}

async function inspectPresentControls(root: Locator, page: Page, record: Visit, seen: Set<string>): Promise<void> {
  await requireInspectedAppearance(page, record, "control scan");
  const paintOf = (value: ControlSample): ControlPaint => ({ color: value.color, background: value.background,
    opacity: value.opacity, contrast: value.contrast, contrastScope: value.contrastScope,
    ...(value.contentsPaint ? { contentsPaint: value.contentsPaint } : {}) });
  const checkPaint = (sample: ControlSample, phase: string, paint: ControlPaint) => {
    if (!sample.disabled && !sample.inert && paint.contrast !== null && paint.contrast < 4.5) {
      paint.contentsAdjudication = adjudicateLowParentPaint(paint.contrast, paint.contentsPaint);
      if (paint.contentsAdjudication.qualified) return;
      record.issues.push({ kind: `contrast-${phase}`, control: sample.name || sample.id,
        detail: `${phase}: original parent ${paint.contrast.toFixed(3)}:1 below 4.5:1; ${paint.color} on ${paint.background}. Solid-composite estimate only. ${paint.contentsAdjudication.failures.join(" ")}` });
    }
  };
  const samples = await sampleControls(root);
  for (const sample of samples) {
    if (!sample.visible || seen.has(sample.id)) continue;
    seen.add(sample.id);
    const control = root.locator(`[data-whole-app-audit-control="${sample.id}"]`);
    sample.initialClippedX = sample.clippedX;
    if (!sample.name && !["input", "select", "textarea"].includes(sample.tag)) record.issues.push({ kind: "unnamed-control", control: sample.id, detail: `Visible ${sample.tag} has no inspected name` });
    try {
      // Preserve real unhovered/unfocused ink. A later readable hover must not
      // overwrite evidence of a low-contrast resting state (e.g. Delete).
      await page.mouse.move(1, 1);
      await requireInspectedAppearance(page, record, "before control base paint");
      await control.evaluate(node => { (node.ownerDocument.activeElement as HTMLElement | null)?.blur?.(); });
      await settledFrame(page);
      await requireInspectedAppearance(page, record, "control base paint");
      const [base] = await sampleControls(root, sample.id);
      if (!base) throw new Error("The control disappeared before its base paint was recorded.");
      sample.basePaint = paintOf(base); checkPaint(sample, "base", sample.basePaint);
      if (sample.disabled || sample.inert) {
        // Pointer movement can inspect disabled :hover paint without waiting
        // for an intentionally non-actionable/inert control to become enabled.
        const bounds = await control.boundingBox();
        if (bounds) await page.mouse.move(Math.max(0, Math.min((await page.viewportSize())!.width - 1, bounds.x + bounds.width / 2)), Math.max(0, Math.min((await page.viewportSize())!.height - 1, bounds.y + bounds.height / 2)));
      } else await control.hover({ timeout: 1500 });
      await settledFrame(page);
      await requireInspectedAppearance(page, record, "control hover paint");
      const [hovered] = await sampleControls(root, sample.id);
      if (!hovered) throw new Error("The control disappeared before its hover paint was recorded.");
      sample.hovered = hovered.hovered;
      if (!sample.disabled && !sample.inert && !sample.hovered) record.issues.push({ kind: "hover-not-received", control: sample.name || sample.id,
        detail: "The pointer action completed, but the control did not retain :hover when its paint was sampled" });
      sample.hoverPaint = paintOf(hovered); checkPaint(sample, "hover", sample.hoverPaint);
      if (!sample.disabled && !sample.inert) {
        await page.mouse.move(1, 1);
        await control.focus({ timeout: 1500 });
        sample.focused = await control.evaluate(node => document.activeElement === node);
        if (!sample.focused) record.issues.push({ kind: "focus-not-received", control: sample.name || sample.id, detail: "Programmatic focus was not received; keyboard order is a separate check" });
        await settledFrame(page);
        await requireInspectedAppearance(page, record, "control focus paint");
      }
      const [after] = await sampleControls(root, sample.id);
      if (after) {
        if (!sample.disabled && !sample.inert) { sample.focusPaint = paintOf(after); checkPaint(sample, "focus", sample.focusPaint); }
        sample.clippedX = after.clippedX;
        sample.color = after.color; sample.background = after.background; sample.contrast = after.contrast; sample.contrastScope = after.contrastScope;
        if (after.clippedX > 2) record.issues.push({ kind: "hover-focus-clipping", control: sample.name || sample.id, detail: `${after.clippedX.toFixed(2)} px clipped after hover/focus` });
      }
    } catch (error) {
      // A disappeared popup is a failed state, not a missing control to skip.
      // Stop this walk immediately; the caller persists its lifecycle record
      // and failure capture. Never reopen/retry to turn it into a green row.
      await requireInspectedAppearance(page, record, "control interaction failure");
      // Inert/disabled controls have no pointer/focus promise. Their actual
      // geometry/ink remains in the journal; don't pretend they were invoked.
      if (!sample.disabled && !sample.inert) record.issues.push({ kind: "hover-focus-unreachable", control: sample.name || sample.id, detail: error instanceof Error ? error.message.split("\n")[0]! : String(error) });
    }
    record.controls.push(sample);
  }
  // Do not call tool/form handlers to make a preview look operational.
  await page.mouse.move(1, 1);
}

async function walkContent(page: Page, root: Locator, record: Visit): Promise<void> {
  await requireInspectedAppearance(page, record, "content walk start");
  await root.evaluate(element => {
    // Opening local disclosures inspects existing content only, not effects.
    for (const detail of element.querySelectorAll<HTMLDetailsElement>("details")) {
      // Absolute action menus are transient, not document disclosures. Leave
      // their state to the explicit menu scenario so they cannot cover cards
      // while the ordinary Home selection state is inspected.
      if (!detail.matches(".library-export-menu,.plugin-runtime-menu")) detail.open = true;
    }
  });
  await settledFrame(page);
  const owners = await root.evaluate(element => {
    for (const previous of document.querySelectorAll<HTMLElement>("[data-whole-app-audit-scroll]")) delete previous.dataset.wholeAppAuditScroll;
    const all = [element, ...element.querySelectorAll<HTMLElement>("*"), document.scrollingElement].filter((value): value is HTMLElement => value instanceof HTMLElement);
    const unique = [...new Set(all)].filter(node => node.scrollHeight > node.clientHeight + 2 && node.clientHeight > 0 && (node === document.scrollingElement || /auto|scroll/.test(getComputedStyle(node).overflowY)));
    // Include scrollable ancestors of the route, not unrelated hidden pages.
    let ancestor = element.parentElement;
    while (ancestor) { if (ancestor.clientHeight > 0 && ancestor.scrollHeight > ancestor.clientHeight + 2 && /auto|scroll/.test(getComputedStyle(ancestor).overflowY) && !unique.includes(ancestor)) unique.push(ancestor); ancestor = ancestor.parentElement; }
    return unique.map((node, index) => { node.dataset.wholeAppAuditScroll = String(index); node.scrollTop = 0; return String(index); });
  });
  const seen = new Set<string>();
  await inspectPresentControls(root, page, record, seen);
  for (const owner of owners) {
    await requireInspectedAppearance(page, record, "scroll owner admission");
    const locator = page.locator(`[data-whole-app-audit-scroll="${owner}"]`);
    await locator.evaluate(node => { node.scrollIntoView({ block: "nearest", inline: "nearest" }); node.scrollTop = 0; });
    const offsets: number[] = []; let previous = -1;
    for (;;) {
      await requireInspectedAppearance(page, record, "scroll step");
      const point = await locator.evaluate(node => ({ top: node.scrollTop, max: Math.max(0, node.scrollHeight - node.clientHeight), step: Math.max(1, Math.floor(node.clientHeight * .8)) }));
      offsets.push(point.top);
      await inspectPresentControls(root, page, record, seen);
      if (point.top >= point.max - 2 || point.top === previous) break;
      previous = point.top;
      await locator.evaluate((node, next) => { node.scrollTop = next; }, Math.min(point.max, point.top + point.step));
      await settledFrame(page);
    }
    const final = await locator.evaluate(node => ({ top: node.scrollTop, max: Math.max(0, node.scrollHeight - node.clientHeight) }));
    const reachedBottom = final.top >= final.max - 2;
    record.scroll.push({ owner, top: offsets[0] ?? 0, bottom: final.top, maximum: final.max, sampledOffsets: offsets, reachedBottom });
    if (!reachedBottom) record.issues.push({ kind: "scroll-bottom-unreached", detail: `Owner ${owner}: ${final.top} of ${final.max}` });
  }
  await page.keyboard.press("Tab");
  await requireInspectedAppearance(page, record, "keyboard traversal");
  const keyboard = await page.evaluate(() => ({ tag: document.activeElement?.tagName, name: document.activeElement?.getAttribute("aria-label"), body: document.activeElement === document.body }));
  if (keyboard.body) record.issues.push({ kind: "keyboard-focus-lost", detail: "Tab left focus on document.body" });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 1) record.issues.push({ kind: "document-horizontal-overflow", detail: `${overflow} px` });
}

async function capture(page: Page, directory: string, record: Visit, suffix: string): Promise<void> {
  const path = join(directory, `${record.key}-${suffix}.jpg`);
  await page.screenshot({ path, type: "jpeg", quality: 78, animations: "disabled" });
  record.captures.push({ path, purpose: suffix, visualReview: "not-yet-reviewed",
    ...(suffix === "lower-content" ? { scrollPositions: record.scroll.map(owner => ({ owner: owner.owner,
      top: owner.captureTop!, maximum: owner.captureMaximum!, reachedBottom: owner.captureReachedBottom === true })) } : {}) });
}

async function restoreLowerContentAfterKeyboard(page: Page, record: Visit): Promise<void> {
  // Tab is a real keyboard check above and may reveal an earlier focused
  // control. Restore measured owners afterward; a filename is not evidence
  // that the captured viewport still shows their bottom content.
  const restored = await page.evaluate(owners => owners.map(owner => {
    const node = document.querySelector<HTMLElement>(`[data-whole-app-audit-scroll="${owner}"]`);
    if (!node) return { owner, missing: true as const };
    const postKeyboardTop = node.scrollTop;
    node.scrollTop = Math.max(0, node.scrollHeight - node.clientHeight);
    return { owner, missing: false as const, postKeyboardTop };
  }), record.scroll.map(owner => owner.owner));
  await settledFrame(page);
  const measured = await page.evaluate(owners => owners.map(owner => {
    const node = document.querySelector<HTMLElement>(`[data-whole-app-audit-scroll="${owner}"]`);
    if (!node) return { owner, missing: true as const };
    const top = node.scrollTop, maximum = Math.max(0, node.scrollHeight - node.clientHeight);
    return { owner, missing: false as const, top, maximum, reachedBottom: top >= maximum - 2 };
  }), record.scroll.map(owner => owner.owner));
  for (const owner of record.scroll) {
    const before = restored.find(value => value.owner === owner.owner), after = measured.find(value => value.owner === owner.owner);
    if (!before || before.missing || !after || after.missing) {
      record.issues.push({ kind: "capture-scroll-owner-missing", detail: `Measured owner ${owner.owner} disappeared before lower-content capture.` });
      continue;
    }
    owner.postKeyboardTop = before.postKeyboardTop;
    owner.captureTop = after.top; owner.captureMaximum = after.maximum; owner.captureReachedBottom = after.reachedBottom;
    if (!after.reachedBottom) record.issues.push({ kind: "capture-scroll-bottom-unreached", detail: `Owner ${owner.owner}: capture at ${after.top} of ${after.maximum}.` });
  }
}

function representative(cell: Cell): boolean {
  // Every independent palette/surface/mode/arrangement at normal size, plus
  // compact/large paired-pack references. These remain unreviewed captures;
  // the other metric visits are not silently called visually accepted.
  return cell.size.id === "normal" || cell.arrangement === "split" && surfaces[palettes.indexOf(cell.palette)] === cell.surface;
}

function atomicJson(path: string, value: unknown): void {
  const temporary = `${path}.pending`;
  writeFileSync(temporary, JSON.stringify(value, null, 2)); renameSync(temporary, path);
}

function fileInventory(sourceDirectories: string[]): Array<{ path: string; bytes: number; sha256: string }> {
  const files: Array<{ path: string; bytes: number; sha256: string }> = [];
  const visit = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) visit(path);
      else if (entry.isFile()) { const bytes = readFileSync(path); files.push({ path: path.slice(repository.length + 1), bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }); }
      else throw new Error(`Audit source inventory refuses an unsupported filesystem entry: ${path}`);
    }
  };
  for (const directory of sourceDirectories) visit(directory);
  return files;
}

function rendererFingerprint(sourceDirectories = [join(repository, "out", "renderer")]): string {
  return createHash("sha256").update(JSON.stringify(fileInventory(sourceDirectories))).digest("hex");
}
function loopbackPreloadInventory() {
  const supplied = process.env.OMNI_AUDIT_LOOPBACK_PRELOAD;
  if (!supplied) return null;
  const path = resolve(supplied);
  if (path !== resolve(repository, "work/audit-loopback-bind.cjs") || !existsSync(path) || lstatSync(path).isSymbolicLink()) throw new Error("Only the explicit owned loopback CLI preload may be recorded.");
  return { path, sha256: createHash("sha256").update(readFileSync(path)).digest("hex"), scope: "test CLI loopback binding only; no OS or product change" };
}
function admitAuditCell(output: string, label: string) {
  const pending = join(output, "RESOURCE_PAUSE.json");
  if (existsSync(pending)) throw new AuditResourcePauseError(JSON.parse(readFileSync(pending, "utf8")) as AuditResourceSnapshot);
  return assertAuditResources(repository, output, label);
}

async function verifyStaticFixture(url: URL, expected: { sourceCommit: string; sourceFingerprint: string; fixtureSha256: string }) {
  const localPath = process.env.OMNI_WHOLE_APP_STATIC_MANIFEST;
  if (!localPath) throw new Error("OMNI_WHOLE_APP_STATIC_MANIFEST must name the owned compiled-fixture manifest; dev/HMR servers are not accepted.");
  const manifestPath = resolve(localPath), within = relative(join(repository, ".runtime"), manifestPath);
  if (!within || isAbsolute(within) || within === ".." || within.startsWith(`..${sep}`) || lstatSync(manifestPath).isSymbolicLink()) throw new Error("Static fixture manifest must be an owned regular runtime file in this exact checkout.");
  const manifestBytes = readFileSync(manifestPath), manifest = JSON.parse(manifestBytes.toString("utf8")) as StaticFixtureManifest;
  expect(manifest.format).toBe("omni-u1-static-fixture"); expect(manifest.version).toBe(1);
  expect(manifest.repositoryRoot).toBe(repository); expect(manifest.sourceCommit).toBe(expected.sourceCommit);
  expect(manifest.sourceFingerprint).toBe(expected.sourceFingerprint); expect(manifest.fixtureSha256).toBe(expected.fixtureSha256);
  expect(manifest.sourceFiles).toEqual(fileInventory([join(repository, "src/renderer"), join(repository, "src/shared")]));
  expect(manifest.buildInputs).toEqual(["tests/e2e/whole-app-state-composition-fixture.html", "tests/e2e/whole-app-state-composition-fixture.tsx",
    "tests/e2e/whole-app-audit-contract.ts", "tests/e2e/whole-app-audit-scope.json", "tests/e2e/whole-app-audit-resources.mjs", "tests/e2e/whole-app-audit-resources.d.mts",
    "tests/e2e/appearance-audit-observation.mjs", "tests/e2e/appearance-audit-observation.d.mts",
    "tests/e2e/whole-app-audit-matrix.spec.ts", "tests/e2e/control-paint-evidence.ts", "tests/e2e/responsive-main.cjs", "package.json", "package-lock.json", "work/u1-static-fixture.mjs"]
    .map(path => { const bytes = readFileSync(resolve(repository, path)); return { path, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }; }));
  expect(manifest.auditSpecSha256).toBe(createHash("sha256").update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-matrix.spec.ts"))).digest("hex"));
  expect(manifest.productRendererBytesSha256).toBe(rendererFingerprint()); expect(manifest.noHmr).toBe(true);
  expect(manifest.boundaries).toEqual({ ipc: false, models: 0, weights: 0, create: 0, load: 0, training: 0, tools: 0, providers: 0, devices: 0, externalEffects: 0 });
  expect(url.search).toBe(""); expect(url.hash).toBe(""); expect(manifest.buildBase).toBe("./");
  const site = join(resolve(manifestPath, ".."), "site");
  if (url.protocol === "file:") expect(url.href).toBe(pathToFileURL(resolve(site, manifest.entryPath.slice(1))).href);
  else expect(manifest.entryPath).toBe(url.pathname);
  if (!Array.isArray(manifest.files) || !manifest.files.length) throw new Error("Static fixture artifact inventory is empty.");
  const seen = new Set<string>();
  for (const file of manifest.files) {
    if (!file.path || isAbsolute(file.path) || file.path.includes("\\") || file.path.split("/").some(part => !part || part === "." || part === "..") ||
        seen.has(file.path) || !Number.isSafeInteger(file.bytes) || file.bytes < 0 || !/^[a-f0-9]{64}$/.test(file.sha256)) throw new Error("Invalid/ambiguous static artifact inventory path/hash.");
    seen.add(file.path);
  }
  expect(seen.has(manifest.entryPath.slice(1))).toBe(true);
  expect(manifest.artifactBytesSha256).toBe(createHash("sha256").update(JSON.stringify(manifest.files)).digest("hex"));
  const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
  const manifestUrl = url.protocol === "file:" ? pathToFileURL(manifestPath).href : new URL("/__omni_u1_manifest.json", url.origin).href;
  if (url.protocol === "file:") {
    expect(fileInventory([site]).map(file => ({ ...file, path: relative(site, resolve(repository, file.path)) }))).toEqual(manifest.files);
    return { manifestPath, manifestUrl, manifestSha256, manifest, site, transport: "owned-file-static" as const };
  }
  const fetchBytes = async (href: string, bytes: number, sha256: string): Promise<Buffer> => {
    const response = await fetch(href, { redirect: "error", signal: AbortSignal.timeout(5000) });
    expect(response.status, href).toBe(200); expect(response.headers.get("x-omni-u1-sha256"), href).toBe(sha256);
    expect(response.headers.get("content-length"), href).toBe(String(bytes));
    const body = Buffer.from(await response.arrayBuffer());
    expect(body.byteLength, href).toBe(bytes); expect(createHash("sha256").update(body).digest("hex"), href).toBe(sha256);
    return body;
  };
  expect(await fetchBytes(manifestUrl, manifestBytes.length, manifestSha256)).toEqual(manifestBytes);
  for (const file of manifest.files) {
    const body = await fetchBytes(new URL(`/${file.path}`, url.origin).href, file.bytes, file.sha256);
    if (`/${file.path}` === manifest.entryPath && body.toString("utf8").includes("/@vite/client")) throw new Error("Dev/HMR fixture response is not an immutable static artifact.");
  }
  return { manifestPath, manifestUrl, manifestSha256, manifest, site, transport: "owned-loopback-http" as const };
}

function dirtyFileInventory(): Array<{ status: string; path: string; previousPath?: string; sha256?: string; removed?: true; symlinkTarget?: string }> {
  const raw = execFileSync("git", ["status", "--porcelain=v1", "-z", "--untracked-files=normal"], { cwd: repository, encoding: "utf8" }).split("\0").filter(Boolean);
  const result: ReturnType<typeof dirtyFileInventory> = [];
  for (let index = 0; index < raw.length; index += 1) {
    const entry = raw[index]!, status = entry.slice(0, 2), path = entry.slice(3), absolute = resolve(repository, path);
    const previousPath = /R|C/.test(status) ? raw[++index] : undefined;
    const within = relative(repository, absolute);
    if (!within || isAbsolute(within) || within === ".." || within.startsWith(`..${sep}`)) throw new Error("Dirty source path escaped the audited checkout.");
    if (!existsSync(absolute)) { result.push({ status, path, previousPath, removed: true }); continue; }
    if (lstatSync(absolute).isSymbolicLink()) { result.push({ status, path, previousPath, symlinkTarget: readlinkSync(absolute) }); continue; }
    if (lstatSync(absolute).isFile()) result.push({ status, path, previousPath, sha256: createHash("sha256").update(readFileSync(absolute)).digest("hex") });
    else result.push({ status, path, previousPath });
  }
  return result;
}

async function journalRows(path: string, requiredKeys: Set<string>): Promise<Map<string, Visit>> {
  const rows = new Map<string, Visit>();
  if (!existsSync(path)) return rows;
  const input = createReadStream(path, { encoding: "utf8" }), lines = createInterface({ input, crlfDelay: Infinity });
  try {
    for await (const line of lines) {
      if (!line) continue;
      const value = JSON.parse(line) as Visit;
      if (!requiredKeys.has(value.key) || !Number.isInteger(value.cellIndex) || !Array.isArray(value.issues)) throw new Error("Invalid existing whole-App audit journal; preserve it and choose a new output directory.");
      // Last exact attempt wins; older failed rows remain in append-only data.
      rows.set(value.key, value);
    }
  } finally { lines.close(); input.destroy(); }
  return rows;
}

const designSelectors: Record<string, string> = {
  "home-default": ".library-page", "home-selection": ".library-page", "home-selected-search": ".library-page",
  "home-export-menu": ".library-selection-toolbar", "home-delete-confirmation": ".instance-delete-dialog",
  "build-identity": ".builder-page", "build-initial-learning": ".builder-page", "build-source-menu": ".universal-source-picker__menu:popover-open",
  "build-memory-storage": ".builder-page", "build-access-review": ".builder-page",
  "training-expanded-form": ".training-evolution-page", "settings-custom-budgets": ".resource-settings-page"
};
async function openDesignBrain(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Open brain Aster", exact: true }).click();
  await expect(page.getByLabel("Message Aster")).toBeVisible();
}
async function workspaceRoute(page: Page, label: string): Promise<void> {
  await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: label, exact: true }).click();
  await expect(page.locator(".workspace-header__view")).toHaveText(label);
}
/** Only ordinary local navigation/drafts. Never Create, save, import/export, delete or run. */
async function prepareDesignState(page: Page, state: string): Promise<void> {
  if (state === "home-selection") {
    await page.getByRole("button", { name: "Select", exact: true }).click();
    await page.getByRole("button", { name: "Select all", exact: true }).click();
    await expect(page.getByRole("toolbar", { name: "Manage selected brains" })).toBeVisible();
    expect(await page.getByRole("checkbox", { name: /^Select brain / }).count()).toBeGreaterThan(1);
  } else if (state === "home-selected-search") {
    await page.getByRole("textbox", { name: "Search brains" }).fill("Aster");
    await expect(page.locator(".library-selection-count")).toContainText("outside this search");
  } else if (state === "home-export-menu") {
    if (await page.locator(".library-export-menu").getAttribute("open") === null) await page.locator(".library-export-menu > summary").click();
    await expect(page.getByRole("group", { name: "Brain export options" })).toBeVisible();
  } else if (state === "home-delete-confirmation") {
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(page.locator(".instance-delete-dialog")).toContainText("Confirmation 1 of 3");
  } else if (state === "build-identity") {
    if (await page.locator(".instance-delete-dialog").isVisible()) await page.locator(".instance-delete-dialog").getByRole("button", { name: "Cancel", exact: true }).click();
    await page.getByRole("button", { name: "Done", exact: true }).click();
    await page.getByRole("button", { name: "Build a new brain", exact: true }).click();
  } else if (state === "build-initial-learning") {
    await page.getByRole("button", { name: /^Step 2:/ }).click();
    await expect(page.getByRole("heading", { name: "Initial learning", exact: true })).toBeVisible();
  } else if (state === "build-source-menu") {
    await page.getByRole("button", { name: "Add source", exact: true }).click();
  } else if (state === "build-memory-storage") {
    await page.keyboard.press("Escape"); await page.getByRole("button", { name: /^Step 3:/ }).click();
    await expect(page.getByRole("heading", { name: "Memory & storage", exact: true })).toBeVisible();
  } else if (state === "build-access-review") {
    await page.getByRole("button", { name: /^Step 4:/ }).click();
    await page.getByRole("checkbox", { name: /^Full Authority/ }).check();
    await expect(page.getByRole("button", { name: /^Create Nova/ })).toBeVisible();
  } else if (state === "training-expanded-form") {
    await workspaceRoute(page, "Training & Evolution");
    await page.getByLabel("Text to learn", { exact: true }).fill("UI-only draft. No source ingestion or learning is requested.");
    await page.getByLabel("Instructions or improvement goal", { exact: true }).fill("UI-only improvement draft; do not submit.");
    const disclosure = page.locator(".learning-workspace details").filter({ has: page.locator("summary", { hasText: "Web scope and dates" }) }).first();
    if (await disclosure.getAttribute("open") === null) await disclosure.locator(":scope > summary").click();
    await expect(page.getByLabel("Only these sites · optional", { exact: true })).toBeVisible();
    await expect(page.locator(".training-evolution-sidebar")).toBeVisible();
    await expect(page.locator(".learning-workspace__form").getByRole("button", { name: "Learn", exact: true })).toBeDisabled();
  } else if (state === "settings-custom-budgets") {
    await workspaceRoute(page, "Settings");
    await page.getByRole("group", { name: "Omni-wide RAM policy" }).getByRole("button", { name: /^Advanced cap/ }).click();
    await page.getByRole("group", { name: "Working-memory size" }).getByRole("button", { name: /^Manual/ }).click();
    await page.getByRole("group", { name: "Shared-storage policy" }).getByRole("button", { name: /^Custom size/ }).click();
    await page.getByLabel("Omni share of safe pool", { exact: false }).fill("60");
    await page.getByLabel("Working-memory tokens", { exact: true }).fill("32768");
    await page.getByLabel("Shared pool capacity", { exact: true }).fill("24");
    await expect(page.getByText("Unsaved changes", { exact: true })).toBeVisible();
  }
}

test("U1 full whole-App stock appearance/workspace matrix records exact no-engine coverage", async ({}, testInfo) => {
  test.skip(process.env.OMNI_WHOLE_APP_AUDIT !== "1", "Explicit opt-in required for the full serial 384-cell U1 audit.");
  const timeout = boundedInteger(process.env.OMNI_WHOLE_APP_AUDIT_TIMEOUT_MS, 90 * 60_000, "whole-App audit timeout");
  if (timeout < 30_000 || timeout > 12 * 60 * 60_000) throw new Error("Whole-App audit timeout must be 30 seconds to 12 hours.");
  test.setTimeout(timeout);
  const selected = selectedCells();
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8" }).trim();
  const renderedBytesSha256 = rendererFingerprint();
  const renderedFiles = fileInventory([join(repository, "out", "renderer")]), dirtyFiles = dirtyFileInventory();
  const auditSpecSha256 = createHash("sha256").update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-matrix.spec.ts"))).digest("hex");
  const loopbackPreload = loopbackPreloadInventory();
  // This serial journal is outside Playwright's auto-cleared output directory.
  // Completed exact-byte visits can resume; new renderer bytes never inherit
  // prior coverage. Do not run parallel writers against the same output path.
  const directory = process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR
    ? resolve(process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR)
    : join(repository, ".runtime", "whole-app-ui-audit", sourceCommit, renderedBytesSha256, auditSpecSha256);
  const screenshots = join(directory, "captures");
  mkdirSync(screenshots, { recursive: true });
  const profile = mkdtempSync(join(tmpdir(), "omni-whole-app-audit-"));
  const journal = join(directory, "visits.jsonl"), metadata = join(directory, "coverage.json");
  const visits: Array<Pick<Visit, "key" | "status" | "route" | "cellIndex">> = [];
  const failures: Array<{ key: string; issues: Issue[] }> = [];
  const prohibitedRequests: string[] = [];
  let application: ElectronApplication | undefined;
  let child: ReturnType<ElectronApplication["process"]> | undefined;
  let terminal: { code: number | null; signal: NodeJS.Signals | null } | undefined;
  let resourcePause: AuditResourceSnapshot | undefined;
  const startedAt = new Date().toISOString();
  const started = performance.now(); let newlyVisited = 0;
  const requiredKeys = new Set(cells.flatMap(cell => coreStates.map(state => `${String(cell.index).padStart(3, "0")}-${cell.key}-${state}`)));
  if (existsSync(metadata)) {
    const previous = JSON.parse(readFileSync(metadata, "utf8")) as { format?: string; sourceCommit?: string; renderedBytesSha256?: string; auditSpecSha256?: string };
    if (previous.format !== "omni-whole-app-ui-audit" || previous.sourceCommit !== sourceCommit || previous.renderedBytesSha256 !== renderedBytesSha256 || previous.auditSpecSha256 !== auditSpecSha256) {
      throw new Error("Existing audit output has different source/rendered bytes; choose a new output directory.");
    }
    if (existsSync(journal)) {
      const prior = await journalRows(journal, requiredKeys);
      for (const value of prior.values()) if (value.status === "visited" && value.issues.length === 0) visits.push({ key: value.key, status: value.status, route: value.route, cellIndex: value.cellIndex });
    }
  }
  const writeCoverage = (state: "running" | "complete" | "incomplete" | "paused"): void => atomicJson(metadata, {
    format: "omni-whole-app-ui-audit", version: 1, scopeId: auditScope.id, loopbackPreload, state, sourceCommit, dirtyFiles, renderedFiles, renderedBytesSha256, auditSpecSha256, startedAt,
    updatedAt: new Date().toISOString(), dimensions: { palettes, surfaces, schemes, arrangements, sizes, routes, designStates },
    requiredCells: cells.length, requiredRouteVisits: cells.length * routes.length,
    requiredDesignVisits: cells.length * designStates.length, requiredCoreVisits: cells.length * coreStates.length,
    selectedCellIndexes: selected.map(cell => cell.index), expectedSelectedRouteVisits: selected.length * routes.length,
    expectedSelectedDesignVisits: selected.length * designStates.length, expectedSelectedCoreVisits: selected.length * coreStates.length,
    visitedRouteCount: visits.filter(visit => routes.some(route => route.id === visit.route)).length,
    visitedCoreCount: visits.length, visited: visits, failures, prohibitedRequests,
    progress: { completedCoreVisits: visits.length, totalCoreVisits: cells.length * coreStates.length,
      elapsedMs: Math.round(performance.now() - started), newVisitsThisAttempt: newlyVisited,
      ...(newlyVisited > 0 ? { averageVisitsPerSecond: newlyVisited / ((performance.now() - started) / 1000),
        estimatedRemainingMs: Math.round((cells.length * coreStates.length - visits.length) * (performance.now() - started) / newlyVisited) } : {}) },
    fullTraversal: visits.length === cells.length * coreStates.length && new Set(visits.map(visit => visit.key)).size === cells.length * coreStates.length,
    visualAcceptance: "not-established-by-metrics-or-unreviewed-captures",
    boundaries: { preview: true, preload: false, modelsCreated: 0, modelsLoaded: 0, training: 0, providers: 0, permissions: 0, externalEffects: 0 },
    profile, childTerminal: terminal ?? null, resourcePause: resourcePause ?? null,
    exclusions: ["real host/neural transitions", "real account/device/platform interoperability", "unbounded user-authored CSS", "screen-reader conformance"]
  });
  writeCoverage("running");
  try {
    admitAuditCell(directory, "core-before-owned-launch");
    const env = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"));
    application = await electron.launch({ args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
      env: { ...env, NODE_ENV: "test", OMNI_RESPONSIVE_REPOSITORY: repository, OMNI_RESPONSIVE_USER_DATA_DIR: profile } });
    child = application.process();
    child.once("exit", (code, signal) => { terminal = { code, signal }; });
    await application.context().route(/^https?:\/\//, async request => { prohibitedRequests.push(request.request().url()); await request.abort("blockedbyclient"); });
    const page = await application.firstWindow();
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect(page.locator(".app-shell")).toBeVisible();
    for (const cell of selected) {
      if (coreStates.every(state => visits.some(value => value.key === `${String(cell.index).padStart(3, "0")}-${cell.key}-${state}`))) continue;
      admitAuditCell(directory, `core-before-cell-${cell.index}`);
      await configureCell(page, cell);
      const entries = [...designStates.slice(0, 10).map(id => ({ id, label: id, selector: designSelectors[id]! })), ...routes,
        ...designStates.slice(10).map(id => ({ id, label: id, selector: designSelectors[id]! }))];
      for (const route of entries) {
        const begin = performance.now();
        const record: Visit = { key: `${String(cell.index).padStart(3, "0")}-${cell.key}-${route.id}`, cellIndex: cell.index,
          route: route.id, state: "actual-production-preview/no-host-connection", status: "visited", startedAt: new Date().toISOString(), elapsedMs: 0, controls: [], scroll: [], captures: [], issues: [] };
        try {
          if (designStates.includes(route.id)) await prepareDesignState(page, route.id);
          else {
            if (await page.locator(".builder-page").isVisible()) await page.getByRole("button", { name: "Brain Library", exact: true }).click();
            if (await page.locator(".library-page").isVisible()) await openDesignBrain(page);
            await workspaceRoute(page, route.label);
          }
          if (visits.some(value => value.key === record.key)) continue;
          const root = page.locator(route.selector);
          await expect(root).toBeVisible();
          // Inspect shared titlebar/rail/header controls once in every cell,
          // not only each route's inner form. Hidden routes remain hidden.
          const auditRoot = route.id === "chat" ? page.locator(".app-shell") : root;
          await settledFrame(page);
          if (representative(cell)) await capture(page, screenshots, record, "entry");
          await walkContent(page, auditRoot, record);
          await restoreLowerContentAfterKeyboard(page, record);
          if (representative(cell)) await capture(page, screenshots, record, "lower-content");
          if (record.issues.length) { record.status = "failed"; await capture(page, screenshots, record, "failure"); }
        } catch (error) {
          record.status = "failed";
          record.issues.push({ kind: "visit-error", detail: error instanceof Error ? error.message : String(error) });
          await capture(page, screenshots, record, "exception").catch(() => undefined);
        }
        record.elapsedMs = Math.round(performance.now() - begin);
        appendFileSync(journal, `${JSON.stringify(record)}\n`, { flush: true });
        visits.push({ key: record.key, status: record.status, route: record.route, cellIndex: cell.index });
        newlyVisited += 1;
        if (record.issues.length) failures.push({ key: record.key, issues: record.issues });
      }
      writeCoverage("running");
    }
    expect(prohibitedRequests, "No provider or external HTTP request is authorized by this audit").toEqual([]);
    expect(visits.filter(value => selected.some(cell => cell.index === value.cellIndex)).length).toBe(selected.length * coreStates.length);
    expect(new Set(visits.map(visit => visit.key)).size).toBe(visits.length);
    expect(loopbackPreloadInventory()).toEqual(loopbackPreload);
    expect(failures, `Read ${journal} and inspect captures before changing UI; captures are not automatically accepted`).toEqual([]);
  } catch (error) {
    if (error instanceof AuditResourcePauseError) resourcePause = error.snapshot;
    else throw error;
  } finally {
    let closeFailure: unknown;
    if (application && child) {
      const deadline = new Promise<never>((_, reject) => {
        const timer = setTimeout(() => reject(new Error("The owned whole-App audit child did not close within 15 seconds.")), 15_000);
        timer.unref();
        child!.once("exit", () => clearTimeout(timer));
      });
      try { await Promise.race([application.close(), deadline]); }
      catch (error) { closeFailure = error; if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); }
      if (child.exitCode !== null || child.signalCode !== null) terminal ??= { code: child.exitCode, signal: child.signalCode };
      if (!terminal || terminal.code !== 0 || terminal.signal !== null) closeFailure ??= new Error("Whole-App audit did not observe a clean exit of its exact owned child.");
    }
    const complete = visits.length === cells.length * coreStates.length && failures.length === 0 && prohibitedRequests.length === 0 && !closeFailure;
    writeCoverage(resourcePause ? "paused" : complete ? "complete" : "incomplete");
    if (terminal && existsSync(profile)) rmSync(profile, { recursive: true, force: false });
    await testInfo.attach("whole-app-coverage", { path: metadata, contentType: "application/json" });
    if (existsSync(journal)) await testInfo.attach("whole-app-visits", { path: journal, contentType: "application/x-ndjson" });
    if (closeFailure) throw closeFailure;
  }
  test.skip(Boolean(resourcePause), "Paused at the measured resource boundary; partial coverage is not a passed gate.");
});

const compositionStates = auditScope.compositionStates;

async function composition(page: Page, method: Exclude<keyof WholeAppCompositionControl, "status" | "training" | "storage" | "finish">): Promise<void> {
  await page.evaluate(name => { if (!window.omniWholeAppComposition) throw new Error("Synthetic composition control is missing."); return window.omniWholeAppComposition[name](); }, method);
}

async function compositionStatus(page: Page) {
  return page.evaluate(() => { if (!window.omniWholeAppComposition) throw new Error("Synthetic composition control is missing."); return window.omniWholeAppComposition.status(); });
}

async function noticeGeometry(page: Page): Promise<NoticeGeometry | undefined> {
  return page.evaluate(() => {
    const notice = document.querySelector<HTMLElement>(".toast");
    const conversation = document.querySelector<HTMLElement>(".chat-layout:not([hidden]) .conversation");
    const composer = conversation?.querySelector<HTMLElement>(":scope > .call-layout > .call-layout__conversation > .composer-wrap");
    if (!notice || !conversation || !composer) return;
    const box = (node: HTMLElement) => { const value = node.getBoundingClientRect(); return { left: value.left, right: value.right, top: value.top, bottom: value.bottom }; };
    const style = getComputedStyle(composer);
    return { notice: box(notice), conversation: box(conversation), composer: { ...box(composer), scrollTop: composer.scrollTop,
      scrollHeight: composer.scrollHeight, clientHeight: composer.clientHeight, marginBottom: style.marginBottom, paddingBottom: style.paddingBottom },
      reservation: conversation.getAttribute("data-notice-reservation"), measuredSpace: conversation.style.getPropertyValue("--conversation-notice-space") };
  });
}

test("U1 whole-App active chronology, off-page completion and overlay composition use typed inert events", async ({}, testInfo) => {
  test.skip(process.env.OMNI_WHOLE_APP_AUDIT !== "1", "Explicit opt-in required for the serial U1 audit.");
  const fixtureUrl = process.env.OMNI_WHOLE_APP_STATE_URL;
  test.skip(!fixtureUrl, "Set OMNI_WHOLE_APP_STATE_URL to the owned compiled static fixture; absence is not a passed composition gate.");
  const url = new URL(fixtureUrl!);
  if (url.protocol !== "file:" && (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || url.pathname !== "/tests/e2e/whole-app-state-composition-fixture.html")) throw new Error("Only the manifest-bound owned static file or explicit loopback composition fixture is allowed.");
  const timeout = boundedInteger(process.env.OMNI_WHOLE_APP_AUDIT_TIMEOUT_MS, 90 * 60_000, "whole-App composition timeout");
  if (timeout < 30_000 || timeout > 12 * 60 * 60_000) throw new Error("Whole-App composition timeout must be 30 seconds to 12 hours.");
  test.setTimeout(timeout);
  const selected = selectedCells();
  const records: Array<{ key: string; cellIndex: number; state: string; issues: Issue[] }> = [], failures: Array<{ key: string; issues: Issue[] }> = [];
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8" }).trim();
  const sourceFingerprint = rendererFingerprint([join(repository, "src", "renderer"), join(repository, "src", "shared")]);
  const sourceFiles = fileInventory([join(repository, "src", "renderer"), join(repository, "src", "shared")]), dirtyFiles = dirtyFileInventory();
  const fixtureSha256 = createHash("sha256").update(sourceFingerprint)
    .update(readFileSync(resolve(repository, "tests/e2e/whole-app-state-composition-fixture.tsx")))
    .update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-contract.ts")))
    .update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-scope.json")))
    .update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-resources.mjs")))
    .update(readFileSync(resolve(repository, "tests/e2e/appearance-audit-observation.mjs")))
    .update(readFileSync(resolve(repository, "tests/e2e/whole-app-audit-matrix.spec.ts")))
    .update(readFileSync(resolve(repository, "tests/e2e/control-paint-evidence.ts"))).digest("hex");
  const staticArtifact = await verifyStaticFixture(url, { sourceCommit, sourceFingerprint, fixtureSha256 });
  const loopbackPreload = loopbackPreloadInventory();
  const directory = process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR
    ? join(resolve(process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR), "composition")
    : join(repository, ".runtime", "whole-app-ui-composition", sourceCommit, fixtureSha256);
  const screenshots = join(directory, "captures"); mkdirSync(screenshots, { recursive: true });
  const profile = mkdtempSync(join(tmpdir(), "omni-whole-app-composition-"));
  const journal = join(directory, "states.jsonl"), metadata = join(directory, "coverage.json");
  const stateKeys = new Set(cells.flatMap(cell => compositionStates.map(state => `${String(cell.index).padStart(3, "0")}-${cell.key}-${state}`)));
  const resumedCells = new Set<number>();
  if (existsSync(metadata)) {
    const previous = JSON.parse(readFileSync(metadata, "utf8")) as { format?: string; sourceCommit?: string; fixtureSha256?: string; staticArtifact?: { manifestSha256?: string } };
    if (previous.format !== "omni-whole-app-composition" || previous.sourceCommit !== sourceCommit || previous.fixtureSha256 !== fixtureSha256 ||
        previous.staticArtifact?.manifestSha256 !== staticArtifact.manifestSha256) throw new Error("Existing composition output belongs to different fixture/source/served artifact bytes; choose a new output directory.");
    const prior = await journalRows(journal, stateKeys);
    for (const cell of cells) {
      const values = compositionStates.map(state => prior.get(`${String(cell.index).padStart(3, "0")}-${cell.key}-${state}`));
      if (values.every(value => value?.status === "visited" && value.issues.length === 0)) {
        resumedCells.add(cell.index);
        for (const value of values) records.push({ key: value!.key, cellIndex: value!.cellIndex, state: value!.state, issues: [] });
      }
    }
  }
  const started = performance.now();
  const resumedCount = records.length; let lastSummary = started;
  const externalRequests: string[] = [], runtimeErrors: string[] = [], servedByteErrors: string[] = [];
  const navigationEvents: Array<{ event: string; url: string; at: string; status?: number; error?: string }> = [];
  const pendingResponseChecks = new Set<Promise<void>>();
  let application: ElectronApplication | undefined, child: ReturnType<ElectronApplication["process"]> | undefined;
  let terminal: { code: number | null; signal: NodeJS.Signals | null } | undefined;
  let resourcePause: AuditResourceSnapshot | undefined;
  const write = (state: "running" | "complete" | "incomplete" | "paused"): void => atomicJson(metadata, {
    format: "omni-whole-app-composition", version: 1, scopeId: auditScope.id, loopbackPreload, state, sourceCommit, dirtyFiles, sourceFiles, sourceFingerprint, fixtureSha256,
    requiredCells: cells.length, selectedCellIndexes: selected.map(cell => cell.index), states: compositionStates,
    expectedFullStateVisits: cells.length * compositionStates.length, visited: records, failures, externalRequests, runtimeErrors,
    staticArtifact, servedByteErrors, navigationEvents,
    elapsedMs: Math.round(performance.now() - started),
    resumedExactStateCount: resumedCount,
    ...(records.length > resumedCount ? { measuredStatesPerSecond: (records.length - resumedCount) / ((performance.now() - started) / 1000),
      estimatedRemainingMs: Math.round((cells.length * compositionStates.length - records.length) * (performance.now() - started) / (records.length - resumedCount)) } : {}),
    fullTraversal: records.length === cells.length * compositionStates.length && new Set(records.map(record => record.key)).size === records.length,
    visualAcceptance: "not-established-by-metrics-or-unreviewed-captures",
    transport: `typed synthetic in-memory events through actual App controller; manifest-bound static ${url.protocol === "file:" ? "files" : "HTTP"} without HMR`,
    boundaries: { ipc: false, preload: false, models: 0, weights: 0, create: 0, load: 0, training: 0, tools: 0, providers: 0, devices: 0, externalEffects: 0 },
    exclusions: ["real neural/host sequencing", "persisted filesystem restart", "same-brain inference", "provider/device/platform qualification"],
    childTerminal: terminal ?? null, resourcePause: resourcePause ?? null
  });
  write("running");
  try {
    const resourceRoot = process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR ? resolve(process.env.OMNI_WHOLE_APP_AUDIT_OUTPUT_DIR) : directory;
    admitAuditCell(resourceRoot, "composition-before-owned-launch");
    const env = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"));
    application = await electron.launch({ args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
      env: { ...env, NODE_ENV: "test", OMNI_RESPONSIVE_REPOSITORY: repository, OMNI_RESPONSIVE_USER_DATA_DIR: profile } });
    child = application.process(); child.once("exit", (code, signal) => { terminal = { code, signal }; });
    await application.context().route(/^https?:\/\//, async request => {
      const requested = new URL(request.request().url());
      if (url.protocol === "http:" && requested.origin === url.origin && staticArtifact.manifest.files.some(file => `/${file.path}` === requested.pathname) && !requested.search) await request.continue();
      else { externalRequests.push(request.request().url()); await request.abort("blockedbyclient"); }
    });
    const page = await application.firstWindow();
    await page.addInitScript(createAppearanceAuditObservation, { capacity: 256, install: true });
    // responsive-main's authorized no-preload bootstrap is a distinct frozen
    // product file, not a fixture asset. Let that exact load finish before
    // applying fixture-only file routing; keep HTTP blocked from the start.
    const bootstrapUrl = new URL(pathToFileURL(resolve(repository, "out/renderer/index.html")));
    bootstrapUrl.searchParams.set("omni-preview", "1");
    await page.waitForURL(bootstrapUrl.href); await page.waitForLoadState("load");
    if (url.protocol === "file:") await application.context().route(/^file:\/\//, async request => {
      const requested = new URL(request.request().url());
      if (!requested.search && !requested.hash && staticArtifact.manifest.files.some(file => pathToFileURL(resolve(staticArtifact.site, file.path)).href === requested.href)) await request.continue();
      else { externalRequests.push(request.request().url()); await request.abort("blockedbyclient"); }
    });
    page.on("pageerror", error => runtimeErrors.push(error.message));
    page.on("request", request => { if (request.isNavigationRequest()) navigationEvents.push({ event: "request", url: request.url(), at: new Date().toISOString() }); });
    page.on("requestfailed", request => { if (request.isNavigationRequest()) navigationEvents.push({ event: "requestfailed", url: request.url(), at: new Date().toISOString(), error: request.failure()?.errorText }); });
    page.on("domcontentloaded", () => navigationEvents.push({ event: "domcontentloaded", url: page.url(), at: new Date().toISOString() }));
    page.on("load", () => navigationEvents.push({ event: "load", url: page.url(), at: new Date().toISOString() }));
    page.on("response", response => {
      const requested = new URL(response.url());
      if (requested.protocol !== url.protocol || url.protocol === "http:" && requested.origin !== url.origin) return;
      const file = staticArtifact.manifest.files.find(value => url.protocol === "file:"
        ? resolve(staticArtifact.site, value.path) === fileURLToPath(requested) : `/${value.path}` === requested.pathname);
      const check = (async () => {
        if (!file || (url.protocol === "file:" ? ![0, 200].includes(response.status()) : response.status() !== 200 || response.headers()["x-omni-u1-sha256"] !== file.sha256)) throw new Error(`Unexpected static response: ${response.status()} ${response.url()}`);
        const body = await response.body();
        if (body.byteLength !== file.bytes || createHash("sha256").update(body).digest("hex") !== file.sha256) throw new Error(`Served bytes differ from the frozen artifact: ${response.url()}`);
        if (response.request().isNavigationRequest()) navigationEvents.push({ event: "response", url: response.url(), status: response.status(), at: new Date().toISOString() });
      })().catch(error => { servedByteErrors.push(error instanceof Error ? error.message : String(error)); });
      pendingResponseChecks.add(check); void check.finally(() => pendingResponseChecks.delete(check));
    });
    await page.goto(url.href); await page.emulateMedia({ reducedMotion: "reduce" });
    for (const cell of selected) {
      if (resumedCells.has(cell.index)) continue;
      admitAuditCell(resourceRoot, `composition-before-cell-${cell.index}`);
      await page.setViewportSize({ width: cell.size.width, height: cell.size.height });
      await expect.poll(() => page.evaluate(() => ({ width: innerWidth, height: innerHeight }))).toEqual({ width: cell.size.width, height: cell.size.height });
      await page.evaluate(value => {
        localStorage.setItem("omni.appearance.v1", JSON.stringify({ schemaVersion: 1, mode: value.scheme, palette: value.palette, layout: value.surface }));
        localStorage.setItem("omni.appearance.workspace.v1", JSON.stringify({ schemaVersion: 1, arrangement: value.arrangement, notificationsEnabled: false, profiles: [] }));
      }, cell);
      await page.reload();
      await page.getByRole("heading", { name: "UI Fixture — no model", exact: true }).click();
      await expect(page.getByLabel("Message UI Fixture — no model")).toBeVisible();
      await expect.poll(async () => (await compositionStatus(page)).streamSubscribers).toBeGreaterThan(0);
      const inspect = async (state: string, selector: string, checks?: () => Promise<Issue[]>): Promise<void> => {
        const begin = performance.now(), record: Visit = { key: `${String(cell.index).padStart(3, "0")}-${cell.key}-${state}`, cellIndex: cell.index,
          route: "whole-app-composition", state, status: "visited", startedAt: new Date().toISOString(), elapsedMs: 0, controls: [], scroll: [], captures: [], issues: [] };
        try {
          const root = page.locator(selector); await expect(root).toBeVisible(); await settledFrame(page);
          if (checks) record.issues.push(...await checks());
          if (state === "chat-failed-input-preserved") record.noticeGeometry = { before: await noticeGeometry(page) };
          if (representative(cell)) await capture(page, screenshots, record, "entry");
          await walkContent(page, root, record);
          await restoreLowerContentAfterKeyboard(page, record);
          if (state === "chat-failed-input-preserved") record.noticeGeometry = { ...record.noticeGeometry, after: await noticeGeometry(page) };
          if (representative(cell)) await capture(page, screenshots, record, "lower-content");
        } catch (error) { record.issues.push({ kind: "composition-error", detail: error instanceof Error ? error.message : String(error) }); }
        if (state.startsWith("appearance-over-training-")) record.appearanceLifecycle = await appearanceObservations(page);
        if (record.issues.length) { record.status = "failed"; failures.push({ key: record.key, issues: record.issues }); await capture(page, screenshots, record, "failure").catch(() => undefined); }
        record.elapsedMs = Math.round(performance.now() - begin);
        appendFileSync(journal, `${JSON.stringify(record)}\n`, { flush: true });
        records.push({ key: record.key, cellIndex: cell.index, state, issues: record.issues });
        // Flush exact state data above, but do not rewrite a growing complete
        // summary on every hover/state and needlessly amplify SSD writes.
        if (performance.now() - lastSummary >= 10_000) { write("running"); lastSummary = performance.now(); }
      };
      await inspect("chat-idle", ".chat-layout:not([hidden])");
      for (const kind of ["Attach", "Learn"] as const) {
        await page.getByRole("button", { name: kind, exact: true }).click();
        await inspect(`chat-${kind.toLowerCase()}-menu`, ".universal-source-picker__menu:popover-open", async () => {
          const menu = page.locator(".universal-source-picker__menu:popover-open");
          return await menu.getByRole("button", { name: "Files & media", exact: true }).count() === 1 &&
            await menu.getByRole("button", { name: "Folder", exact: true }).count() === 1 ? [] : [{ kind: "source-menu-missing-choice", detail: "Unified source menu must expose files/media and folders without invoking a picker." }];
        });
        await page.keyboard.press("Escape");
      }
      await workspaceRoute(page, "Tools & permissions");
      await inspect("plugins-maker", ".plugin-manager");
      for (const [starter, state] of [["Feature & menu", "plugins-feature-source"], ["App styling", "plugins-style-source"], ["Training format", "plugins-training-parser"]] as const) {
        await page.getByRole("button", { name: `+ ${starter}`, exact: true }).click();
        await inspect(state, ".plugin-module-authoring");
      }
      await workspaceRoute(page, "Conversation");
      const input = page.getByLabel("Message UI Fixture — no model");
      await input.fill("Synthetic composition request. Do not run a tool, model or host action.");
      await page.getByRole("button", { name: "Send message", exact: true }).click();
      await expect.poll(async () => (await compositionStatus(page)).pendingTurns.length).toBe(1);
      await composition(page, "start"); await composition(page, "openingText");
      await inspect("chat-streaming-before-ponder", ".chat-layout:not([hidden])");
      await composition(page, "ponder"); await composition(page, "toolDuringPonder"); await composition(page, "phaseOnlyUnboundAction");
      await inspect("chat-ponder-tool-nesting", ".chat-layout:not([hidden])", async () => page.locator(".message-stream").evaluate(element => {
        const ponder = [...element.querySelectorAll(".chat-action-card")].find(card => /^Ponder(?:ing)?$/.test(card.querySelector(":scope > summary strong")?.textContent?.trim() ?? ""));
        const tool = [...element.querySelectorAll(".chat-action-card")].find(card => card.querySelector(":scope > .chat-action-card__body > code")?.textContent?.includes("fixture.inspect"));
        const unbound = [...element.querySelectorAll(".chat-action-card")].find(card => card.querySelector(":scope > .chat-action-card__body > code")?.textContent?.includes("fixture.unbound"));
        const issues: Issue[] = [];
        if (!ponder || !tool || ponder === tool || !ponder.contains(tool)) issues.push({ kind: "ponder-tool-not-nested", detail: "Same-turn tool with explicit native parent provenance is not inside its actual Ponder disclosure." });
        if (!unbound || unbound.parentElement?.closest(".chat-action-card")) issues.push({ kind: "unbound-phase-guessed-as-parent", detail: "A phase-only history action must remain visibly flat; selectionPhase is not ancestry." });
        return issues;
      }));
      await composition(page, "requestToolApproval");
      await inspect("chat-approval-during-ponder", ".chat-layout:not([hidden])");
      await composition(page, "resumeText");
      await inspect("chat-continuation-chronology", ".chat-layout:not([hidden])", async () => page.locator(".message-stream").evaluate(element => {
        const rangeFor = (needle: string): Range | undefined => {
          const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT); let current: Node | null;
          while ((current = walker.nextNode())) { const position = current.textContent?.indexOf(needle) ?? -1; if (position >= 0) { const range = document.createRange(); range.setStart(current, position); range.setEnd(current, position + needle.length); return range; } }
          return;
        };
        const opening = rangeFor("Synthetic opening before Ponder."), ponder = rangeFor("Synthetic Ponder publication for layout inspection"), resumed = rangeFor("Synthetic continuation after Ponder.");
        return opening && ponder && resumed && opening.compareBoundaryPoints(Range.START_TO_START, ponder) < 0 && ponder.compareBoundaryPoints(Range.START_TO_START, resumed) < 0
          ? [] : [{ kind: "reply-ponder-order", detail: "Opening text, emitted Ponder and resumed ordinary text are not in their actual stream order." }];
      }));
      await workspaceRoute(page, "Brain map");
      await inspect("chat-offpage-active", ".workspace-body");
      await page.evaluate(() => window.omniWholeAppComposition!.finish());
      await expect.poll(async () => (await compositionStatus(page)).pendingTurns.length).toBe(0);
      await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: "Conversation", exact: true }).click();
      await inspect("chat-offpage-complete-return", ".chat-layout:not([hidden])", async () => {
        const issues: Issue[] = [];
        if (await page.getByRole("button", { name: "Queue message", exact: true }).count()) issues.push({ kind: "terminal-queue-remains", detail: "The actual controller retains Queue after the synthetic terminal receipt." });
        if (await page.locator(".pondering-label").count()) issues.push({ kind: "terminal-indicator-remains", detail: "A response indicator remains after the exact terminal receipt." });
        return issues;
      });
      await page.getByRole("button", { name: "Brain library", exact: true }).click();
      await page.getByRole("heading", { name: "UI Fixture — no model", exact: true }).click();
      await inspect("chat-restored-in-memory-ledger", ".chat-layout:not([hidden])", async () => {
        const status = await compositionStatus(page);
        return status.receipts.filter(value => value.operation === "synthetic-send-registered").length === 1 && await page.locator(".message--brain").count() === 1
          ? [] : [{ kind: "restored-duplicate-or-missing-reply", detail: "Remounting actual App chat duplicated submission or lost the synthetic saved reply." }];
      });
      await input.fill("Synthetic failing UI request; preserve its visible input and useful error.");
      await page.getByRole("button", { name: "Send message", exact: true }).click();
      await expect.poll(async () => (await compositionStatus(page)).pendingTurns.length).toBe(1);
      await composition(page, "start"); await composition(page, "fail");
      await inspect("chat-failed-input-preserved", ".chat-layout:not([hidden])", async () => {
        const visible = await page.locator(".message--human").allTextContents();
        return visible.some(value => value.includes("Synthetic failing UI request")) ? [] : [{ kind: "failed-input-disappeared", detail: "The actual App discarded the synthetic accepted input on a failed reply." }];
      });
      // Actual App onBuild/onStorageOperation/onImportFailed subscriptions;
      // these are layout event fixtures, not simulated successful training.
      for (const phase of ["extracting", "learning", "saving", "finalizing", "complete"] as const) {
        await page.evaluate(value => window.omniWholeAppComposition!.training(value), phase);
        if (phase === "finalizing") { await composition(page, "notification"); await page.evaluate(() => window.omniWholeAppComposition!.storage("paused")); }
        await inspect(`training-${phase}`, ".build-readiness-overlay");
        await page.getByRole("button", { name: "Appearance settings", exact: true }).click();
        await inspect(`appearance-over-training-${phase}`, '.appearance-panel[role="dialog"]', async () => page.locator('.appearance-panel[role="dialog"]').evaluate(element => {
          const bounds = element.getBoundingClientRect(), point = { x: Math.max(0, Math.min(innerWidth - 1, bounds.left + bounds.width / 2)), y: Math.max(0, Math.min(innerHeight - 1, bounds.top + Math.min(40, bounds.height / 2))) };
          const top = document.elementFromPoint(point.x, point.y);
          return top && element.contains(top) ? [] : [{ kind: "appearance-covered", detail: "Appearance is obscured by another actual App layer." }];
        }));
        // inspect() above already sealed the failed row and screenshot. Do
        // not add a second 30s locator wait, reopen it, or advance as a pass.
        const appearanceEnd = await appearanceObservations(page);
        if (await page.locator('.appearance-panel[role="dialog"]').count() !== 1 ||
            appearanceAuditFailureMessage(appearanceEnd?.current, `completion of ${phase} overlay inspection`)) {
          throw new Error(`Appearance panel disappeared while inspecting the ${phase} overlay; failure and lifecycle evidence were preserved.`);
        }
        await page.getByRole("button", { name: "Close appearance settings", exact: true }).click();
      }
      expect((await compositionStatus(page)).denied.filter(value => /brain\.(?:create|load)|train\.|chat\.(?:approve|decline|cancel)/.test(value))).toEqual([]);
      write("running"); lastSummary = performance.now();
    }
    expect(records.filter(value => selected.some(cell => cell.index === value.cellIndex)).length).toBe(selected.length * compositionStates.length);
    expect(new Set(records.map(record => record.key)).size).toBe(records.length);
    expect(rendererFingerprint([join(repository, "src", "renderer"), join(repository, "src", "shared")]), "Source changed during static fixture inspection; do not combine different runtime bytes").toBe(sourceFingerprint);
    await Promise.all([...pendingResponseChecks]); expect(servedByteErrors).toEqual([]);
    expect(await verifyStaticFixture(url, { sourceCommit, sourceFingerprint, fixtureSha256 })).toEqual(staticArtifact);
    expect(externalRequests).toEqual([]); expect(runtimeErrors).toEqual([]);
    expect(loopbackPreloadInventory()).toEqual(loopbackPreload);
    expect(failures, `Inspect ${journal}; typed fixture transitions do not prove real runtime behavior`).toEqual([]);
  } catch (error) {
    if (error instanceof AuditResourcePauseError) resourcePause = error.snapshot;
    else throw error;
  } finally {
    let closeFailure: unknown;
    if (application && child) {
      const timerDeadline = new Promise<never>((_, reject) => { const timer = setTimeout(() => reject(new Error("Owned composition child close timed out.")), 15_000); timer.unref(); child!.once("exit", () => clearTimeout(timer)); });
      try { await Promise.race([application.close(), timerDeadline]); }
      catch (error) { closeFailure = error; if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); }
      if (child.exitCode !== null || child.signalCode !== null) terminal ??= { code: child.exitCode, signal: child.signalCode };
      if (!terminal || terminal.code !== 0 || terminal.signal !== null) closeFailure ??= new Error("Exact composition child did not exit cleanly.");
    }
    write(resourcePause ? "paused" : records.length === cells.length * compositionStates.length && !failures.length && !externalRequests.length && !runtimeErrors.length && !servedByteErrors.length && !closeFailure ? "complete" : "incomplete");
    if (terminal && existsSync(profile)) rmSync(profile, { recursive: true, force: false });
    await testInfo.attach("whole-app-composition-coverage", { path: metadata, contentType: "application/json" });
    if (existsSync(journal)) await testInfo.attach("whole-app-composition-states", { path: journal, contentType: "application/x-ndjson" });
    if (closeFailure) throw closeFailure;
  }
  test.skip(Boolean(resourcePause), "Paused at the measured resource boundary; partial coverage is not a passed gate.");
});
