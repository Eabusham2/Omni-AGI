import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { _electron as electron, expect, test, type Page } from "@playwright/test";

const repository = resolve(process.cwd());
const artifactDirectory = resolve(repository, "test-results", "ui-audit");
const requestedMatrix = process.env.OMNI_UI_AUDIT_MATRIX?.trim();
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
  )
);

const packs = [
  { id: "default", palette: "violet", layout: "standard" },
  { id: "classic", palette: "graphite", layout: "classic" },
  { id: "colorful", palette: "spectrum", layout: "expressive" },
  { id: "liquid-glass", palette: "aqua", layout: "glass" }
] as const;

const supportedViewports = [
  { id: "floor", width: 360, height: 640 },
  { id: "mobile", width: 390, height: 844 },
  { id: "tablet", width: 768, height: 1_024 },
  { id: "short-desktop", width: 1_024, height: 600 },
  { id: "design-target", width: 1_480, height: 940 },
  { id: "ultrawide", width: 2_560, height: 1_080 }
] as const;

const workspaceViews = [
  { label: "Conversation", marker: ".chat-layout:not([hidden])" },
  { label: "Data & training", marker: ".data-page" },
  { label: "Brain map", marker: ".map-page" },
  { label: "Trace & journal", marker: ".trace-page" },
  { label: "Imagination", marker: ".imagine-page" },
  { label: "Tools & permissions", marker: ".tools-page" },
  { label: "Device & runtime", marker: ".resource-settings-page" },
  { label: "Forks & agents", marker: ".agents-page" },
  { label: "Evolution", marker: ".evolution-page" }
] as const;

const workspaceArrangements = [
  { label: "Side by side", value: "split" },
  { label: "Full focus", value: "focus" },
  { label: "Top + below", value: "stacked" },
  { label: "Mixed canvas", value: "mixed" }
] as const;

type GeometryIssue = {
  kind: string;
  control: string;
  detail: string;
};

function matrixRequested(matrix: string): boolean {
  if (!requestedMatrix) return true;
  return requestedMatrix
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .some((entry) => matrix.includes(entry));
}

async function launchAuditApplication() {
  return electron.launch({
    args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
    env: {
      ...inheritedEnvironment,
      NODE_ENV: "test",
      OMNI_RESPONSIVE_REPOSITORY: repository
    }
  });
}

async function setViewport(
  page: Page,
  viewport: { width: number; height: number }
): Promise<void> {
  await page.setViewportSize(viewport);
  const expected = { width: viewport.width, height: viewport.height };
  await expect
    .poll(() => page.evaluate(() => ({ width: innerWidth, height: innerHeight })))
    .toEqual(expected);
}

async function setAppearance(
  page: Page,
  mode: "light" | "dark",
  pack: (typeof packs)[number],
  arrangement: (typeof workspaceArrangements)[number]["value"] = "split"
): Promise<void> {
  await page.evaluate(
    ({ modeValue, palette, layout, workspaceArrangement }) => {
      window.localStorage.setItem(
        "omni.appearance.v1",
        JSON.stringify({ schemaVersion: 1, mode: modeValue, palette, layout })
      );
      window.localStorage.setItem(
        "omni.appearance.workspace.v1",
        JSON.stringify({
          schemaVersion: 1,
          arrangement: workspaceArrangement,
          notificationsEnabled: false,
          profiles: []
        })
      );
    },
    {
      modeValue: mode,
      palette: pack.palette,
      layout: pack.layout,
      workspaceArrangement: arrangement
    }
  );
  await page.reload();
  await page.waitForLoadState("domcontentloaded");
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-color-scheme", mode);
  await expect(root).toHaveAttribute("data-palette", pack.palette);
  await expect(root).toHaveAttribute("data-layout", pack.layout);
  await expect(root).toHaveAttribute("data-workspace-arrangement", arrangement);
}

async function openAster(page: Page): Promise<void> {
  await page
    .locator("article.brain-card")
    .filter({ has: page.getByRole("heading", { name: "Aster", exact: true }) })
    .click();
  await expect(page.getByLabel("Message Aster")).toBeVisible();
}

async function installMeasuredResourceAuditFixture(
  page: Page,
  constrained: boolean
): Promise<void> {
  await page.evaluate(({ constrainedDevice }) => {
    const gib = 1_073_741_824;
    const mib = 1_048_576;
    const floorTokens = constrainedDevice ? 8_192 : 32_768;
    const autoTokens = constrainedDevice ? 12_288 : 65_536;
    const extendedTokens = constrainedDevice ? 16_384 : 98_304;
    const maximumTokens = constrainedDevice ? 24_576 : 131_072;
    const suitableMaximumTokens = constrainedDevice ? 15_360 : 81_920;
    const selectedItems = constrainedDevice ? 131_072 : 196_608;
    const residentMemoryItems = constrainedDevice ? 32_768 : selectedItems;
    const pagedMemoryItems = constrainedDevice
      ? selectedItems - residentMemoryItems
      : 0;

    type FixtureRequest = {
      mode: "auto" | "extended" | "manual";
      requestedContextTokens?: string;
      systemRamMode?: "auto" | "manual";
      systemRamSharePercent?: number;
      storagePoolMode?: "auto" | "manual";
      storagePoolBytes?: string;
      trainingSourceBytes?: number;
    };

    const resourcePlan = async (request: FixtureRequest) => {
      const requestedContext = Number.parseInt(
        request.requestedContextTokens ?? "",
        10
      );
      const selectedTokens = request.mode === "manual"
        ? requestedContext || 1
        : request.mode === "extended"
          ? extendedTokens
          : autoTokens;
      const requiredStoragePoolBytes = constrainedDevice ? 24 * gib : 64 * gib;
      const maximumStoragePoolBytes = constrainedDevice ? 180 * gib : 1_200 * gib;
      const requestedStoragePoolBytes = Number.parseInt(request.storagePoolBytes ?? "", 10);
      const storagePoolMode = request.storagePoolMode ?? "auto";
      const sharedStoragePoolBytes = storagePoolMode === "manual"
        ? requestedStoragePoolBytes || 1
        : constrainedDevice
          ? 48 * gib
          : 128 * gib;
      const contextAllowed = selectedTokens >= floorTokens && selectedTokens <= maximumTokens;
      const storageAllowed = sharedStoragePoolBytes >= requiredStoragePoolBytes &&
        sharedStoragePoolBytes <= maximumStoragePoolBytes;
      const allowed = contextAllowed && storageAllowed;
      const contextResidentBytes = selectedTokens * (constrainedDevice ? 192 * 1_024 : 128 * 1_024);
      const systemRamSharePercent = request.systemRamMode === "manual"
        ? Math.max(30, Math.min(100, request.systemRamSharePercent ?? 65))
        : constrainedDevice
          ? 65
          : 80;
      const storageClass = constrainedDevice ? "moderate-storage" : "fast-storage";
      const diskTotalBytes = constrainedDevice ? 512 * gib : 2_000 * gib;
      const diskFreeBytes = constrainedDevice ? 230 * gib : 1_500 * gib;
      const mandatoryReserveBytes = 20 * gib;
      const modelBytes = constrainedDevice ? 3_995_620 : 31_964_960;
      const checkpointBytes = constrainedDevice ? 15_982_480 : 127_859_840;
      const maximumWorkingMemorySpillBytes = constrainedDevice ? 6 * gib : 0;
      const futureGrowthBytes = constrainedDevice ? 8 * gib : 16 * gib;
      const operationWriteBytes = constrainedDevice ? 5 * gib : 12 * gib;
      const projectedRemainingBytes = Math.max(
        0,
        diskFreeBytes - modelBytes - checkpointBytes -
          maximumWorkingMemorySpillBytes - futureGrowthBytes - operationWriteBytes
      );
      return {
        schemaVersion: 1,
        diskSpace: {
          schemaVersion: 1,
          measuredAt: "2026-08-26T12:00:00.000Z",
          diskTotalBytes,
          diskFreeBytes,
          mandatoryReserveBytes,
          selectedDatasetBytes: request.trainingSourceBytes ?? 0,
          modelBytes,
          checkpointBytes,
          maximumWorkingMemorySpillBytes,
          futureGrowthBytes,
          operationWriteBytes,
          projectedRemainingBytes,
          projectedAboveReserveBytes: Math.max(
            0,
            projectedRemainingBytes - mandatoryReserveBytes
          ),
          paused: projectedRemainingBytes <= mandatoryReserveBytes
        },
        architecture: {
          format: "omni-ground-up-architecture-profile",
          formatVersion: 1,
          architecture: "OmniCortex",
          origin: "ground-up-random-initialization",
          externalPretrainedWeights: false,
          hardwareTier: constrainedDevice ? "personal" : "workstation",
          dModel: constrainedDevice ? 256 : 512,
          layers: constrainedDevice ? 8 : 16,
          feedForward: constrainedDevice ? 768 : 1_536,
          vsaDimensions: constrainedDevice ? 2_048 : 4_096,
          routerNeurons: constrainedDevice ? 128 : 256,
          workingMemoryItems: selectedItems,
          workspaceLatents: constrainedDevice ? 128 : 256,
          exactLogicalParameterCount: constrainedDevice ? 819_200 : 6_488_064,
          exactPackedProjectionParameterCount: constrainedDevice ? 786_432 : 6_291_456,
          exactPackedTableParameterCount: constrainedDevice ? 16_384 : 131_072,
          exactPackedTernaryParameterCount: constrainedDevice ? 819_200 : 6_488_064,
          packedTernaryWeightBytes: constrainedDevice ? 204_800 : 1_622_016,
          packedWorkspaceTableBytes: constrainedDevice ? 16_384 : 131_072,
          packedMetaplasticityReserveBytes: constrainedDevice ? 65_536 : 131_072,
          fixedControlBufferReserveBytes: 16_384,
          packedUpdateScratchBytes: 4_194_304,
          residentInferenceStateBytes: constrainedDevice ? 450_560 : 2_424_832,
          minimumTrainingStateBytes: constrainedDevice ? 4_644_864 : 6_619_136,
          checkpointTensorBytes: constrainedDevice ? 1_499_136 : 3_473_408,
          parameterCountBasis: "architecture-logical-neural-elements",
          checkpointByteBasis: "packed-weights-plus-nonweight-state-reserve"
        },
        mode: request.mode,
        allowed,
        blockers: [
          ...(contextAllowed
            ? []
            : selectedTokens < floorTokens
              ? [`This context is below the ${floorTokens.toLocaleString()}-token baseline measured for this device.`]
              : [`The requested active context cannot stay resident on this device (safe/model maximum ${maximumTokens.toLocaleString()} tokens).`]),
          ...(storageAllowed
            ? []
            : sharedStoragePoolBytes < requiredStoragePoolBytes
              ? [`The shared pool must be at least ${Math.ceil(requiredStoragePoolBytes / gib)} GiB.`]
              : [`The shared pool exceeds the ${Math.floor(maximumStoragePoolBytes / gib)} GiB physical maximum.`])
        ],
        warnings: constrainedDevice
          ? ["Storage offload is required and is estimated to reduce memory-heavy speed by about 18%."]
          : selectedTokens > suitableMaximumTokens
            ? ["This active context is above the measured green range."]
            : [],
        hardwareTier: constrainedDevice ? "personal" : "workstation",
        selectedItems,
        selectedItemsText: String(selectedItems),
        sliderMaximumItems: constrainedDevice ? 524_288 : 2_097_152,
        sliderMaximumItemsText: constrainedDevice ? "524288" : "2097152",
        suitableRange: {
          minimumItems: constrainedDevice ? 16_384 : 65_536,
          autoItems: selectedItems,
          maximumItems: constrainedDevice ? 196_608 : 393_216,
          extendedItems: constrainedDevice ? 196_608 : 393_216
        },
        context: {
          selectedTokens,
          floorTokens,
          autoTokens,
          extendedTokens,
          maximumTokens,
          suitableMinimumTokens: floorTokens,
          suitableMaximumTokens,
          evidence: {
            source: "live-device-model-measurement",
            modelContextLimitTokens: maximumTokens,
            safeRamAfterModelBytes: constrainedDevice ? 3.2 * gib : 42 * gib,
            contextResidentBudgetBytes: constrainedDevice ? 3.2 * gib : 42 * gib,
            estimatedKvActivationBytesPerToken: constrainedDevice ? 192 * 1_024 : 128 * 1_024,
            contextWorkspaceMultiplier: 2,
            minimumContextWorkspaceBytes: constrainedDevice ? 512 * mib : gib,
            selectedContextResidentBytes: contextResidentBytes,
            acceleratorAvailable: !constrainedDevice,
            storageClass,
            measuredStorageBytesPerSecond: constrainedDevice ? 260 * mib : 2.1 * gib,
            autoContextRamFraction: constrainedDevice ? 0.22 : 0.34
          }
        },
        resources: {
          totalMemoryBytes: constrainedDevice ? 16 * gib : 64 * gib,
          availableMemoryBytes: constrainedDevice ? 9 * gib : 54 * gib,
          diskTotalBytes: constrainedDevice ? 512 * gib : 2_000 * gib,
          diskFreeBytes: constrainedDevice ? 230 * gib : 1_500 * gib,
          ramReserveBytes: constrainedDevice ? 1.5 * gib : 4 * gib,
          mandatoryFreeDiskBytes: 20 * gib,
          checkpointHeadroomBytes: constrainedDevice ? 384 * mib : 1.2 * gib,
          usableDiskAfterReserveBytes: constrainedDevice ? 209 * gib : 1_478 * gib,
          modelBytes: constrainedDevice ? 945_000_000 : 3 * gib,
          modelWorkingSetBytes: constrainedDevice ? 1.32 * gib : 4.2 * gib,
          estimatedResidentModelBytes: constrainedDevice ? 1.1 * gib : 3.6 * gib,
          fullFoundationRuntimeBytes: constrainedDevice ? 1.55 * gib : 4.8 * gib,
          minimumLayerResidencyBytes: constrainedDevice ? 220 * mib : 640 * mib,
          residentFoundationBytes: constrainedDevice ? 1.55 * gib : 4.8 * gib,
          runtimeOverheadBytes: 256 * mib,
          configuredMemorySpillBytes: constrainedDevice ? 6 * gib : 0,
          safeRamPoolBytes: constrainedDevice ? 7.5 * gib : 50 * gib,
          availableSafeRamBytes: constrainedDevice ? 6.4 * gib : 50 * gib,
          systemRamBudgetBytes: constrainedDevice ? 4.875 * gib : 40 * gib,
          currentOmniAvailableBytes: constrainedDevice ? 3.775 * gib : 40 * gib,
          currentOmniShortfallBytes: constrainedDevice ? 1.1 * gib : 0,
          systemRamMode: request.systemRamMode ?? "auto",
          systemRamSharePercent,
          autoSystemRamSharePercent: constrainedDevice ? 65 : 80,
          storagePoolMode,
          sharedStoragePoolBytes,
          requiredStoragePoolBytes,
          maximumStoragePoolBytes,
          trainingSourceBytes: request.trainingSourceBytes ?? 0,
          trainingScratchBytes: constrainedDevice ? 3 * gib : 12 * gib,
          futureGrowthHeadroomBytes: constrainedDevice ? 8 * gib : 16 * gib
        },
        offload: {
          required: constrainedDevice,
          residentMemoryItems,
          pagedMemoryItems,
          modelSpillBytes: 0,
          modelOffloadScratchBytes: constrainedDevice ? 2 * gib : 0,
          memorySpillBytes: constrainedDevice ? 6 * gib : 0,
          estimatedSlowdownPercent: constrainedDevice ? 18 : 0,
          benchmark: {
            measuredAt: "2026-08-26T12:00:00.000Z",
            sampleBytes: 64 * mib,
            memoryBytesPerSecond: constrainedDevice ? 12 * gib : 42 * gib,
            storageBytesPerSecond: constrainedDevice ? 260 * mib : 2.1 * gib,
            cacheHit: true
          }
        },
        semantics: {
          unit: "recurrent-paged-memory-item",
          denseAttentionClaim: false,
          contextFloorTokens: floorTokens,
          contextPagedToStorage: false,
          capacityPersistsAcrossPressure: true,
          storagePoolShareRule: "largest-brain-not-sum",
          hotRamPriority: ["currently firing", "frequently used"],
          spillOrder: ["cold scratch trail", "inactive working patterns"]
        },
        training: {
          policy: "ram-first-adaptive-streaming",
          physicalBatchSize: constrainedDevice ? 1 : 8,
          gradientAccumulation: constrainedDevice ? 8 : 1,
          windowTokens: constrainedDevice ? 2_048 : 16_384,
          allSourceBytesVisited: true,
          scratchMode: "emergency-checkpoint-only",
          scratchUsedAsVirtualRam: false,
          sequentialScratchWrites: true,
          minimumScratchIntervalSeconds: constrainedDevice ? 900 : 300,
          storageClass,
          measuredStorageBytesPerSecond: constrainedDevice ? 260 * mib : 2.1 * gib
        }
      };
    };

    Object.defineProperty(window, "omni", {
      configurable: true,
      value: {
        catalog: {
          hardwareProfile: async () => ({
            platform: constrainedDevice ? "linux" : "darwin",
            architecture: "arm64",
            logicalCpus: constrainedDevice ? 8 : 24,
            totalMemoryBytes: constrainedDevice ? 16 * gib : 64 * gib,
            availableMemoryBytes: constrainedDevice ? 9 * gib : 54 * gib,
            gpu: constrainedDevice
              ? { available: false }
              : { available: true, vendor: "Apple", device: "Integrated accelerator" },
            recommendedTier: constrainedDevice ? "personal" : "workstation",
            recommendation: constrainedDevice
              ? "Use measured paging for cold neural memory."
              : "Keep the foundation, context, and neural memory resident."
          }),
          resourcePlan
        },
        mobile: {
          status: async () => ({
            schemaVersion: 1,
            protocolVersion: 1,
            state: "stopped",
            allowLan: false,
            baseUrls: [],
            devices: []
          })
        },
        chat: {
          onAction: () => () => undefined,
          onStream: () => () => undefined
        },
        data: {
          listBuildResources: async () => [],
          selectBuildResources: async (kind: "files" | "folder") => ({
            id: `audit-${kind}`,
            kind,
            label: kind === "folder" ? "Audit corpus folder" : "Audit corpus.parquet",
            itemCount: kind === "folder" ? 12 : 1,
            fileCount: kind === "folder" ? 12 : 1,
            bytes: 6 * gib
          }),
          discardBuildResource: async () => undefined
        }
      }
    });
  }, { constrainedDevice: constrained });
}

function issueMessage(matrix: string, issues: GeometryIssue[]): string {
  return issues
    .map((issue) => `${matrix} [${issue.kind}] ${issue.control}: ${issue.detail}`)
    .join("\n");
}

/**
 * Geometry audit for every rendered button-like control. It checks the WCAG
 * 2.2 minimum target floor, a larger compact-screen target, label containment,
 * clipping, overlap, viewport reachability, and accidental stacking covers.
 */
async function auditGeometry(page: Page): Promise<GeometryIssue[]> {
  return page.evaluate(() => {
    const issues: GeometryIssue[] = [];
    const compact = innerWidth <= 650;
    const tolerance = 1.25;

    const rendered = (element: HTMLElement): boolean => {
      if (element.closest("[hidden]")) return false;
      const closedDetails = element.closest<HTMLDetailsElement>("details:not([open])");
      if (closedDetails) {
        const summary = closedDetails.querySelector<HTMLElement>(":scope > summary");
        if (!summary?.contains(element)) return false;
      }
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        Number(style.opacity || "1") > 0.01 &&
        rect.width > 0 &&
        rect.height > 0
      );
    };

    const name = (element: HTMLElement): string => {
      const accessible = element.getAttribute("aria-label")?.trim();
      if (accessible) return accessible;
      const copy = (element.innerText ?? element.textContent ?? "").replace(/\s+/g, " ").trim();
      if (copy) return copy.slice(0, 90);
      return element.getAttribute("title")?.trim() || `${element.tagName.toLowerCase()}.${element.className}`;
    };

    const describe = (element: HTMLElement): string => {
      const classes = typeof element.className === "string"
        ? element.className.trim().split(/\s+/).filter(Boolean).slice(0, 3).join(".")
        : "";
      return `${element.tagName.toLowerCase()}${classes ? `.${classes}` : ""} “${name(element)}”`;
    };

    const overflowAllowsScroll = (element: HTMLElement, axis: "x" | "y"): boolean => {
      for (let parent = element.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        const overflow = axis === "x" ? style.overflowX : style.overflowY;
        if (overflow === "auto" || overflow === "scroll") return true;
      }
      return false;
    };

    const pointVisibleThroughAncestors = (
      element: HTMLElement,
      x: number,
      y: number
    ): boolean => {
      for (let parent = element.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        const rect = parent.getBoundingClientRect();
        if (
          ["auto", "scroll", "hidden", "clip"].includes(style.overflowX) &&
          (x < rect.left || x > rect.right)
        ) return false;
        if (
          ["auto", "scroll", "hidden", "clip"].includes(style.overflowY) &&
          (y < rect.top || y > rect.bottom)
        ) return false;
      }
      return true;
    };

    const visibleBounds = (element: HTMLElement) => {
      const source = element.getBoundingClientRect();
      let left = Math.max(0, source.left);
      let top = Math.max(0, source.top);
      let right = Math.min(innerWidth, source.right);
      let bottom = Math.min(innerHeight, source.bottom);
      for (let parent = element.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        const rect = parent.getBoundingClientRect();
        if (["auto", "scroll", "hidden", "clip"].includes(style.overflowX)) {
          left = Math.max(left, rect.left);
          right = Math.min(right, rect.right);
        }
        if (["auto", "scroll", "hidden", "clip"].includes(style.overflowY)) {
          top = Math.max(top, rect.top);
          bottom = Math.min(bottom, rect.bottom);
        }
      }
      return { left, top, right, bottom, width: right - left, height: bottom - top };
    };

    const controls = Array.from(
      document.querySelectorAll<HTMLElement>("button, summary, [role='button']")
    ).filter(rendered);

    const rootOverflow = {
      document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      body: document.body.scrollWidth - document.body.clientWidth,
      shell: Math.ceil(
        (document.querySelector<HTMLElement>(".app-shell")?.scrollWidth ?? 0) -
          (document.querySelector<HTMLElement>(".app-shell")?.clientWidth ?? 0)
      )
    };
    for (const [surface, overflow] of Object.entries(rootOverflow)) {
      if (overflow > 1) {
        issues.push({
          kind: "root-overflow",
          control: surface,
          detail: `${overflow}px of horizontal overflow`
        });
      }
    }

    for (const control of controls) {
      const rect = control.getBoundingClientRect();
      const controlName = describe(control);
      const intersectsViewport =
        rect.right > 0 && rect.left < innerWidth && rect.bottom > 0 && rect.top < innerHeight;
      const targetFloor = compact && intersectsViewport ? 36 : 24;

      if (rect.width + tolerance < targetFloor || rect.height + tolerance < targetFloor) {
        issues.push({
          kind: compact && intersectsViewport ? "compact-hit-area" : "hit-area",
          control: controlName,
          detail: `${rect.width.toFixed(1)}×${rect.height.toFixed(1)}px; expected at least ${targetFloor}×${targetFloor}px`
        });
      }

      const hasRailTooltip = Boolean(control.querySelector(".rail-label"));
      if (
        !hasRailTooltip &&
        (control.scrollWidth > control.clientWidth + 1 ||
          control.scrollHeight > control.clientHeight + 1)
      ) {
        issues.push({
          kind: "label-overflow",
          control: controlName,
          detail: `content ${control.scrollWidth}×${control.scrollHeight}px exceeds box ${control.clientWidth}×${control.clientHeight}px`
        });
      }

      const textContainers = Array.from(
        control.querySelectorAll<HTMLElement>("span, strong, small, em, code")
      ).filter((child) => {
        if (!rendered(child) || child.closest(".rail-label")) return false;
        const position = getComputedStyle(child).position;
        return position !== "absolute" && position !== "fixed";
      });
      for (const textContainer of textContainers) {
        const child = textContainer.getBoundingClientRect();
        if (
          child.left < rect.left - tolerance ||
          child.right > rect.right + tolerance ||
          child.top < rect.top - tolerance ||
          child.bottom > rect.bottom + tolerance
        ) {
          issues.push({
            kind: "label-clipping",
            control: controlName,
            detail: `${textContainer.tagName.toLowerCase()} text escapes its button bounds`
          });
          break;
        }
        const fontSize = Number.parseFloat(getComputedStyle(textContainer).fontSize);
        if (Number.isFinite(fontSize) && fontSize < 10) {
          issues.push({
            kind: "button-font",
            control: controlName,
            detail: `${fontSize.toFixed(1)}px visible button copy is below the 10px audit floor`
          });
          break;
        }
      }

      let scrollableX = false;
      let scrollableY = false;
      for (let parent = control.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        const parentRect = parent.getBoundingClientRect();
        scrollableX ||= style.overflowX === "auto" || style.overflowX === "scroll";
        scrollableY ||= style.overflowY === "auto" || style.overflowY === "scroll";
        if (
          !scrollableX &&
          (style.overflowX === "hidden" || style.overflowX === "clip") &&
          (rect.left < parentRect.left - tolerance || rect.right > parentRect.right + tolerance)
        ) {
          issues.push({
            kind: "ancestor-clipping",
            control: controlName,
            detail: `extends horizontally outside ${parent.tagName.toLowerCase()}.${parent.className}`
          });
          break;
        }
        if (
          !scrollableY &&
          (style.overflowY === "hidden" || style.overflowY === "clip") &&
          (rect.top < parentRect.top - tolerance || rect.bottom > parentRect.bottom + tolerance)
        ) {
          issues.push({
            kind: "ancestor-clipping",
            control: controlName,
            detail: `extends vertically outside ${parent.tagName.toLowerCase()}.${parent.className}`
          });
          break;
        }
      }

      if (
        (rect.left < -tolerance || rect.right > innerWidth + tolerance) &&
        !overflowAllowsScroll(control, "x")
      ) {
        issues.push({
          kind: "viewport-clipping",
          control: controlName,
          detail: `horizontal bounds ${rect.left.toFixed(1)}–${rect.right.toFixed(1)}px exceed 0–${innerWidth}px without a scroll container`
        });
      }

      if (intersectsViewport) {
        const visible = visibleBounds(control);
        if (visible.width <= 1 || visible.height <= 1) continue;
        const x = Math.max(0, Math.min(innerWidth - 1, visible.left + visible.width / 2));
        const y = Math.max(0, Math.min(innerHeight - 1, visible.top + visible.height / 2));
        if (!pointVisibleThroughAncestors(control, x, y)) continue;
        const topmost = document.elementFromPoint(x, y);
        const activeDialog = document.querySelector<HTMLElement>("[role='dialog']");
        const coveredByIntentionalDialog = Boolean(activeDialog && !activeDialog.contains(control));
        if (
          topmost &&
          !control.contains(topmost) &&
          !topmost.contains(control) &&
          !coveredByIntentionalDialog &&
          getComputedStyle(control).pointerEvents !== "none"
        ) {
          issues.push({
            kind: "stacking-cover",
            control: controlName,
            detail: `center point is covered by ${topmost.tagName.toLowerCase()}.${(topmost as HTMLElement).className}`
          });
        }
      }
    }

    const onScreenControls = controls.filter((control) => {
      const rect = visibleBounds(control);
      const x = Math.max(0, Math.min(innerWidth - 1, rect.left + rect.width / 2));
      const y = Math.max(0, Math.min(innerHeight - 1, rect.top + rect.height / 2));
      return (
        rect.right > 0 &&
        rect.left < innerWidth &&
        rect.bottom > 0 &&
        rect.top < innerHeight &&
        pointVisibleThroughAncestors(control, x, y)
      );
    });
    const activeDialog = document.querySelector<HTMLElement>("[role='dialog']");
    for (let leftIndex = 0; leftIndex < onScreenControls.length; leftIndex += 1) {
      const left = onScreenControls[leftIndex]!;
      const leftRect = visibleBounds(left);
      for (let rightIndex = leftIndex + 1; rightIndex < onScreenControls.length; rightIndex += 1) {
        const right = onScreenControls[rightIndex]!;
        if (left.contains(right) || right.contains(left)) continue;
        if (
          activeDialog &&
          activeDialog.contains(left) !== activeDialog.contains(right)
        ) continue;
        const rightRect = visibleBounds(right);
        const overlapWidth = Math.min(leftRect.right, rightRect.right) - Math.max(leftRect.left, rightRect.left);
        const overlapHeight = Math.min(leftRect.bottom, rightRect.bottom) - Math.max(leftRect.top, rightRect.top);
        if (overlapWidth > 1.5 && overlapHeight > 1.5) {
          issues.push({
            kind: "control-overlap",
            control: describe(left),
            detail:
              `overlaps ${describe(right)} by ${overlapWidth.toFixed(1)}×${overlapHeight.toFixed(1)}px; ` +
              `bounds ${leftRect.left.toFixed(1)},${leftRect.top.toFixed(1)}–${leftRect.right.toFixed(1)},${leftRect.bottom.toFixed(1)} ` +
              `and ${rightRect.left.toFixed(1)},${rightRect.top.toFixed(1)}–${rightRect.right.toFixed(1)},${rightRect.bottom.toFixed(1)}`
          });
        }
      }
    }

    for (const selector of [
      ".appearance-panel",
      ".instance-delete-dialog",
      ".conversation",
      ".cortex-panel",
      ".content-page",
      ".builder-main"
    ]) {
      const surface = document.querySelector<HTMLElement>(selector);
      if (!surface || !rendered(surface)) continue;
      const rect = surface.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) {
        issues.push({ kind: "collapsed-surface", control: selector, detail: "surface has no usable area" });
      }
      if (
        rect.left < -tolerance ||
        rect.right > innerWidth + tolerance ||
        rect.top < -tolerance ||
        rect.bottom > innerHeight + tolerance
      ) {
        issues.push({
          kind: "surface-placement",
          control: selector,
          detail: `bounds ${rect.left.toFixed(1)},${rect.top.toFixed(1)} ${rect.width.toFixed(1)}×${rect.height.toFixed(1)} exceed the viewport`
        });
      }
    }

    return issues;
  });
}

async function expectGeometry(page: Page, matrix: string): Promise<void> {
  const issues = await auditGeometry(page);
  expect(issues, issueMessage(matrix, issues)).toEqual([]);
}

type ScrollAuditState = "top" | "middle" | "bottom";

type ScrollAuditTarget = {
  id: string;
  label: string;
  maxScroll: number;
};

type ScrollAnchor = {
  id: string;
  label: string;
  left: number;
  top: number;
  width: number;
  height: number;
};

const scrollStates: Array<{ id: ScrollAuditState; fraction: number }> = [
  { id: "top", fraction: 0 },
  { id: "middle", fraction: 0.5 },
  { id: "bottom", fraction: 1 }
];

function artifactSlug(value: string): string {
  return value.toLocaleLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

async function prepareScrollAudit(page: Page): Promise<ScrollAuditTarget[]> {
  return page.evaluate(() => {
    document.querySelectorAll<HTMLElement>("[data-ui-audit-scroll-id]").forEach((element) => {
      element.removeAttribute("data-ui-audit-scroll-id");
    });

    const candidates = new Set<HTMLElement>();
    if (document.scrollingElement instanceof HTMLElement) {
      const rootStyle = getComputedStyle(document.scrollingElement);
      if (rootStyle.overflowY === "auto" || rootStyle.overflowY === "scroll") {
        candidates.add(document.scrollingElement);
      } else {
        document.scrollingElement.scrollTop = 0;
      }
    }
    document.querySelectorAll<HTMLElement>("*").forEach((element) => {
      const style = getComputedStyle(element);
      if (style.overflowY === "auto" || style.overflowY === "scroll") candidates.add(element);
    });

    const rendered = (element: HTMLElement): boolean => {
      if (element.closest("[hidden]")) return false;
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        Number(style.opacity || "1") > 0.01 &&
        rect.width > 0 &&
        rect.height > 0
      );
    };

    return [...candidates]
      .filter((element) => rendered(element) && element.scrollHeight - element.clientHeight > 1)
      .map((element, index) => {
        const id = `scroll-${index + 1}`;
        element.dataset.uiAuditScrollId = id;
        const classes = element.className
          .toString()
          .trim()
          .split(/\s+/)
          .filter(Boolean)
          .slice(0, 3)
          .join(".");
        return {
          id,
          label: `${element.tagName.toLowerCase()}${classes ? `.${classes}` : ""}`,
          maxScroll: Math.max(0, element.scrollHeight - element.clientHeight)
        };
      });
  });
}

async function captureScrollAnchors(page: Page): Promise<ScrollAnchor[]> {
  return page.evaluate(() => {
    const explicit = new Set(
      Array.from(
        document.querySelectorAll<HTMLElement>(
          ".titlebar, .workspace-rail, .workspace-header, .composer-wrap, .builder-sidebar, " +
          ".builder-footer, .appearance-panel, .instance-delete-dialog"
        )
      )
    );
    document.querySelectorAll<HTMLElement>("*").forEach((element) => {
      const position = getComputedStyle(element).position;
      if (position === "fixed" || position === "sticky") explicit.add(element);
    });

    return [...explicit].flatMap((element, index) => {
      if (element.closest("[hidden]")) return [];
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        Number(style.opacity || "1") <= 0.01 ||
        rect.width <= 0 ||
        rect.height <= 0
      ) return [];

      const position = style.position;
      const scrollingAncestor = element.parentElement?.closest<HTMLElement>(
        "[data-ui-audit-scroll-id]"
      );
      if (scrollingAncestor && position !== "fixed" && position !== "sticky") return [];

      const classes = element.className
        .toString()
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 3)
        .join(".");
      return [{
        id: `${element.tagName.toLowerCase()}.${classes || "anchor"}-${index}`,
        label: `${element.tagName.toLowerCase()}${classes ? `.${classes}` : ""}`,
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height
      }];
    });
  });
}

async function moveScrollAudit(
  page: Page,
  state: { id: ScrollAuditState; fraction: number },
  anchors: ScrollAnchor[]
): Promise<GeometryIssue[]> {
  return page.evaluate(
    async ({ stateValue, anchorValues }) => {
      const issues: Array<{ kind: string; control: string; detail: string }> = [];
      const tolerance = 2;
      const targets = Array.from(
        document.querySelectorAll<HTMLElement>("[data-ui-audit-scroll-id]")
      );

      for (const target of targets) {
        target.style.scrollBehavior = "auto";
        const maxScroll = Math.max(0, target.scrollHeight - target.clientHeight);
        target.scrollTop = Math.round(maxScroll * stateValue.fraction);
      }
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });

      // Some responsive panels finish a content-driven reflow after the first
      // scroll frame. Re-apply the requested position against that settled
      // geometry so the audit measures reachability, not a stale pre-reflow
      // scroll range.
      for (const target of targets) {
        const maxScroll = Math.max(0, target.scrollHeight - target.clientHeight);
        target.scrollTop = Math.round(maxScroll * stateValue.fraction);
      }
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });

      for (const target of targets) {
        const maxScroll = Math.max(0, target.scrollHeight - target.clientHeight);
        const expected = Math.round(maxScroll * stateValue.fraction);
        if (Math.abs(target.scrollTop - expected) > tolerance) {
          const classes = target.className
            .toString()
            .trim()
            .split(/\s+/)
            .filter(Boolean)
            .slice(0, 3)
            .join(".");
          issues.push({
            kind: "scroll-reachability",
            control: `${target.tagName.toLowerCase()}${classes ? `.${classes}` : ""}`,
            detail: `${stateValue.id} stopped at ${target.scrollTop.toFixed(1)}px; expected ${expected}px of ${maxScroll}px`
          });
        }
      }

      const currentAnchors = new Map<string, HTMLElement>();
      const explicit = new Set(
        Array.from(
          document.querySelectorAll<HTMLElement>(
            ".titlebar, .workspace-rail, .workspace-header, .composer-wrap, .builder-sidebar, " +
            ".builder-footer, .appearance-panel, .instance-delete-dialog"
          )
        )
      );
      document.querySelectorAll<HTMLElement>("*").forEach((element) => {
        const position = getComputedStyle(element).position;
        if (position === "fixed" || position === "sticky") explicit.add(element);
      });
      [...explicit].forEach((element, index) => {
        const classes = element.className
          .toString()
          .trim()
          .split(/\s+/)
          .filter(Boolean)
          .slice(0, 3)
          .join(".");
        currentAnchors.set(`${element.tagName.toLowerCase()}.${classes || "anchor"}-${index}`, element);
      });

      for (const anchor of anchorValues) {
        const element = currentAnchors.get(anchor.id);
        if (!element) {
          issues.push({
            kind: "persistent-surface",
            control: anchor.label,
            detail: `surface disappeared at ${stateValue.id}`
          });
          continue;
        }
        const rect = element.getBoundingClientRect();
        const delta = Math.max(
          Math.abs(rect.left - anchor.left),
          Math.abs(rect.top - anchor.top),
          Math.abs(rect.width - anchor.width),
          Math.abs(rect.height - anchor.height)
        );
        if (delta > tolerance) {
          issues.push({
            kind: "persistent-surface",
            control: anchor.label,
            detail: `moved or resized by ${delta.toFixed(1)}px at ${stateValue.id}`
          });
        }
        if (
          rect.left < -tolerance ||
          rect.right > innerWidth + tolerance ||
          rect.top < -tolerance ||
          rect.bottom > innerHeight + tolerance
        ) {
          issues.push({
            kind: "persistent-surface",
            control: anchor.label,
            detail: `bounds ${rect.left.toFixed(1)},${rect.top.toFixed(1)} ${rect.width.toFixed(1)}×${rect.height.toFixed(1)} leave viewport at ${stateValue.id}`
          });
        }
      }

      if (stateValue.id === "bottom") {
        const visibleBounds = (element: HTMLElement) => {
          const source = element.getBoundingClientRect();
          let left = Math.max(0, source.left);
          let top = Math.max(0, source.top);
          let right = Math.min(innerWidth, source.right);
          let bottom = Math.min(innerHeight, source.bottom);
          for (let parent = element.parentElement; parent; parent = parent.parentElement) {
            const style = getComputedStyle(parent);
            const rect = parent.getBoundingClientRect();
            if (["auto", "scroll", "hidden", "clip"].includes(style.overflowX)) {
              left = Math.max(left, rect.left);
              right = Math.min(right, rect.right);
            }
            if (["auto", "scroll", "hidden", "clip"].includes(style.overflowY)) {
              top = Math.max(top, rect.top);
              bottom = Math.min(bottom, rect.bottom);
            }
          }
          return { left, top, right, bottom, width: right - left, height: bottom - top };
        };
        const terminalSelectors = [
          ".library-bottom",
          ".builder-main__content > .builder-stage > :last-child",
          ".builder-footer",
          ".message-stream > :last-child",
          ".composer-wrap",
          ".data-page > :last-child",
          ".map-page > :last-child",
          ".trace-page > :last-child",
          ".imagine-page > :last-child",
          ".tools-page > :last-child",
          ".resource-settings-page > :last-child",
          ".agents-page > :last-child",
          ".evolution-page > :last-child",
          ".instance-delete-dialog__actions",
          ".appearance-panel > :last-child",
          ".live-perception-panel__body > :last-child"
        ];
        for (const selector of terminalSelectors) {
          for (const element of document.querySelectorAll<HTMLElement>(selector)) {
            if (element.closest("[hidden]")) continue;
            const style = getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            if (
              style.display === "none" ||
              style.visibility === "hidden" ||
              Number(style.opacity || "1") <= 0.01 ||
              rect.width <= 0 ||
              rect.height <= 0
            ) continue;
            const visible = visibleBounds(element);
            if (visible.width <= 2 || visible.height <= 2) {
              issues.push({
                kind: "end-content",
                control: selector,
                detail: "terminal content is not reachable in the viewport at the bottom scroll position"
              });
              continue;
            }
            const x = Math.max(0, Math.min(innerWidth - 1, visible.left + visible.width / 2));
            const y = Math.max(0, Math.min(innerHeight - 1, visible.top + visible.height / 2));
            const topmost = document.elementFromPoint(x, y);
            const coveredByDialog = Boolean(
              document.querySelector<HTMLElement>("[role='dialog']") &&
              !element.closest("[role='dialog']")
            );
            if (
              topmost &&
              !element.contains(topmost) &&
              !topmost.contains(element) &&
              !coveredByDialog
            ) {
              issues.push({
                kind: "end-content-cover",
                control: selector,
                detail: `terminal content is covered by ${topmost.tagName.toLowerCase()}.${(topmost as HTMLElement).className}`
              });
            }
          }
        }
      }

      return issues;
    },
    { stateValue: state, anchorValues: anchors }
  );
}

async function auditScrollableJourney(
  page: Page,
  matrix: string,
  screenshotPrefix?: string
): Promise<void> {
  const targets = await prepareScrollAudit(page);
  const anchors = await captureScrollAnchors(page);
  if (screenshotPrefix) {
    console.log(
      `UI_SCROLL_TARGETS ${matrix}: ${targets.map((target) => `${target.label}=${target.maxScroll}px`).join(", ") || "none"}`
    );
  }

  for (const state of scrollStates) {
    const movementIssues = await moveScrollAudit(page, state, anchors);
    expect(
      movementIssues,
      issueMessage(
        `${matrix}/scroll-${state.id}`,
        movementIssues
      ) + `\nScroll targets: ${targets.map((target) => `${target.label} (${target.maxScroll}px)`).join(", ") || "none"}`
    ).toEqual([]);
    await expectGeometry(page, `${matrix}/scroll-${state.id}`);
    if (screenshotPrefix) {
      await page.screenshot({
        animations: "disabled",
        path: resolve(
          artifactDirectory,
          `${artifactSlug(screenshotPrefix)}-${state.id}.png`
        )
      });
    }
  }

  await moveScrollAudit(page, scrollStates[0]!, anchors);
}

async function auditAppearancePanel(page: Page, matrix: string): Promise<void> {
  await page.getByRole("button", { name: "Appearance settings" }).click();
  const panel = page.getByRole("dialog", { name: "Appearance settings" });
  await expect(panel).toBeVisible();
  await auditScrollableJourney(page, `${matrix}/appearance-panel`);
  await panel.getByRole("button", { name: "Close appearance settings" }).click();
  await expect(panel).toBeHidden();
}

/**
 * Device permission prompts cannot be accepted in the headless matrix, so the
 * audited fixture mirrors the expanded, negotiated panel without fabricating a
 * media session. Controller behavior is covered by focused unit tests.
 */
async function mountLivePerceptionAuditFixture(page: Page): Promise<void> {
  await page.evaluate(() => {
    document.querySelector("[data-ui-audit-live-perception]")?.remove();
    const host = document.querySelector<HTMLElement>(".composer-wrap");
    if (!host) throw new Error("Chat composer was not found.");
    const panel = document.createElement("details");
    panel.open = true;
    panel.className = "live-perception-panel";
    panel.dataset.uiAuditLivePerception = "true";
    panel.innerHTML = `
      <summary>
        <span class="live-perception-panel__icon" aria-hidden="true"></span>
        <span><strong>Live Perception</strong><small>Camera · 1920×1080 @ 24 FPS</small></span>
        <em class="is-active"><i></i> Observing</em>
      </summary>
      <div class="live-perception-panel__body">
        <div class="live-perception-toolbar">
          <fieldset><legend>Source</legend>
            <button aria-pressed="true">Camera</button><button>Microphone</button><button>Screen</button><button>Mixed</button>
          </fieldset>
          <fieldset><legend>Quality</legend>
            <button aria-pressed="true">Auto</button><button>Motion</button><button>Balanced</button><button>Detail</button>
          </fieldset>
          <div class="live-perception-actions"><button>Stop</button><button class="is-danger">Cancel</button></div>
        </div>
        <div class="live-perception-facts">
          <span><small>Applied capture</small><strong>1920×1080 @ 24 FPS</strong></span>
          <span><small>Resource class</small><strong>High</strong></span>
          <span><small>Accepted</small><strong>1,248</strong></span>
          <span><small>Backpressure</small><strong>2</strong></span>
        </div>
        <p class="live-perception-control-status"><span>Brain configure · applied · 1920×1080 @ 24 FPS · 15s · r4</span></p>
        <details class="live-perception-advanced" open>
          <summary>Snapshot bursts, source bounds, and control audit</summary>
          <div>
            <section class="live-perception-snapshot">
              <div><strong>Snapshot burst</strong><small>Temporary visual samples enter this same brain only after every frame is accepted.</small></div>
              <fieldset><legend>Resolution</legend><button aria-pressed="true">Native</button><button>Current</button><button>Custom</button></fieldset>
              <div class="live-perception-burst-values">
                <label><span>Frames</span><input type="number" value="3" /></label>
                <label><span>Interval (ms)</span><input type="number" value="250" /></label>
              </div>
              <div class="live-perception-snapshot-actions"><button class="is-primary">Capture burst</button><button>Cancel burst</button><span>Capturing · 2/3 accepted</span></div>
            </section>
            <dl class="live-perception-audit">
              <div><dt>Source native</dt><dd>3840×2160 @ 60 FPS</dd></div>
              <div><dt>Capture revision</dt><dd>r4</dd></div>
              <div><dt>Transport window</dt><dd>4 in flight · 8,388,608 bytes/packet</dd></div>
              <div><dt>Auto plan</dt><dd>Scaled from measured resources and the source-native signal.</dd></div>
              <div><dt>Last control</dt><dd>Brain configure · applied</dd></div>
              <div><dt>Brain actions</dt><dd>ponder, learn</dd></div>
            </dl>
            <p class="live-perception-privacy">Raw packets are not stored. Accepted perception does not commit dataset coverage.</p>
          </div>
        </details>
      </div>`;
    host.prepend(panel);
  });
  await expect(page.locator("[data-ui-audit-live-perception]")).toBeVisible();
}

test("nested terminal coverage samples the ancestor-clipped visible area", async () => {
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.setViewportSize({ width: 360, height: 640 });
    await page.setContent(`
      <style>
        * { box-sizing: border-box; }
        html, body { width: 360px; height: 640px; margin: 0; overflow: hidden; }
        .message-stream {
          position: absolute;
          z-index: 1;
          inset: 0 0 auto;
          height: 120px;
          background: #dde4ef;
        }
        .composer-wrap {
          position: absolute;
          z-index: 2;
          top: 120px;
          left: 20px;
          width: 320px;
          height: 180px;
          overflow: auto;
        }
        .live-perception-panel__body {
          height: 180px;
          overflow: auto;
        }
        .fixture-terminal { height: 500px; background: #d8fff2; }
        .fixture-footer { height: 80px; }
      </style>
      <div class="message-stream">Conversation</div>
      <div class="composer-wrap" data-ui-audit-scroll-id="composer">
        <div class="live-perception-panel__body" data-ui-audit-scroll-id="perception">
          <div class="fixture-terminal">Terminal perception content</div>
        </div>
        <div class="fixture-footer"></div>
      </div>
    `);

    const issues = await moveScrollAudit(
      page,
      { id: "bottom", fraction: 1 },
      []
    );
    const samples = await page.evaluate(() => {
      const terminal = document.querySelector<HTMLElement>(".fixture-terminal")!;
      const source = terminal.getBoundingClientRect();
      const rawX = (Math.max(0, source.left) + Math.min(innerWidth, source.right)) / 2;
      const rawY = (Math.max(0, source.top) + Math.min(innerHeight, source.bottom)) / 2;
      let left = Math.max(0, source.left);
      let top = Math.max(0, source.top);
      let right = Math.min(innerWidth, source.right);
      let bottom = Math.min(innerHeight, source.bottom);
      for (let parent = terminal.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        const rect = parent.getBoundingClientRect();
        if (["auto", "scroll", "hidden", "clip"].includes(style.overflowX)) {
          left = Math.max(left, rect.left);
          right = Math.min(right, rect.right);
        }
        if (["auto", "scroll", "hidden", "clip"].includes(style.overflowY)) {
          top = Math.max(top, rect.top);
          bottom = Math.min(bottom, rect.bottom);
        }
      }
      const rawTopmost = document.elementFromPoint(rawX, rawY) as HTMLElement | null;
      const clippedTopmost = document.elementFromPoint(
        left + (right - left) / 2,
        top + (bottom - top) / 2
      ) as HTMLElement | null;
      return {
        rawTopmost: rawTopmost?.className ?? null,
        clippedTopmost: clippedTopmost?.className ?? null
      };
    });

    expect(samples.rawTopmost).toBe("message-stream");
    expect(samples.clippedTopmost).toBe("fixture-terminal");
    expect(issues, issueMessage("nested-terminal-clipping", issues)).toEqual([]);
  } finally {
    await application.close();
  }
});

test("all themes keep Library, every Build step, and chat usable at every supported resolution", async () => {
  test.setTimeout(900_000);
  mkdirSync(artifactDirectory, { recursive: true });
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");

    for (const viewport of supportedViewports) {
      await setViewport(page, viewport);
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const matrix = `${viewport.id}/${mode}/${pack.id}`;
          if (!matrixRequested(matrix)) continue;
          await test.step(matrix, async () => {
            await setAppearance(page, mode, pack);
            const captureScrollReferences =
              matrix === "mobile/light/classic" ||
              matrix === "design-target/dark/liquid-glass";
            const captureLivePerceptionScroll =
              captureScrollReferences || matrix === "floor/light/liquid-glass";
            await auditScrollableJourney(
              page,
              `${matrix}/library`,
              captureScrollReferences ? `${matrix}-library` : undefined
            );
            await auditAppearancePanel(page, matrix);

            await page.getByRole("button", { name: "Build a new brain" }).click();
            await expect(page.getByText("Step 1 of 4", { exact: true })).toBeVisible();
            await auditScrollableJourney(
              page,
              `${matrix}/build-identity`,
              captureScrollReferences ? `${matrix}-build-identity` : undefined
            );
            await page.getByRole("button", { name: "Continue" }).click();
            await expect(page.getByText("Step 2 of 4", { exact: true })).toBeVisible();
            await page.evaluate(() => window.scrollTo(0, 0));
            await auditScrollableJourney(
              page,
              `${matrix}/build-senses-data`,
              captureScrollReferences ? `${matrix}-build-senses-data` : undefined
            );
            await page.getByRole("button", { name: "Continue" }).click();
            await expect(page.getByText("Step 3 of 4", { exact: true })).toBeVisible();
            await page.evaluate(() => {
              document.querySelectorAll<HTMLDetailsElement>(
                ".system-ram-budget, .research-diagnostics"
              ).forEach((details) => {
                details.open = true;
              });
              window.scrollTo(0, 0);
            });
            await expect(page.locator(".system-ram-budget")).toHaveCount(2);
            await expect(page.locator(".system-ram-budget").first()).toHaveAttribute("open", "");
            await expect(page.locator(".research-diagnostics")).toHaveAttribute("open", "");
            await auditScrollableJourney(
              page,
              `${matrix}/build-memory-storage`,
              captureScrollReferences ? `${matrix}-build-memory-storage` : undefined
            );
            await page.getByRole("button", { name: "Continue" }).click();
            await expect(page.getByText("Step 4 of 4", { exact: true })).toBeVisible();
            await page.evaluate(() => {
              window.scrollTo(0, 0);
              document.documentElement.scrollTop = 0;
              document.body.scrollTop = 0;
            });
            await auditScrollableJourney(
              page,
              `${matrix}/build-access`,
              captureScrollReferences ? `${matrix}-build-access` : undefined
            );

            if (matrix === "floor/dark/default") {
              await page.screenshot({
                path: resolve(artifactDirectory, "floor-dark-default-build-access.png")
              });
            }

            await page.getByLabel("Open brain library").click();
            await openAster(page);
            await auditScrollableJourney(
              page,
              `${matrix}/conversation`,
              captureScrollReferences ? `${matrix}-conversation` : undefined
            );

            await mountLivePerceptionAuditFixture(page);
            await auditScrollableJourney(
              page,
              `${matrix}/live-perception-expanded`,
              captureLivePerceptionScroll ? `${matrix}-live-perception-expanded` : undefined
            );

            if (matrix === "design-target/dark/liquid-glass") {
              await page.screenshot({
                path: resolve(artifactDirectory, "desktop-dark-liquid-glass-live-perception.png")
              });
            }
          });
        }
      }
    }
  } finally {
    await application.close();
  }
});

test("measured context and placement stay legible across every theme", async () => {
  test.setTimeout(480_000);
  mkdirSync(artifactDirectory, { recursive: true });
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const resourceViewports = [
      { id: "resource-mobile", width: 390, height: 844, constrained: true },
      { id: "resource-desktop", width: 1_480, height: 940, constrained: false }
    ] as const;

    for (const viewport of resourceViewports) {
      await setViewport(page, viewport);
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const baseMatrix = `${viewport.id}/${mode}/${pack.id}`;
          if (!matrixRequested(baseMatrix)) continue;
          await test.step(baseMatrix, async () => {
            await setAppearance(page, mode, pack);
            await installMeasuredResourceAuditFixture(page, viewport.constrained);
            await page.getByRole("button", { name: "Build a new brain" }).click();
            await page.getByRole("button", { name: "Continue" }).click();
            await expect(page.getByText("Step 2 of 4", { exact: true })).toBeVisible();
            await page.getByRole("button", { name: "Files & datasets", exact: true }).click();
            await expect(page.getByText("Audit corpus.parquet", { exact: true })).toBeVisible();
            await page.getByRole("button", { name: "Remove Audit corpus.parquet from this build" }).click();
            await expect(page.getByText("No user resources selected.", { exact: false })).toBeVisible();
            await page.getByRole("button", { name: "Files & datasets", exact: true }).click();
            await page.getByRole("button", { name: "Continue" }).click();
            await expect(page.getByText("Step 3 of 4", { exact: true })).toBeVisible();
            await expect(page.locator(".memory-storage-inputs")).toContainText("6.0 GB local data");

            const floorTokens = viewport.constrained ? 8_192 : 32_768;
            const autoTokens = viewport.constrained ? 12_288 : 65_536;
            const maximumTokens = viewport.constrained ? 24_576 : 131_072;
            await expect(page.locator(".working-memory-planner__heading em")).toHaveText(
              `${autoTokens.toLocaleString()} tokens`
            );

            const placement = page.locator(
              '[aria-label="Core, active-context, and neural-memory placement"]'
            );
            await expect(placement).toHaveCount(1);
            await expect(placement).toBeVisible();
            await expect(placement).toContainText("Core + runtime");
            await expect(placement).toContainText("Active context");
            await expect(placement).toContainText("never paged to storage");
            await expect(placement).toContainText("Neural memory");
            await expect(placement).toContainText(
              viewport.constrained
                ? "Cold patterns use storage · about 18% slower when paged"
                : "No neural-memory storage paging"
            );
            await expect(page.locator(".working-memory-planner [title]")).toHaveCount(0);

            const manualMode = page.locator(".working-memory-planner__choices .is-manual");
            await expect(manualMode).toHaveCount(1);
            await manualMode.click();
            await expect(manualMode).toHaveAttribute("aria-pressed", "true");

            const manualInput = page.getByLabel("Active context tokens", { exact: true });
            const capacitySlider = page.getByLabel("Manual active-context size", { exact: true });
            await expect(manualInput).toHaveCount(1);
            await expect(capacitySlider).toHaveCount(1);
            await expect(capacitySlider).toHaveAttribute("min", "1");
            await expect(capacitySlider).toHaveAttribute("max", String(maximumTokens));
            await expect(capacitySlider).toHaveValue(String(autoTokens));
            await expect(page.locator(".working-memory-planner__manual > small")).toContainText(
              `Floor ${floorTokens.toLocaleString()} · Auto ${autoTokens.toLocaleString()} · resident/model maximum ${maximumTokens.toLocaleString()}`
            );

            const bands = await capacitySlider.evaluate((element) => {
              const style = getComputedStyle(element);
              const read = (name: string): number =>
                Number.parseFloat(style.getPropertyValue(name));
              const box = element.getBoundingClientRect();
              return {
                blackLow: read("--capacity-black-low"),
                redLow: read("--capacity-red-low"),
                orangeLow: read("--capacity-orange-low"),
                yellowLow: read("--capacity-yellow-low"),
                greenStart: read("--capacity-green-start"),
                greenEnd: read("--capacity-green-end"),
                yellowHigh: read("--capacity-yellow-high"),
                orangeHigh: read("--capacity-orange-high"),
                redHigh: read("--capacity-red-high"),
                height: box.height,
                background: style.backgroundImage
              };
            });
            expect(bands.blackLow).toBeLessThan(bands.redLow);
            expect(bands.redLow).toBeLessThan(bands.orangeLow);
            expect(bands.orangeLow).toBeLessThan(bands.yellowLow);
            expect(bands.yellowLow).toBeLessThanOrEqual(bands.greenStart);
            expect(bands.greenStart).toBeLessThan(bands.greenEnd);
            expect(bands.greenEnd).toBeLessThan(bands.yellowHigh);
            expect(bands.yellowHigh).toBeLessThan(bands.orangeHigh);
            expect(bands.orangeHigh).toBeLessThan(bands.redHigh);
            expect(bands.redHigh).toBeLessThanOrEqual(100);
            expect(bands.background).toContain("linear-gradient");
            expect(bands.height).toBeGreaterThanOrEqual(viewport.constrained ? 36 : 24);

            await manualInput.fill(String(Math.floor(floorTokens / 2)));
            await expect(page.locator(".working-memory-planner__status")).toContainText(
              "Build is locked until this fits"
            );
            await expect(manualInput).toHaveAttribute("aria-invalid", "true");
            await expect(page.getByRole("button", { name: /Continue/ })).toBeDisabled();
            await manualInput.fill(String(autoTokens));
            await expect(page.locator(".working-memory-planner__status")).toContainText(
              viewport.constrained
                ? "Cold memory may use storage"
                : "Fits inside the Omni RAM envelope"
            );
            await expect(manualInput).toHaveAttribute("aria-invalid", "false");
            await expect(page.getByRole("button", { name: /Continue/ })).toBeEnabled();

            const sharedPool = page.locator("details.storage-pool-budget");
            await sharedPool.locator("summary").click();
            await sharedPool.getByRole("button", { name: "Custom size" }).click();
            const poolInput = sharedPool.getByLabel("Build shared pool capacity");
            await expect(poolInput).toHaveValue(viewport.constrained ? "48" : "128");
            await expect(sharedPool).toContainText(
              "The 20.0 GB free-space reserve remains outside this pool."
            );
            await expect(sharedPool).toContainText(
              viewport.constrained
                ? "Required 24.0 GB · physical maximum 180.0 GB"
                : "Required 64.0 GB · physical maximum 1.17 TB"
            );
            await poolInput.fill("1");
            await expect(page.locator(".working-memory-planner__status")).toContainText(
              "Build is locked until this fits"
            );
            await expect(page.getByRole("button", { name: /Continue/ })).toBeDisabled();
            await poolInput.fill(viewport.constrained ? "48" : "128");
            await expect(page.getByRole("button", { name: /Continue/ })).toBeEnabled();

            await page.evaluate(() => {
              document.querySelectorAll<HTMLDetailsElement>(".system-ram-budget")
                .forEach((details) => { details.open = true; });
            });
            await auditScrollableJourney(
              page,
              `${baseMatrix}/measured-context`,
              `${baseMatrix}-measured-context`
            );

            await page.getByLabel("Open brain library").click();
            await expect(page.getByRole("button", { name: "Build a new brain" })).toBeVisible();
            await expect(page.locator("#root")).not.toBeEmpty();
            await expect.poll(() => pageErrors).toEqual([]);

            await page.evaluate(() => {
              Reflect.deleteProperty(window, "omni");
            });
            await openAster(page);
            await installMeasuredResourceAuditFixture(page, viewport.constrained);
            const workspace = page.getByLabel("Workspace");
            await workspace.getByRole("button", { name: "Device & runtime", exact: true }).click();
            await expect.poll(() => pageErrors).toEqual([]);
            const settings = page.locator(".resource-settings-page");
            await expect(settings).toBeVisible();
            await expect(settings).toContainText("Physical RAM");
            await expect(settings).toContainText("Free or reclaimable now");
            await expect(settings).toContainText("Saved Omni RAM capacity");
            await expect(settings).toContainText("Available inside capacity now");
            await expect(settings).toContainText("Temporarily occupied by other apps");
            await expect(settings).not.toContainText("NaN");
            await expect(settings).not.toContainText("undefined");

            const extendedMode = settings.getByRole("button", { name: /^Extended/ });
            await extendedMode.click();
            await expect(extendedMode).toHaveAttribute("aria-pressed", "true");
            const manualContextMode = settings.getByRole("button", { name: /^Manual/ });
            await manualContextMode.click();
            await expect(manualContextMode).toHaveAttribute("aria-pressed", "true");
            const contextInput = settings.getByLabel("Active context tokens", { exact: true });
            const contextSlider = settings.getByLabel("Resident active-context tokens", { exact: true });
            await expect(contextInput).toBeVisible();
            await expect(contextSlider).toHaveAttribute("max", String(maximumTokens));
            await contextInput.fill(String(autoTokens));
            await expect(settings.locator(".resource-settings-preflight")).not.toHaveClass(/is-pending/);

            const customStorageMode = settings.getByRole("button", { name: /^Custom size/ });
            await customStorageMode.click();
            await expect(customStorageMode).toHaveAttribute("aria-pressed", "true");
            const storageInput = settings.getByLabel("Shared pool capacity", { exact: true });
            const storageSlider = settings.getByLabel("Shared storage pool in GiB", { exact: true });
            const requiredStorageGiB = viewport.constrained ? 24 : 64;
            const maximumStorageGiB = viewport.constrained ? 180 : 1_200;
            await expect(storageInput).toBeVisible();
            await expect(storageSlider).toHaveAttribute("min", String(requiredStorageGiB));
            await expect(storageSlider).toHaveAttribute("max", String(maximumStorageGiB));
            await storageInput.fill(String(requiredStorageGiB));
            await expect(settings.locator(".resource-settings-preflight")).not.toHaveClass(/is-pending/);
            await expect(settings.locator(".resource-settings-preflight")).not.toHaveClass(/is-blocked/);

            await auditScrollableJourney(
              page,
              `${baseMatrix}/device-resource-controls`,
              baseMatrix === "resource-mobile/light/classic" ||
                baseMatrix === "resource-desktop/dark/liquid-glass"
                ? `${baseMatrix}-device-resource-controls`
                : undefined
            );
          });
        }
      }
    }
  } finally {
    await application.close();
  }
});

test("every in-brain workspace, dialog, rail label, and layout arrangement stays reachable", async () => {
  test.setTimeout(900_000);
  mkdirSync(artifactDirectory, { recursive: true });
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");

    for (const viewport of supportedViewports) {
      await setViewport(page, viewport);
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const matrix = `${viewport.id}/${mode}/${pack.id}`;
          if (!matrixRequested(matrix)) continue;
          await setAppearance(page, mode, pack);
          await openAster(page);
          const workspace = page.getByLabel("Workspace");
          await expect(
            workspace.locator("button[title]"),
            `${matrix} workspace navigation must use bounded in-app labels instead of native tooltips`
          ).toHaveCount(0);
          const captureScrollReferences =
            matrix === "mobile/light/classic" ||
            matrix === "design-target/dark/liquid-glass";

          for (const view of workspaceViews) {
            await test.step(`${matrix}/${view.label}`, async () => {
              const navigation = workspace.getByRole("button", {
                name: view.label,
                exact: true
              });
              await expect(navigation).toHaveCount(1);
              await navigation.click();
              await expect(page.locator(".workspace-header__view")).toHaveText(view.label);
              await expect(page.locator(view.marker)).toBeVisible();
              if (viewport.width <= 720) {
                await expect(
                  navigation.locator(".rail-label"),
                  `${matrix}/${view.label} compact navigation label must stay inside its accessible icon control`
                ).toBeHidden();
              }
              await auditScrollableJourney(
                page,
                `${matrix}/${view.label}`,
                captureScrollReferences ? `${matrix}-${view.label}` : undefined
              );
            });
          }

          const deleteButton = page.getByRole("button", {
            name: "Delete instance Aster",
            exact: true
          });
          await deleteButton.click();
          const deleteDialog = page.getByRole("dialog", { name: "Delete instance Aster?" });
          await expect(deleteDialog).toBeVisible();
          await auditScrollableJourney(page, `${matrix}/delete-dialog-1`);
          await deleteDialog.getByRole("button", { name: "I understand, continue" }).click();
          await deleteDialog.getByLabel("Exact instance name").fill("Aster");
          await auditScrollableJourney(page, `${matrix}/delete-dialog-2`);
          await deleteDialog.getByRole("button", { name: "Name matches, continue" }).click();
          await auditScrollableJourney(page, `${matrix}/delete-dialog-3`);
          await deleteDialog.getByRole("button", { name: "Cancel" }).click();
          await expect(deleteDialog).toBeHidden();

          await workspace
            .getByRole("button", { name: "Conversation", exact: true })
            .click();

          if (viewport.width > 720) {
            const railButtons = workspace.getByRole("button");
            const count = await railButtons.count();
            for (let index = 0; index < count; index += 1) {
              const button = railButtons.nth(index);
              await button.hover();
              const label = button.locator(".rail-label");
              await expect(label).toBeVisible();
              const labelBox = await label.boundingBox();
              expect(labelBox, `${matrix} rail label ${index + 1} must have bounds`).not.toBeNull();
              expect(labelBox!.x).toBeGreaterThanOrEqual(0);
              expect(labelBox!.x + labelBox!.width).toBeLessThanOrEqual(viewport.width + 1);
              expect(labelBox!.y).toBeGreaterThanOrEqual(0);
              expect(labelBox!.y + labelBox!.height).toBeLessThanOrEqual(viewport.height + 1);
            }
          }

          for (const arrangement of workspaceArrangements) {
            await page.getByRole("button", { name: "Appearance settings" }).click();
            const panel = page.getByRole("dialog", { name: "Appearance settings" });
            const arrangementButton = panel.getByRole("button", {
              name: arrangement.label,
              exact: true
            });
            await arrangementButton.click();
            await expect(page.locator("html")).toHaveAttribute(
              "data-workspace-arrangement",
              arrangement.value
            );
            await panel.getByRole("button", { name: "Close appearance settings" }).click();
            await auditScrollableJourney(
              page,
              `${matrix}/arrangement-${arrangement.value}`
            );
          }

          if (matrix === "mobile/light/classic") {
            await page.screenshot({
              path: resolve(artifactDirectory, "mobile-light-classic-chat.png")
            });
          }
        }
      }
    }
  } finally {
    await application.close();
  }
});

test("text scaling does not clip controls in any layout pack", async () => {
  test.setTimeout(360_000);
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    await application.evaluate(({ BrowserWindow }) => {
      BrowserWindow.getAllWindows()[0]?.webContents.setZoomFactor(1);
    });
    await setViewport(page, { width: 1_440, height: 900 });

    for (const pack of packs) {
      await setAppearance(page, "dark", pack);
      await openAster(page);
      for (const scale of [1.25, 1.5, 2]) {
        await application.evaluate(({ BrowserWindow }, zoomFactor) => {
          BrowserWindow.getAllWindows()[0]?.webContents.setZoomFactor(zoomFactor);
        }, scale);
        await expectGeometry(page, `${pack.id}/text-scale-${scale}`);
      }
      await application.evaluate(({ BrowserWindow }) => {
        BrowserWindow.getAllWindows()[0]?.webContents.setZoomFactor(1);
      });
    }
  } finally {
    await application.evaluate(({ BrowserWindow }) => {
      BrowserWindow.getAllWindows()[0]?.webContents.setZoomFactor(1);
    });
    await application.close();
  }
});

test("visual review references cover every pack and mode from mobile through desktop", async () => {
  test.setTimeout(360_000);
  mkdirSync(artifactDirectory, { recursive: true });
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");

    for (const mode of ["light", "dark"] as const) {
      for (const pack of packs) {
        await setViewport(page, supportedViewports[4]);
        await setAppearance(page, mode, pack);
        await openAster(page);
        await page.getByRole("button", { name: "Tools & permissions", exact: true }).hover();
        await page.screenshot({
          animations: "disabled",
          path: resolve(artifactDirectory, `${pack.id}-${mode}-desktop-conversation.png`)
        });
        await page.getByRole("button", { name: "Brain map", exact: true }).click();
        await expect(page.locator(".map-page")).toBeVisible();
        await page.screenshot({
          animations: "disabled",
          path: resolve(artifactDirectory, `${pack.id}-${mode}-desktop-brain-map.png`)
        });

        await setViewport(page, supportedViewports[1]);
        await setAppearance(page, mode, pack);
        await openAster(page);
        await page.getByRole("button", { name: "Brain map", exact: true }).click();
        await expect(page.locator(".map-page")).toBeVisible();
        await page.screenshot({
          animations: "disabled",
          path: resolve(artifactDirectory, `${pack.id}-${mode}-mobile-brain-map.png`)
        });
      }
    }
  } finally {
    await application.close();
  }
});

test("focused and highlighted edge controls keep their geometry in every theme", async () => {
  test.setTimeout(360_000);
  mkdirSync(artifactDirectory, { recursive: true });
  const application = await launchAuditApplication();
  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    const edgeViewports = [supportedViewports[1], supportedViewports[4]] as const;

    for (const viewport of edgeViewports) {
      await setViewport(page, viewport);
      for (const mode of ["light", "dark"] as const) {
        for (const pack of packs) {
          const matrix = `${viewport.id}/${mode}/${pack.id}`;
          await setAppearance(page, mode, pack);
          await openAster(page);
          const mapNavigation = page.getByRole("button", { name: "Brain map", exact: true });
          await mapNavigation.click();
          await expect(page.locator(".map-page")).toBeVisible();
          await expect(mapNavigation).toHaveClass(/is-active/);

          const controls = [
            page.getByRole("button", { name: "Appearance settings" }),
            page.getByRole("button", { name: "Delete instance Aster" }),
            mapNavigation,
            page.getByRole("button", { name: "Conversation", exact: true }).first()
          ];
          for (const control of controls) {
            const before = await control.boundingBox();
            expect(before, `${matrix} focus target must have geometry`).not.toBeNull();
            // Establish keyboard modality before moving focus to a specific
            // edge target so Chromium applies :focus-visible consistently.
            await page.keyboard.press("Tab");
            await control.focus();
            const focused = await control.evaluate((element) => {
              const style = getComputedStyle(element);
              const rect = element.getBoundingClientRect();
              return {
                focusVisible: element.matches(":focus-visible"),
                outlineStyle: style.outlineStyle,
                outlineWidth: Number.parseFloat(style.outlineWidth),
                left: rect.left,
                right: rect.right,
                top: rect.top,
                bottom: rect.bottom,
                width: rect.width,
                height: rect.height
              };
            });
            expect(focused.focusVisible, `${matrix} target must enter :focus-visible`).toBe(true);
            expect(focused.outlineStyle, `${matrix} focus ring style`).not.toBe("none");
            expect(focused.outlineWidth, `${matrix} focus ring width`).toBeGreaterThanOrEqual(2);
            expect(Math.abs(focused.width - before!.width), `${matrix} focus must not resize width`).toBeLessThan(0.6);
            expect(Math.abs(focused.height - before!.height), `${matrix} focus must not resize height`).toBeLessThan(0.6);
            expect(focused.left).toBeGreaterThanOrEqual(-0.6);
            expect(focused.right).toBeLessThanOrEqual(viewport.width + 0.6);
            expect(focused.top).toBeGreaterThanOrEqual(-0.6);
            expect(focused.bottom).toBeLessThanOrEqual(viewport.height + 0.6);
          }

          const selectedVisual = await mapNavigation.evaluate((element) => {
            const style = getComputedStyle(element);
            const marker = getComputedStyle(element, "::before");
            return {
              background: style.backgroundColor,
              markerWidth: Number.parseFloat(marker.width),
              markerHeight: Number.parseFloat(marker.height)
            };
          });
          expect(
            selectedVisual.background !== "rgba(0, 0, 0, 0)" ||
              selectedVisual.markerWidth > 0 ||
              selectedVisual.markerHeight > 0,
            `${matrix} selected navigation must remain visibly highlighted`
          ).toBe(true);

          if (matrix === "mobile/dark/default") {
            await mapNavigation.focus();
            await page.screenshot({
              animations: "disabled",
              path: resolve(artifactDirectory, "mobile-dark-default-focused-brain-map.png")
            });
          }
        }
      }
    }
  } finally {
    await application.close();
  }
});
