import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const stylesPath = resolve(process.cwd(), "src/renderer/src/styles.css");
const appPath = resolve(process.cwd(), "src/renderer/src/App.tsx");
const styles = readFileSync(stylesPath, "utf8");
const app = readFileSync(appPath, "utf8");

describe("UI visual integrity", () => {
  it("keeps explicit pixel typography at or above the readability floor", () => {
    const offenders: string[] = [];
    for (const match of styles.matchAll(/font-size:\s*([0-9.]+)px/g)) {
      if (Number(match[1]) < 10) offenders.push(match[0]);
    }
    for (const match of styles.matchAll(/font:\s*([0-9.]+)px(?:\/[0-9.]+)?\s/g)) {
      if (Number(match[1]) < 10) offenders.push(match[0].trim());
    }
    expect(offenders).toEqual([]);
  });

  it("uses theme tokens for graph ink and separate in-flow map-area pagination", () => {
    expect(styles).toContain("--graph-node-ink:");
    expect(styles).toContain("--graph-label:");
    expect(styles).toContain("--graph-edge-positive:");
    expect(styles).toMatch(/\.button\s*\{[^}]*min-height:\s*36px;/);
    const map = readFileSync(resolve(process.cwd(), "src/renderer/src/WholeBrainMap.tsx"), "utf8");
    const mapStyles = readFileSync(resolve(process.cwd(), "src/renderer/src/wholeBrainMap.css"), "utf8");
    expect(mapStyles).toMatch(/\.whole-brain-map__navigation\s*\{[^}]*display:\s*flex;[^}]*flex-wrap:\s*wrap;/);
    expect(mapStyles.match(/\.whole-brain-map__navigation\s*\{([^}]+)\}/)?.[1]).not.toMatch(/position:\s*(?:absolute|fixed)/);
    const graph = readFileSync(resolve(process.cwd(), "src/renderer/src/BrainMapSubstrateGraph.tsx"), "utf8");
    expect(graph).toContain('className={`graph-edge ${');
    expect(graph).toContain('"graph-edge--negative"');
    expect(graph).toContain('"graph-edge--positive"');
    expect(map).toContain('className="whole-brain-map__navigation"');
    expect(map).toContain('aria-label="Previous map area"');
    expect(map).toContain('aria-label="Next map area"');
    expect(map).toContain("if (page.hasMore && page.nextCursor)");
  });

  it("anchors jump-to-latest inside the message viewport instead of the composer", () => {
    expect(app).toContain('className="message-stream-wrap"');
    expect(styles).toMatch(/\.message-stream-wrap\s*\{[^}]*overflow:\s*hidden;/s);
    expect(styles).toMatch(/\.chat-jump-latest\s*\{[^}]*bottom:\s*12px;/s);
  });

  it("keeps compact neural totals and response activity readable", () => {
    expect(styles).toMatch(/\.memory-continuum-totals\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
    expect(styles).toMatch(/\.response-token-counter\s*\{[^}]*font-size:\s*10\.5px;/s);
    expect(styles).toMatch(/\.chat-action-card__summary-copy small\s*\{[^}]*text-overflow:\s*ellipsis;/s);
  });

  it("keeps media output mode controls inside the compact hit-area floor", () => {
    expect(styles).toMatch(
      /\.media-output-mode button\s*\{[^}]*min-height:\s*36px;/
    );
  });

  it("uses theme surfaces behind toast and loading text in both color schemes", () => {
    expect(styles).toMatch(/\.toast\s*\{[^}]*background:\s*color-mix\(in srgb, var\(--panel-raised\)/s);
    expect(styles).toMatch(/\.loading-overlay\s*\{[^}]*background:\s*color-mix\(in srgb, var\(--canvas\)/s);
  });

  it("defines status and elevation tokens consumed by visible controls", () => {
    for (const token of ["success", "warning", "danger", "text-muted", "shadow-lg", "shadow-xl", "radius-lg"]) {
      expect(styles).toMatch(new RegExp(`^\\s*--${token}:`, "m"));
    }
  });

  it("shows authoritative neural parameter totals in a responsive text badge, not a broken gauge", () => {
    expect(app).toContain('className="neural-parameter-count"');
    expect(app).toContain("<NeuralParameterCount accounting={chatParameterAccounting}");
    expect(app).toContain("accounting.totalNeuralParameters.toLocaleString");
    expect(styles).toMatch(/\.neural-parameter-count\s*\{[^}]*max-width:\s*100%;[^}]*font-variant-numeric:\s*tabular-nums;/s);
    expect(styles).not.toContain("--health-score");
  });

  it("keeps storage progress reachable above compact-window safe areas", () => {
    expect(styles).toMatch(/\.storage-operation-card\s*\{[^}]*max-height:\s*calc\(100dvh/);
    expect(styles).toMatch(/\.storage-operation-card\s*\{[^}]*overflow-y:\s*auto;/s);
    expect(styles).toMatch(/@media \(max-width: 640px\)\s*\{\s*\.storage-operation-card\s*\{[^}]*var\(--safe-bottom\)/s);
  });
});
