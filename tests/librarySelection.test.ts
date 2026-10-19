import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import type { BrainSummary } from "../src/shared/types";
import { librarySelection, selectVisibleLibraryBrains, toggleLibrarySelection } from "../src/renderer/src/librarySelection";

const brain = (id: string): BrainSummary => ({ id, name: id, preset: "whole-brain", runtime: "adaptive-core", updatedAt: "2026-10-18",
  concepts: 0, synapses: 0, activeMode: false, generation: 1 });

describe("Home selected-brain management", () => {
  it("keeps hidden search selections while Select all adds only the current results", () => {
    const first = brain("first"), second = brain("second"), third = brain("third");
    const selected = selectVisibleLibraryBrains(new Set([first.id]), [second]);
    expect([...selected]).toEqual([first.id, second.id]);
    expect(librarySelection([first, second, third], selected, [second])).toMatchObject({
      selected: [first, second], hiddenSelected: 1, allVisibleSelected: true, canMerge: true
    });
    expect(librarySelection([first, second, third], selected, [third]).allVisibleSelected).toBe(false);
    expect(librarySelection([first, second], selected, []).allVisibleSelected).toBe(false);
  });

  it("toggles exact IDs without modifying a prior selection", () => {
    const previous = new Set(["same-name-one"]);
    const next = toggleLibrarySelection(previous, "same-name-two");
    expect([...next]).toEqual(["same-name-one", "same-name-two"]);
    expect([...previous]).toEqual(["same-name-one"]);
    expect([...toggleLibrarySelection(next, "same-name-one")]).toEqual(["same-name-two"]);
  });

  it("does not send stale or unselected identities to management callbacks", () => {
    const first = brain("one"), second = brain("two");
    expect(librarySelection([first, second], new Set(["removed", second.id])).selected).toEqual([second]);
    expect(librarySelection([first, second], new Set()).selected).toEqual([]);
  });

  it("requires at least two native candidates without claiming architecture compatibility has already passed", () => {
    const first = brain("one"), second = brain("two");
    expect(librarySelection([first, second], new Set([first.id])).canMerge).toBe(false);
    expect(librarySelection([first, first], new Set([first.id])).canMerge).toBe(false);
    expect(librarySelection([first, second], new Set([first.id, second.id])).canMerge).toBe(true);
    const unsupported = { ...second, runtime: "unsupported" as BrainSummary["runtime"] };
    expect(librarySelection([first, unsupported], new Set([first.id, second.id]))).toMatchObject({
      canMerge: false, mergeReason: "Only native adaptive-core brains can merge."
    });
  });

  it("keeps destructive confirmation and native export in parent callbacks, with no duplicate copy action or portable panel", async () => {
    const source = await readFile(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    const library = source.slice(source.indexOf("function LibraryPage("), source.indexOf("function DeleteInstanceDialog("));
    expect(library).toContain("onDelete([...selection.selected])");
    expect(library).toContain('mode: BrainExportMode = "current"');
    expect(library).toContain("onExport(captured, mode)");
    expect(library).not.toContain(".brain.remove(");
    expect(library).not.toContain("portableControls");
    expect(library).not.toContain("Review a new merge");
    expect(library).not.toContain("brain-card__duplicate");
    expect(library).toContain('aria-label="Manage selected brains"');
    expect(library).toContain('role={selecting ? "checkbox" : "button"}');
  });

  it("keeps compact Home export choices on the frozen selected array through existing export modes", async () => {
    const source = await readFile(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    const library = source.slice(source.indexOf("function LibraryPage("), source.indexOf("function DeleteInstanceDialog("));
    expect(library).toContain('<details className="library-export-menu"');
    for (const mode of ["current", "origin", "referenced"]) expect(library).toContain(`manage("export", "${mode}")`);
    for (const label of ["Current brain", "Original origin", "Local reference"]) expect(library).toContain(label);
    expect(library).toContain('mode: BrainExportMode = "current"');
    expect(library).toContain("const captured = [...selection.selected]");
    expect(library).toContain("actionInFlight.current || busy || !selection.selected.length");
    expect(library).toContain("onExport(captured, mode)");
    expect(library).not.toContain(".brain.export(");
  });
});
