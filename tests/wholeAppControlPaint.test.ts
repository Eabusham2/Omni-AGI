import { describe, expect, it } from "vitest";
import { adjudicateLowParentPaint, type AuditInkPaint, type ControlContentsPaint } from "./e2e/control-paint-evidence";

const ink = (kind: "text" | "graphic", contrast: number | null, change: Partial<AuditInkPaint> = {}): AuditInkPaint => ({
  kind, target: kind === "text" ? ":scope > span" : ":scope > svg > path", channel: kind === "text" ? "text-fill" : "svg-stroke",
  color: "rgb(0, 0, 0)", background: "rgba(255,255,255,1)", opacity: "1", contrast,
  contrastScope: contrast === null ? "unmeasured" : "estimated-solid-composite", ...change
});
const contents = (textRuns: AuditInkPaint[] = [ink("text", 9)], graphics: AuditInkPaint[] = [ink("graphic", 4.145)]): ControlContentsPaint => ({
  format: "visible-control-ink-v1", complete: true, textRuns, graphics, unmeasured: []
});

describe("whole-App low-parent paint adjudication (inert measured-data fixtures)", () => {
  it("does not change the already-passing or originally-unmeasured parent path", () => {
    for (const original of [4.5, 7, null]) expect(adjudicateLowParentPaint(original, contents([ink("text", 1)], [ink("graphic", 1)])))
      .toMatchObject({ examined: false, qualified: false, failures: [] });
  });

  it("retains original paint while qualifying genuinely readable nested text plus 3:1 graphics", () => {
    const measured = contents([ink("text", 9)], [ink("graphic", 4.145), ink("graphic", 3.9, {
      channel: "pseudo-background", target: ":scope::before", contrastScope: "conservative-solid-ancestor-envelope"
    })]);
    const original = JSON.stringify(measured);
    expect(adjudicateLowParentPaint(4.145, measured)).toEqual({ examined: true, qualified: true, textMinimum: 4.5, graphicMinimum: 3, failures: [] });
    expect(JSON.stringify(measured)).toBe(original);
  });

  it("still fails direct text below 4.5 rather than treating all parent paint as icon ink", () => {
    const result = adjudicateLowParentPaint(4.145, contents([ink("text", 4.145)], []));
    expect(result.qualified).toBe(false);
    expect(result.failures.join(" ")).toContain("requires 4.5:1");
  });

  it("fails readable nested text plus a graphic below 3:1", () => {
    expect(adjudicateLowParentPaint(4.1, contents([ink("text", 12)], [ink("graphic", 2.999)])).qualified).toBe(false);
  });

  it("keeps exact independent text 4.5 and graphic 3 boundaries", () => {
    expect(adjudicateLowParentPaint(3.5, contents([ink("text", 4.5)], [ink("graphic", 3)])).qualified).toBe(true);
    expect(adjudicateLowParentPaint(3.5, contents([ink("text", 4.499)], [ink("graphic", 3)])).qualified).toBe(false);
    expect(adjudicateLowParentPaint(3.5, contents([], [ink("graphic", 3)])).qualified).toBe(true);
  });

  it("never clears the original failure for absent, empty, partial, filtered or otherwise unmeasured paint", () => {
    const cases: Array<ControlContentsPaint | undefined> = [undefined, contents([], []), { ...contents(), complete: false },
      { ...contents(), unmeasured: ["generated text/image cannot be measured"] },
      contents([ink("text", null)]), contents([ink("text", 8)], [ink("graphic", null)]),
      contents([ink("text", Number.NaN)]), contents([ink("text", 8, { contrastScope: "unmeasured" })])];
    for (const value of cases) expect(adjudicateLowParentPaint(4.1, value).qualified).toBe(false);
  });

  it("does not use the solid-decoration envelope to relax text measurement", () => {
    expect(adjudicateLowParentPaint(4.1, contents([ink("text", 8, { contrastScope: "conservative-solid-ancestor-envelope" })])).qualified).toBe(false);
  });

  it("rejects a corrupted kind or invalid original ratio rather than selecting the lower threshold", () => {
    expect(adjudicateLowParentPaint(4.1, contents([ink("graphic", 3.5)], [])).qualified).toBe(false);
    expect(adjudicateLowParentPaint(Number.NaN, contents()).qualified).toBe(false);
    expect(adjudicateLowParentPaint(0, contents()).qualified).toBe(false);
  });

  it("does not call a zero-area decorative shadow measured text or graphic contrast", () => {
    const decoration = { target: ":scope::before", reason: "zero-area-empty-pseudo-background" as const,
      width: 0, height: 2, boxShadow: "rgba(8,127,122,.68) 0px 0px 7px", contrastClaimed: false as const };
    expect(adjudicateLowParentPaint(4.145, { ...contents([], []), excludedDecorations: [decoration] }).qualified).toBe(false);
    expect(adjudicateLowParentPaint(4.145, { ...contents([], [ink("graphic", 4.145)]), excludedDecorations: [decoration] }).qualified).toBe(true);
    expect(decoration.contrastClaimed).toBe(false);
  });
});
