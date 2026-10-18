import { useEffect, useMemo, useRef, useState } from "react";
import type { PonderPresentation } from "./ponderSelfReportPresentation";
import "./ponderSelfReport.css";

/** Only visible publication is paced. No calls into model/control APIs. */
export function PonderSelfReportView({ row, readable, onReadableChange, flush = false }: {
  row: PonderPresentation; readable: boolean; onReadableChange(value: boolean): void; flush?: boolean;
}) {
  const points = useMemo(() => Array.from(row.report.text), [row.report.text]);
  const [shown, setShown] = useState(() => readable && row.live && !flush ? 0 : points.length);
  const shownRef = useRef(shown);
  useEffect(() => {
    if (!readable || !row.live || flush) { shownRef.current = points.length; setShown(points.length); return; }
    let handle = 0, previous = performance.now(), credit = 0, retired = false;
    const frame = (now: number) => {
      if (retired) return;
      credit += Math.max(0, now - previous) * 0.04; previous = now;
      const count = Math.floor(credit); credit -= count;
      if (count) { shownRef.current = Math.min(points.length, shownRef.current + count); setShown(shownRef.current); }
      if (shownRef.current < points.length) handle = window.requestAnimationFrame(frame);
    };
    if (shownRef.current < points.length) handle = window.requestAnimationFrame(frame);
    return () => { retired = true; window.cancelAnimationFrame(handle); };
  }, [points.length, readable, row.live, flush]);
  const text = !readable || !row.live || flush ? row.report.text : points.slice(0, shown).join("");
  return <section className="ponder-self-report" aria-label="Ponder self-report">
    <div className="ponder-self-report__heading">
      <small>Self-report{row.report.state === "partial" ? " · published excerpt" : ""}</small>
      <button type="button" className="button" aria-pressed={readable} onClick={() => onReadableChange(!readable)}
        title="Slow the display of emitted Ponder text; computation continues at its normal speed.">Readable speed</button>
    </div>
    <p className="ponder-self-report__text">{text}</p>
  </section>;
}
