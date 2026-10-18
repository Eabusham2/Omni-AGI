import { useMemo, useRef, useState } from "react";
import type { AutomationProgram, AutomationStep } from "../../shared/automation";
import { ACTION_NODE_HEIGHT, ACTION_NODE_WIDTH, actionCanvasMode, actionPositions, actionSequenceCurve, addActionNode, moveActionNode, removeActionNode, reorderActionStep } from "./actionCanvas";
import "./actionCanvas.css";

/** Controlled inert authoring. Save/Run remain separate host-owned operations. */
export function ActionCanvasEditor({ program, disabled, onChange, onValidityChange }: {
  program: AutomationProgram; disabled?: boolean; onChange(value: AutomationProgram): void; onValidityChange(valid: boolean): void;
}) {
  const [selectedId, setSelectedId] = useState(program.steps[0]?.id ?? "");
  const [error, setError] = useState("");
  const [validArguments, setValidArguments] = useState(true);
  const drag = useRef<{ id: string; pointer: number; startX: number; startY: number; x: number; y: number } | undefined>(undefined);
  const editingBlocked = disabled || !validArguments;
  const mode = program.canvas?.mode ?? "grid";
  const selected = program.steps.find(step => step.id === selectedId) ?? program.steps[0];
  const selectedIndex = selected ? program.steps.indexOf(selected) : -1;
  const positions = useMemo(() => actionPositions(program), [program]);
  const change = (value: AutomationProgram) => { setError(""); onChange(value); };
  const updateStep = (value: AutomationStep) => change({ ...program, steps: program.steps.map(step => step.id === value.id ? value : step) });
  const safely = (work: () => void) => { try { work(); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } };
  const width = positions.reduce((max, point) => Math.max(max, point.x + ACTION_NODE_WIDTH + 64), 816);
  const height = positions.reduce((max, point) => Math.max(max, point.y + ACTION_NODE_HEIGHT + 24), 280);
  return <section className="action-canvas-editor" aria-label="Action canvas editor">
    <div className="automation-buttons" role="group" aria-label="Canvas view">
      {(["grid", "freeform"] as const).map(value => <button key={value} type="button" aria-pressed={mode === value} disabled={editingBlocked} onClick={() => change(actionCanvasMode(program, value))}>{value === "grid" ? "Grid" : "Freeform"}</button>)}
      <button type="button" disabled={editingBlocked} onClick={() => { const next = addActionNode(program, "tool"); setSelectedId(next.steps.at(-1)!.id); change(next); }}>Add tool</button>
      <button type="button" disabled={editingBlocked} onClick={() => { const next = addActionNode(program, "wait"); setSelectedId(next.steps.at(-1)!.id); change(next); }}>Add wait</button>
    </div>
    <label>Name<input disabled={editingBlocked} value={program.name} onChange={event => change({ ...program, name: event.target.value })} /></label>
    <p className="action-canvas-caption">Connections show step order. Editing does not run any tools.</p>
    <div className={`action-canvas-viewport action-canvas-viewport--${mode}`}>
      <div className={`action-canvas-nodes action-canvas-nodes--${mode}`} style={mode === "freeform" ? { width, height } : undefined}>
        {mode === "freeform" ? <svg className="action-canvas-links" width={width} height={height} aria-hidden="true">
          {positions.slice(1).map((point, index) => <path key={`${program.steps[index]!.id}:${program.steps[index + 1]!.id}`} d={actionSequenceCurve(positions[index]!, point)} />)}
        </svg> : null}
        {program.steps.map((step, index) => <button type="button" key={step.id} className="action-canvas-node" aria-pressed={selected?.id === step.id}
          disabled={editingBlocked} style={mode === "freeform" ? { left: positions[index]!.x, top: positions[index]!.y, width: ACTION_NODE_WIDTH, minHeight: ACTION_NODE_HEIGHT } : undefined}
          onClick={() => setSelectedId(step.id)}
          onKeyDown={event => {
            if (mode !== "freeform" || !event.altKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
            event.preventDefault(); const point = positions[index]!, n = event.shiftKey ? 40 : 10;
            change(moveActionNode(program, step.id, point.x + (event.key === "ArrowRight" ? n : event.key === "ArrowLeft" ? -n : 0), point.y + (event.key === "ArrowDown" ? n : event.key === "ArrowUp" ? -n : 0)));
          }}
          onPointerDown={event => {
            if (mode !== "freeform" || editingBlocked || event.button !== 0) return;
            setSelectedId(step.id); event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = { id: step.id, pointer: event.pointerId, startX: event.clientX, startY: event.clientY, ...positions[index]! };
          }}
          onPointerMove={event => {
            const current = drag.current; if (!current || current.id !== step.id || current.pointer !== event.pointerId) return;
            change(moveActionNode(program, step.id, current.x + event.clientX - current.startX, current.y + event.clientY - current.startY));
          }}
          onPointerUp={event => { if (drag.current?.pointer === event.pointerId) { drag.current = undefined; event.currentTarget.releasePointerCapture(event.pointerId); } }}
          onLostPointerCapture={() => { drag.current = undefined; }}>
          <strong>{index + 1}. {step.kind === "wait" ? "Wait" : step.toolId}</strong>
          <span>{step.kind === "wait" ? `${step.delayMs} ms` : step.action}</span>
          <small>{step.when ? "Conditional · " : ""}{step.id}</small>
        </button>)}
      </div>
    </div>
    {error ? <p role="alert" className="automation-error">{error}</p> : null}
    {selected ? <fieldset><legend>Step {selectedIndex + 1} · {selected.id}</legend>
      <div className="automation-buttons"><button type="button" disabled={editingBlocked || selectedIndex === 0} onClick={() => safely(() => change(reorderActionStep(program, selected.id, -1)))}>Move earlier</button><button type="button" disabled={editingBlocked || selectedIndex === program.steps.length - 1} onClick={() => safely(() => change(reorderActionStep(program, selected.id, 1)))}>Move later</button><button type="button" disabled={editingBlocked || program.steps.length <= 1} onClick={() => safely(() => change(removeActionNode(program, selected.id)))}>Remove step</button></div>
      {selected.kind === "tool" ? <><label>Tool ID<input disabled={editingBlocked} value={selected.toolId} onChange={event => updateStep({ ...selected, toolId: event.target.value })} /></label><label>Action<input disabled={editingBlocked} value={selected.action} onChange={event => updateStep({ ...selected, action: event.target.value })} /></label>
        <label>Arguments (JSON object)<textarea key={selected.id} rows={4} disabled={disabled} defaultValue={JSON.stringify(selected.arguments, null, 2)} onChange={event => {
          try {
            const value: unknown = JSON.parse(event.target.value); if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Arguments must be a JSON object.");
            setValidArguments(true); onValidityChange(true); updateStep({ ...selected, arguments: value as Extract<AutomationStep, { kind: "tool" }>["arguments"] });
          } catch (reason) { setValidArguments(false); onValidityChange(false); setError(reason instanceof Error ? reason.message : String(reason)); }
        }} /></label></> : <label>Wait (milliseconds)<input disabled={disabled} type="number" min="0" step="1" value={selected.delayMs} onChange={event => safely(() => { const delayMs = Number(event.target.value); if (!Number.isSafeInteger(delayMs) || delayMs < 0) throw new Error("Use a nonnegative whole millisecond duration."); updateStep({ ...selected, delayMs }); })} /></label>}
      {mode === "freeform" ? <div className="action-canvas-coordinates">{(["x", "y"] as const).map(axis => <label key={axis}>{axis.toUpperCase()}<input disabled={editingBlocked} type="number" min="0" value={positions[selectedIndex]![axis]} onChange={event => safely(() => { const point = positions[selectedIndex]!; change(moveActionNode(program, selected.id, axis === "x" ? Number(event.target.value) : point.x, axis === "y" ? Number(event.target.value) : point.y)); })} /></label>)}<small>Drag a node, use coordinates, or Alt + arrow keys. Shift moves farther.</small></div> : null}
      {selected.when ? <details><summary>Condition · edit in JSON</summary><pre>{JSON.stringify(selected.when, null, 2)}</pre></details> : null}
    </fieldset> : null}
  </section>;
}
