import { useEffect, useRef, useState } from "react";
import type { ScheduleRecord } from "../../shared/clock";
import { AutomationWorkspace } from "./AutomationWorkspace";
import { ClockWorkspacePanel } from "./ClockWorkspacePanel";

export function TimeWorkspace({ brainId, onToast, onActiveMode }: { brainId: string; onToast?(message: string): void; onActiveMode?(): void }) {
  const [editing, setEditing] = useState<ScheduleRecord>();
  const [open, setOpen] = useState(false);
  const section = useRef<HTMLDetailsElement>(null);
  useEffect(() => { setEditing(undefined); setOpen(false); }, [brainId]);
  useEffect(() => { if (editing) section.current?.scrollIntoView({ block: "start", behavior: "auto" }); }, [editing]);
  return <div className="time-workspace">
    <ClockWorkspacePanel key={brainId} brainId={brainId} onToast={onToast} onActiveMode={onActiveMode}
      onEditAutomationSchedule={schedule => { setEditing(schedule); setOpen(true); }} />
    <details ref={section} className="time-workspace__actions" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
      <summary>Reusable action files & independent automation</summary>
      {open ? <AutomationWorkspace key={brainId} brainId={brainId} scheduleToEdit={editing} onScheduleEditFinished={() => setEditing(undefined)} onToast={onToast} /> : null}
    </details>
  </div>;
}
