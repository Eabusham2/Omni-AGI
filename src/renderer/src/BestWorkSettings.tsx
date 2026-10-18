import React, { useRef, useState } from "react";

export interface BestWorkPreferences {
  bestWorkEnabled: boolean;
  bestWorkExplorationBranches: number;
}

export function BestWorkSettings({ preferences, onSave, applyBlockedReason }: {
  preferences: BestWorkPreferences;
  onSave: (value: BestWorkPreferences) => Promise<void>;
  applyBlockedReason?: string;
}) {
  const [enabled, setEnabled] = useState(preferences.bestWorkEnabled);
  const [branches, setBranches] = useState(String(preferences.bestWorkExplorationBranches));
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const submitting = useRef(false);
  const count = Number(branches);
  const valid = branches.trim() !== "" && Number.isSafeInteger(count) && count >= 1 && count <= 16;
  const dirty = enabled !== preferences.bestWorkEnabled || count !== preferences.bestWorkExplorationBranches;

  const save = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    if (submitting.current || !dirty || !valid || applyBlockedReason) return;
    submitting.current = true;
    setSaving(true);
    setMessage("Applying at a safe boundary…");
    try {
      await onSave({ bestWorkEnabled: enabled, bestWorkExplorationBranches: count });
      setMessage("Best Work preference saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Best Work could not be saved.");
    } finally {
      submitting.current = false;
      setSaving(false);
    }
  };

  return <section className="surface best-work-settings" aria-labelledby="best-work-heading">
    <h2 id="best-work-heading">Best Work</h2>
    <p>Allow extra candidate responses on harder turns. Uses more time and memory.</p>
    <form onSubmit={(event) => void save(event)}>
      <label className="best-work-settings__toggle">
        <input type="checkbox" checked={enabled} disabled={saving} onChange={(event) => setEnabled(event.target.checked)} />
        <span>Enable Best Work</span>
      </label>
      <details>
        <summary>Search budget</summary>
        <label htmlFor="best-work-branches">Maximum extra candidates</label>
        <input id="best-work-branches" type="number" min="1" max="16" step="1" inputMode="numeric"
          value={branches} disabled={saving} aria-invalid={!valid} aria-describedby="best-work-budget-note"
          onChange={(event) => setBranches(event.target.value)} />
        <p id="best-work-budget-note">1–16, plus the ordinary response. Actual use adapts to uncertainty and available resources.</p>
      </details>
      <button type="submit" className="button secondary" disabled={saving || !dirty || !valid || Boolean(applyBlockedReason)}>
        {saving ? "Applying…" : "Save Best Work"}
      </button>
      {applyBlockedReason && dirty ? <p role="status">{applyBlockedReason}</p> : null}
      {message ? <p role="status" aria-live="polite">{message}</p> : null}
    </form>
  </section>;
}
