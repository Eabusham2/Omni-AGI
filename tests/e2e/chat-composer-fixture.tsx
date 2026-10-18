import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ComposerSubmitControl } from "../../src/renderer/src/ComposerSubmitControl";
import { ChatQueueStack, type QueueStackItem } from "../../src/renderer/src/ChatQueueStack";
import "../../src/renderer/src/styles.css";

type FixtureMode = "idle" | "busy" | "finished";
interface FixtureLog { id: number; text: string }

const INITIAL_DRAFT = "This unsent draft must remain in the composer while fixture controls and queued messages change.";
const INITIAL_QUEUE: QueueStackItem[] = [
  { id: "queued-research", text: "Compare the two long research notes, preserve every source distinction, and do not collapse uncertain claims into a confident summary." },
  { id: "queued-code", text: "After the current reply, inspect the implementation details and explain the smallest safe code change without pretending any command was executed." },
  { id: "queued-follow-up", text: "Keep this final follow-up in the visible ordered queue so editing, removing, moving, Send now, and Steer controls can be inspected directly." }
];

function moveItem(items: QueueStackItem[], id: string, direction: -1 | 1): QueueStackItem[] {
  const index = items.findIndex(item => item.id === id), target = index + direction;
  if (index < 0 || target < 0 || target >= items.length) return items;
  const next = [...items], [selected] = next.splice(index, 1); next.splice(target, 0, selected!); return next;
}

function Fixture() {
  const [mode, setMode] = useState<FixtureMode>("idle");
  const [draft, setDraft] = useState(INITIAL_DRAFT);
  const [queued, setQueued] = useState<QueueStackItem[]>(INITIAL_QUEUE);
  const [stopping, setStopping] = useState(false);
  const [logs, setLogs] = useState<FixtureLog[]>([
    { id: 1, text: "Fixture ready · synthetic callbacks only · neural work 0 · host submissions 0" }
  ]);
  const active = mode === "busy";
  const log = (text: string): void => setLogs(current => [...current.slice(-7), { id: (current.at(-1)?.id ?? 0) + 1, text }]);
  const chooseMode = (next: FixtureMode): void => {
    setMode(next); setStopping(false);
    log(`Synthetic state changed to ${next} · draft preserved · no chat or neural work`);
  };
  const sendQueued = (id: string): void => {
    log(`${active ? "Synthetic Steer" : "Synthetic Send now"} callback · ${id} · queue retained · host submissions 0`);
  };

  return <main data-fixture="chat-composer-queue" data-fixture-state={mode}
    style={{ boxSizing: "border-box", height: "100dvh", overflow: "auto", maxWidth: 1040, margin: "0 auto", padding: 24 }}>
    <header>
      <p className="eyebrow">No-engine UI fixture</p>
      <h1>Production composer and queue controls</h1>
      <p><strong>Synthetic callbacks only.</strong> No brain, model, neural work, provider, file access, IPC, chat submission, or persistent user data.</p>
    </header>

    <section aria-label="Synthetic state controls" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBlock: 16 }}>
      <button className="button" type="button" aria-pressed={mode === "idle"} onClick={() => chooseMode("idle")}>Idle</button>
      <button className="button" type="button" aria-pressed={mode === "busy"} onClick={() => chooseMode("busy")}>Busy · responding</button>
      <button className="button" type="button" aria-pressed={mode === "finished"} onClick={() => chooseMode("finished")}>Finished</button>
      <button className="button" type="button" onClick={() => {
        document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark";
      }}>Switch appearance</button>
      <button className="button" type="button" onClick={() => { setQueued(INITIAL_QUEUE); log("Synthetic queue reset · draft preserved"); }}>Reset queue</button>
    </section>

    <section aria-labelledby="fixture-state-heading" className="panel" style={{ padding: 16, marginBottom: 16 }}>
      <h2 id="fixture-state-heading">State · {mode}</h2>
      <p>{active ? "The production submit control reads Queue; hold Ctrl or Command to expose Steer." :
        mode === "finished" ? "The synthetic reply is finished; the production submit control reads Send." :
          "The fixture is idle; the production submit control reads Send."}</p>
      <ChatQueueStack items={queued} busy={active}
        onEdit={(id, text) => { setQueued(current => current.map(item => item.id === id ? { ...item, text } : item));
          log(`Synthetic Edit callback · ${id} · host submissions 0`); }}
        onRemove={id => { setQueued(current => current.filter(item => item.id !== id));
          log(`Synthetic Remove callback · ${id} · host submissions 0`); }}
        onMove={(id, direction) => { setQueued(current => moveItem(current, id, direction));
          log(`Synthetic Reorder callback · ${id} · ${direction < 0 ? "earlier" : "later"} · host submissions 0`); }}
        onSendNow={sendQueued} />
    </section>

    <section className="composer" aria-label="Synthetic composer">
      <label htmlFor="fixture-draft"><strong>Unsent draft</strong></label>
      <textarea id="fixture-draft" value={draft} onChange={event => setDraft(event.currentTarget.value)}
        rows={5} style={{ width: "100%", boxSizing: "border-box" }} />
      <p className="muted">Changing fixture state or invoking a synthetic callback does not clear this draft.</p>
      <div className="composer__bottom">
        <span className="composer__input-count" aria-hidden="true" />
        <ComposerSubmitControl active={active} stopping={stopping} canSend={Boolean(draft.trim())} canSteer
          onSend={() => log(`${active ? "Synthetic Queue" : "Synthetic Send"} callback · draft preserved · host submissions 0`)}
          onSteer={() => log("Synthetic Steer callback · modifier observed · draft preserved · host submissions 0")}
          onStop={() => { setStopping(true); log("Synthetic Stop callback · no active neural work was cancelled"); }} />
      </div>
    </section>

    <section aria-labelledby="fixture-log-heading" style={{ marginTop: 18 }}>
      <h2 id="fixture-log-heading">Fixture callback log</h2>
      <ol aria-live="polite">{logs.map(entry => <li key={entry.id}>{entry.text}</li>)}</ol>
      <p><strong>Neural work:</strong> 0 · <strong>Host submissions:</strong> 0 · <strong>Provider calls:</strong> 0</p>
    </section>
  </main>;
}

document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
