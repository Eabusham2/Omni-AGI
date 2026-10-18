import React, { useCallback, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { McpHostServicesPanel } from "../../src/renderer/src/McpHostServicesPanel";
import type {
  McpDiagnosticPage,
  McpDiagnosticsApi,
  McpRootCommand,
  McpRootsApi,
  McpRootsStatus
} from "../../src/shared/mcpHostServices";
import "../../src/renderer/src/styles.css";

const BRAIN_ID = "mcp-host-fixture-brain";
const SERVER_ID = "mcp-host-fixture-server";
const LONG_URI = "file:///synthetic-reviewed-folder/" +
  "very-long-folder-segment-without-any-real-filesystem-authority/".repeat(7) + "documents";

function rootsStatus(state: McpRootsStatus["state"], overrides: Partial<McpRootsStatus> = {}): McpRootsStatus {
  const roots = state === "empty" ? [] : [{
    name: "Synthetic reviewed project folder with an intentionally long display name",
    uri: LONG_URI,
    accessible: true
  }];
  return {
    format: "omni-mcp-roots-status",
    version: 1,
    brainId: BRAIN_ID,
    serverId: SERVER_ID,
    endpoint: "stdio:synthetic-no-process",
    revision: state === "empty" ? 0 : state === "review-required" ? 1 : 2,
    roots,
    state,
    connectionReady: true,
    ...(state === "shared" ? { sharedConnectionEpoch: 7, notification: "sent" as const } : {}),
    disclosureOnly: true,
    filesystemSandbox: false,
    permissionsPortable: false,
    ...overrides
  };
}

function diagnosticPage(afterSequence?: number): McpDiagnosticPage {
  const origin = { brainId: BRAIN_ID, serverId: SERVER_ID, serverLabel: "Synthetic MCP server",
    endpoint: "stdio:synthetic-no-process", connectionEpoch: 7 };
  if (afterSequence === undefined) return {
    records: [
      { format: "omni-mcp-diagnostic", version: 1, sequence: 1, observedAt: "2026-10-15T12:00:00.000Z", origin,
        source: "server", level: "notice", logger: "synthetic.transport", data: { event: "initialized", networkCalls: 0 },
        sourceSha256: "a".repeat(64), redactionApplied: true, untrustedData: true, nativeLearning: false },
      { format: "omni-mcp-diagnostic", version: 1, sequence: 2, observedAt: "2026-10-15T12:00:01.000Z", origin,
        source: "app", level: "warning", logger: "synthetic.long-payload", data: { uri: LONG_URI, note: "Untrusted synthetic diagnostic data wraps without granting access." },
        sourceSha256: "b".repeat(64), redactionApplied: true, untrustedData: true, nativeLearning: false }
    ],
    nextSequence: 2,
    hasMore: true,
    retainedBytes: 2048,
    accepted: 4,
    rejected: 2,
    rateLimited: 1,
    evicted: 3,
    retention: "bounded-current-process-only"
  };
  return {
    records: [{ format: "omni-mcp-diagnostic", version: 1, sequence: 3, observedAt: "2026-10-15T12:00:02.000Z", origin,
      source: "server", level: "info", logger: "synthetic.page-two", data: { page: 2, complete: true },
      sourceSha256: "c".repeat(64), redactionApplied: false, untrustedData: true, nativeLearning: false }],
    hasMore: false,
    retainedBytes: 2304,
    accepted: 5,
    rejected: 2,
    rateLimited: 1,
    evicted: 3,
    retention: "bounded-current-process-only"
  };
}

interface DelayedSelection {
  owner: number;
  status: McpRootsStatus;
  resolve(value: McpRootsStatus): void;
}

function Fixture() {
  const [connected, setConnected] = useState(true), [enabled, setEnabled] = useState(true);
  const [apiOwner, setApiOwner] = useState(1), [compact, setCompact] = useState(false);
  const [delayNext, setDelayNext] = useState(false), [wrongOwnerNext, setWrongOwnerNext] = useState(false);
  const [calls, setCalls] = useState({ roots: 0, diagnostics: 0, picker: 0, network: 0 });
  const [events, setEvents] = useState<string[]>([
    "NO-BRAIN fixture ready · synthetic callbacks only · picker 0 · network 0"
  ]);
  const current = useRef<McpRootsStatus>(rootsStatus("empty"));
  const delayed = useRef<DelayedSelection[]>([]);
  const delayRef = useRef(delayNext), wrongRef = useRef(wrongOwnerNext);
  delayRef.current = delayNext; wrongRef.current = wrongOwnerNext;
  const log = useCallback((message: string) => setEvents(value => [...value.slice(-11), message]), []);
  const rootCall = useCallback(() => setCalls(value => ({ ...value, roots: value.roots + 1 })), []);
  const diagnosticCall = useCallback(() => setCalls(value => ({ ...value, diagnostics: value.diagnostics + 1 })), []);

  const roots = useMemo<McpRootsApi>(() => {
    const checked = (request: McpRootCommand): void => {
      if (request.brainId !== BRAIN_ID || request.serverId !== SERVER_ID || !request.requestId) {
        throw new Error("Synthetic request lost its exact owner.");
      }
    };
    return {
      status: async (brainId, serverId) => {
        rootCall(); log(`Synthetic roots status · API owner ${apiOwner}`);
        if (brainId !== BRAIN_ID || serverId !== SERVER_ID) throw new Error("Synthetic wrong status owner.");
        return structuredClone(current.current);
      },
      select: async request => {
        checked(request); rootCall(); log(`Synthetic Choose invoked · API owner ${apiOwner} · no native picker opened`);
        const wrong = wrongRef.current;
        if (wrong) { wrongRef.current = false; setWrongOwnerNext(false); }
        const next = rootsStatus("review-required", wrong ? { brainId: "foreign-brain" } : {});
        if (delayRef.current) {
          delayRef.current = false; setDelayNext(false);
          return new Promise<McpRootsStatus>(resolve => {
            delayed.current.push({ owner: apiOwner, status: next, resolve });
            log("Synthetic selection response is held for stale-owner testing");
          });
        }
        if (!wrong) current.current = next;
        return structuredClone(next);
      },
      review: async request => {
        checked(request); rootCall(); log(`Synthetic disclosure reviewed · API owner ${apiOwner}`);
        current.current = rootsStatus("shared"); return structuredClone(current.current);
      },
      clear: async request => {
        checked(request); rootCall(); log(`Synthetic disclosure cleared · API owner ${apiOwner}`);
        current.current = rootsStatus("empty"); return structuredClone(current.current);
      }
    };
  }, [apiOwner, log, rootCall]);

  const diagnostics = useMemo<McpDiagnosticsApi>(() => ({
    page: async (brainId, serverId, afterSequence) => {
      diagnosticCall(); log(`Synthetic diagnostic page ${afterSequence === undefined ? 1 : 2} · API owner ${apiOwner}`);
      if (brainId !== BRAIN_ID || serverId !== SERVER_ID) throw new Error("Synthetic wrong diagnostic owner.");
      return diagnosticPage(afterSequence);
    }
  }), [apiOwner, diagnosticCall, log]);

  const releaseDelayed = (): void => {
    const value = delayed.current.shift();
    if (!value) { log("No synthetic response is waiting"); return; }
    if (value.owner === apiOwner && value.status.brainId === BRAIN_ID) current.current = value.status;
    value.resolve(structuredClone(value.status));
    log("Released old synthetic response; production owner fence decides whether it is visible");
  };

  return <main style={{ height: "100dvh", overflow: "auto", boxSizing: "border-box", padding: 20 }}>
    <div style={{ width: "100%", maxWidth: compact ? 360 : 980, margin: "0 auto", minWidth: 0 }}>
      <header>
        <p className="eyebrow">NO-BRAIN · NO-NETWORK UI fixture</p>
        <h1>Production MCP host-services panel</h1>
        <p><strong>Synthetic roots, review states, and current-process diagnostic pages only.</strong> No preload, folder picker, filesystem grant, MCP process, network, provider, brain, worker, model, or learning.</p>
      </header>
      <section aria-label="Synthetic fixture controls" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBlock: 16 }}>
        <button className="button" type="button" aria-pressed={connected} onClick={() => setConnected(value => !value)}>{connected ? "Connected · synthetic" : "Disconnected · synthetic"}</button>
        <button className="button" type="button" aria-pressed={enabled} onClick={() => setEnabled(value => !value)}>{enabled ? "Enabled · synthetic" : "Disabled · synthetic"}</button>
        <button className="button" type="button" onClick={() => { setApiOwner(value => value + 1); log("Synthetic API owner changed · old response must stay stale"); }}>Change API owner</button>
        <button className="button" type="button" aria-pressed={delayNext} onClick={() => setDelayNext(value => !value)}>Delay next selection</button>
        <button className="button" type="button" onClick={releaseDelayed}>Release oldest delayed response</button>
        <button className="button" type="button" aria-pressed={wrongOwnerNext} onClick={() => setWrongOwnerNext(value => !value)}>Wrong owner next</button>
        <button className="button" type="button" aria-pressed={compact} onClick={() => setCompact(value => !value)}>{compact ? "Wide content" : "Compact 360px content"}</button>
        <button className="button" type="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      </section>
      <p role="status"><strong>Observed synthetic calls:</strong> roots {calls.roots} · diagnostics {calls.diagnostics} · native picker {calls.picker} · network {calls.network}</p>
      <section className="integration-card" aria-label="MCP server fixture card">
        <div className="integration-card__head"><div><strong>Synthetic filesystem-capable MCP connection</strong><small>Exact production panel below · fake typed APIs</small></div></div>
        <McpHostServicesPanel brainId={BRAIN_ID} serverId={SERVER_ID} connected={connected} enabled={enabled} roots={roots} diagnostics={diagnostics} />
      </section>
      <section aria-labelledby="mcp-host-fixture-events" style={{ marginTop: 24 }}>
        <h2 id="mcp-host-fixture-events">Synthetic callback log</h2>
        <ol aria-live="polite">{events.map((event, index) => <li key={`${index}-${event}`}>{event}</li>)}</ol>
        <p><strong>Authority effects:</strong> 0 · <strong>filesystem selections:</strong> 0 · <strong>network requests:</strong> 0 · <strong>neural calls:</strong> 0</p>
      </section>
    </div>
  </main>;
}

document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
