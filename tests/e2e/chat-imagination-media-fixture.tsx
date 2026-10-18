import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import type { ActionEvent } from "../../src/shared/types";
import { ChatImaginationMedia } from "../../src/renderer/src/ChatImaginationMedia";
import "../../src/renderer/src/styles.css";

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#241d46"/><stop offset="1" stop-color="#176b70"/></linearGradient></defs><rect width="960" height="540" rx="34" fill="url(#g)"/><circle cx="220" cy="250" r="118" fill="#a893ff" opacity=".65"/><circle cx="700" cy="250" r="150" fill="#4dc3bc" opacity=".42"/><path d="M230 370 C420 170 550 420 740 170" fill="none" stroke="#f7f2ff" stroke-width="18" stroke-linecap="round"/><text x="480" y="475" fill="#fff" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="32">Synthetic fixture artifact</text></svg>`;
const media = `data:image/svg+xml;base64,${btoa(svg)}`;
const event: ActionEvent = {
  id: "fixture-imagination-action",
  brainId: "fixture-brain",
  action: { kind: "imagine", source: "organic", toolId: "modality.imagine", action: "generate", arguments: { modality: "image" } },
  state: "complete",
  createdAt: "2026-10-13T12:00:00.000Z",
  updatedAt: "2026-10-13T12:00:02.000Z",
  preview: { revision: 4, progress: 0.9, statusLabel: "Actual decoder preview", modality: "image", mimeType: "image/svg+xml", dataUrl: media },
  execution: { id: "fixture-execution", toolId: "modality.imagine", action: "generate", state: "complete",
    startedAt: "2026-10-13T12:00:00.100Z", finishedAt: "2026-10-13T12:00:02.000Z",
    output: { modality: "image", mimeType: "image/svg+xml", dataUrl: media, path: "/fixture/brain/artifacts/final.svg" } }
};

function Fixture() {
  const [dark, setDark] = useState(false), [narrow, setNarrow] = useState(false);
  return <main style={{ maxWidth: 860, margin: "0 auto", padding: 24 }}>
    <h1>Chat imagination media visibility</h1>
    <p>Actual React presentation with a synthetic validated event only. No model, decoder, provider, file, device, or permission runs.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button className="button" onClick={() => { const next = !dark; setDark(next); document.documentElement.dataset.colorScheme = next ? "dark" : "light"; }}>Switch theme</button>
      <button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Desktop width" : "320px surface"}</button>
    </div>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%", marginTop: 16 }}>
      <div className="chat-action-stream" aria-label="Synthetic imagination event">
        <ChatImaginationMedia event={event} />
        <details className="chat-action-card chat-action-card--complete">
          <summary><span className="chat-action-card__summary-copy"><strong>Image imagination</strong><small>Done · audit remains collapsed</small></span></summary>
          <div className="chat-action-card__body"><p>Fixture-only action audit.</p></div>
        </details>
      </div>
    </div>
  </main>;
}

document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
