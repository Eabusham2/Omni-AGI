import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { MapBasemapPanel } from "../../src/renderer/src/MapBasemapPanel";
import { MapRasterView } from "../../src/renderer/src/MapRasterView";
import type { MapBasemapProfile, MapBasemapsApi, MapRasterResult } from "../../src/shared/mapBasemaps";
import "../../src/renderer/src/styles.css";

function Fixture() {
  const [dark, setDark] = useState(false), [compact, setCompact] = useState(false), [owner, setOwner] = useState(1);
  const [receipts, setReceipts] = useState<string[]>([]), [held, setHeld] = useState(false), [image, setImage] = useState<MapRasterResult>();
  const hold = useRef(false), release = useRef<(() => void) | undefined>(undefined), entries = useRef<MapBasemapProfile[]>([]);
  const log = (value: string) => setReceipts(rows => [...rows, value]);
  const api = useMemo<MapBasemapsApi>(() => ({
    list: async () => { log(`Synthetic list · owner ${owner}`); if (hold.current) { hold.current = false; setHeld(true);
      return new Promise(resolve => { release.current = () => resolve([{ id: "retired-source", revision: 1, label: "Retired owner — must not appear" } as MapBasemapProfile]); }); }
      return structuredClone(entries.current); },
    save: async value => { log(`Synthetic save · ${value.id} · enabled ${value.enabled}`);
      const { expectedRevision, credential: _credential, ...body } = value;
      const saved: MapBasemapProfile = { ...body, revision: expectedRevision + 1, credentialConfigured: false, credentialPersistence: "session" };
      entries.current = [...entries.current.filter(row => row.id !== saved.id), saved]; return structuredClone(saved); },
    remove: async id => { log(`Synthetic remove · ${id}`); entries.current = entries.current.filter(row => row.id !== id); }
  }), [owner]);
  const publish = async (corrupt: boolean) => {
    // One real PNG pixel, not geographic imagery or a provider/decoder-quality claim.
    const encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGP4DwQACfsD/fteaysAAAAASUVORK5CYII=";
    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), n => n.toString(16).padStart(2, "0")).join("");
    const now = new Date().toISOString();
    setImage({ format: "omni-map-raster", version: 1, provider: "mapbox-static-image", profileId: "synthetic", profileRevision: 1,
      label: "One-pixel retained-image fixture — not geography", bounds: [-1, -1, 1, 1], width: 1, height: 1, coordinateReferenceSystem: "EPSG:3857",
      images: [{ dataUrl: `data:image/png;base64,${encoded}`, sha256: corrupt ? "0".repeat(64) : sha256, bytes: bytes.length,
        mime: "image/png", width: 1, height: 1, bounds: [-1, -1, 1, 1], extent: "provider-fit-to-request", cached: false, sourceObservedAt: now }],
      attribution: "Synthetic local fixture", attributionUrl: "https://example.invalid/fixture", attributionLinks: [{ label: "Synthetic local fixture", url: "https://example.invalid/fixture" }],
      rights: { retentionLearningAndRedistributionAuthorized: true, authorizationReference: "Owned synthetic UI fixture", licenseUrl: "https://example.invalid/fixture" },
      provenance: { source: "Synthetic local bytes; no provider contacted", observedAt: now, requestSha256: "a".repeat(64), brandingRequested: true, automaticRetry: false },
      basemapDownloaded: true, untrustedTaskData: true });
  };
  return <main style={{ maxWidth: compact ? 360 : 940, margin: "0 auto", padding: 16, minWidth: 0, boxSizing: "border-box" }}>
    <h1>Optional map controls</h1><p>Synthetic UI only. No provider, credential, permission, model or filesystem operation.</p>
    <div className="map-connections__actions">
      <button className="button" onClick={() => { setDark(!dark); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(!compact)}>Compact content</button>
      <button className="button" onClick={() => { hold.current = true; }}>Hold next list</button>
      <button className="button" onClick={() => setOwner(value => value + 1)}>Replace API owner</button>
      <button className="button" disabled={!held} onClick={() => { release.current?.(); release.current = undefined; setHeld(false); }}>Deliver old list</button>
      <button className="button" onClick={() => void publish(false)}>Show retained pixel</button>
      <button className="button" onClick={() => void publish(true)}>Show checksum error</button>
    </div>
    <section className="surface integration-surface"><MapBasemapPanel api={api} /></section>
    {image ? <MapRasterView result={image} /> : null}
    <h2>Synthetic receipts</h2><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipts.join("\n") || "No synthetic operation"}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
