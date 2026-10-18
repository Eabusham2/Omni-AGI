import { useEffect, useMemo, useRef, useState } from "react";
import { mercatorY, type MapRasterImage, type MapRasterResult } from "../../shared/mapBasemaps";
import { checkedMapRasterResult } from "./mapRasterPresentation";
import "./maps.css";

async function decodedImage(image: MapRasterImage): Promise<ImageBitmap> {
  const encoded = image.dataUrl.slice(image.dataUrl.indexOf(",") + 1), binary = atob(encoded), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  if (bytes.byteLength !== image.bytes) throw new Error("Retained map image byte count changed.");
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), n => n.toString(16).padStart(2, "0")).join("");
  if (digest !== image.sha256) throw new Error("Retained map image failed its exact checksum.");
  const bitmap = await createImageBitmap(new Blob([bytes], { type: image.mime }));
  if (bitmap.width !== image.width || bitmap.height !== image.height) { bitmap.close(); throw new Error("Retained map image dimensions changed."); } return bitmap;
}

/** Local raster bytes only: rendering never causes provider requests or credential use. */
export function MapRasterView({ result }: { result: MapRasterResult }) {
  const data = useMemo(() => checkedMapRasterResult(result), [result]), canvas = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState("Waiting to render retained map…"), [error, setError] = useState("");
  useEffect(() => {
    const element = canvas.current; if (!element || !data) return;
    let active = true, visible = typeof IntersectionObserver === "undefined", generation = 0, rendering = false, requested = false;
    const render = async (run: number): Promise<void> => {
      if (!active || !visible) return;
      const context = element.getContext("2d"); if (!context) { setError("Raster drawing is unavailable; the exact map bytes remain retained."); return; }
      const width = Math.max(1, Math.round(element.clientWidth * (window.devicePixelRatio || 1))), height = Math.max(1, Math.round(element.clientHeight * (window.devicePixelRatio || 1)));
      element.width = width; element.height = height; context.clearRect(0, 0, width, height); setStatus("Rendering retained map…"); setError("");
      const north = mercatorY(data.bounds[3]), south = mercatorY(data.bounds[1]);
      try {
        for (const image of data.images) {
          if (!active || !visible || run !== generation) return;
          const bitmap = await decodedImage(image);
          try {
            if (!active || !visible || run !== generation) return;
            if (data.provider === "mapbox-static-image") {
              const scale = Math.min(width / image.width, height / image.height), w = image.width * scale, h = image.height * scale;
              context.drawImage(bitmap, (width - w) / 2, (height - h) / 2, w, h);
            } else {
              const x = (image.bounds[0] - data.bounds[0]) / (data.bounds[2] - data.bounds[0]) * width, y = (mercatorY(image.bounds[3]) - north) / (south - north) * height;
              const w = (image.bounds[2] - image.bounds[0]) / (data.bounds[2] - data.bounds[0]) * width, h = (mercatorY(image.bounds[1]) - mercatorY(image.bounds[3])) / (south - north) * height;
              context.drawImage(bitmap, x, y, w, h);
            }
          } finally { bitmap.close(); }
        }
        if (active && visible && run === generation) setStatus("Retained map rendered. No network request was made.");
      } catch (failure) { if (active && run === generation) { context.clearRect(0, 0, width, height); setError(failure instanceof Error ? failure.message : String(failure)); } }
    };
    // Coalesce resize/visibility changes: never run overlapping raster decoders.
    const draw = (): void => {
      generation++; requested = true; if (rendering) return; rendering = true;
      void (async () => { try { while (active && visible && requested) { requested = false; await render(generation); } }
        finally { rendering = false; } })().catch(failure => { if (active) setError(failure instanceof Error ? failure.message : String(failure)); });
    };
    const resized = new ResizeObserver(draw); resized.observe(element);
    const visibility = typeof IntersectionObserver === "undefined" ? undefined : new IntersectionObserver(entries => {
      visible = entries.some(entry => entry.target === element && entry.isIntersecting);
      if (visible) draw(); else { generation++; element.width = 1; element.height = 1; }
    }); visibility?.observe(element); draw();
    return () => { active = false; generation++; resized.disconnect(); visibility?.disconnect(); element.width = 1; element.height = 1; };
  }, [data]);
  if (!data) return <p role="alert">The retained map could not be safely displayed.</p>;
  const attributionLinks = [...new Map([...(data.provider === "mapbox-static-image" ? [
    { label: "© Mapbox", url: "https://www.mapbox.com/about/maps/" }, { label: "© OpenStreetMap contributors", url: "https://www.openstreetmap.org/copyright" }
  ] : []), ...data.attributionLinks].map(link => [link.url, link])).values()];
  return <section className="map-result-view map-raster-view" aria-label="Retained basemap"><strong>{data.label}</strong>
    <canvas ref={canvas} role="img" aria-label={`Retained map from ${data.label}`}>Retained raster map; no remote image is loaded.</canvas>
    {error ? <p role="alert">{error}</p> : <p role="status">{status}</p>}
    <p>Requested area: {data.bounds.map(n => n.toLocaleString(undefined, { maximumFractionDigits: 6 })).join(", ")}{data.provider === "mapbox-static-image" ? ". The provider fits this area to the image; its visible edges may extend beyond these bounds." : "."}</p>
    <p className="map-result-view__attribution">{attributionLinks.map((link, index) => <span key={`${link.url}:${index}`}>{index ? " · " : ""}<a href={link.url} target="_blank" rel="noopener noreferrer">{link.label}</a></span>)}</p>
    <details><summary>Map source</summary><p>{data.provenance.source}</p><p>Source received {data.provenance.observedAt}. {data.images.length.toLocaleString()} retained {data.images.length === 1 ? "image" : "images"}.</p>
      <p>Authorization recorded by the user: {data.rights.authorizationReference}. <a href={data.rights.licenseUrl} target="_blank" rel="noopener noreferrer">Source license</a>. Omni does not determine that this declaration grants legal rights.</p></details>
  </section>;
}
