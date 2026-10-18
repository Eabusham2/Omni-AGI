import { useEffect, useMemo, useRef, useState } from "react";
import type { MapDisplayResult } from "../../shared/maps";
import { checkedMapGeoJson } from "../../shared/mapGeoJson";
import { safeMapAttributionUrl, transformedMapView } from "./mapPresentation";
import "./maps.css";

type Command = { kind: "point" | "move" | "line"; coordinate: number[] } | { kind: "begin" | "stroke" | "close" };
function* commands(geoJson: MapDisplayResult["geoJson"]): Generator<Command> {
  const path = function* (points: unknown, closed = false): Generator<Command> {
    if (!Array.isArray(points) || !points.length) return; yield { kind: "begin" };
    for (let index = 0; index < points.length; index++) yield { kind: index ? "line" : "move", coordinate: points[index] as number[] };
    if (closed) yield { kind: "close" }; yield { kind: "stroke" };
  };
  for (const feature of geoJson.features) {
    const pending = [feature.geometry];
    while (pending.length) { const shape = pending.pop(); if (!shape) continue;
      const coords = shape.coordinates as unknown[];
      if (shape.type === "GeometryCollection") { const children = shape.geometries as Record<string, unknown>[]; for (let i = children.length - 1; i >= 0; i--) pending.push(children[i]!); }
      else if (shape.type === "Point") yield { kind: "point", coordinate: coords as number[] };
      else if (shape.type === "MultiPoint") for (const coordinate of coords) yield { kind: "point", coordinate: coordinate as number[] };
      else if (shape.type === "LineString") yield* path(coords);
      else if (shape.type === "MultiLineString") for (const line of coords) yield* path(line);
      else if (shape.type === "Polygon") for (const ring of coords) yield* path(ring, true);
      else if (shape.type === "MultiPolygon") for (const polygon of coords) for (const ring of polygon as unknown[]) yield* path(ring, true);
    }
  }
}
const themeListeners = new Set<() => void>(); let themeObserver: MutationObserver | undefined;
function themeChanges(listener: () => void): () => void {
  themeListeners.add(listener);
  if (!themeObserver) { themeObserver = new MutationObserver(() => { for (const changed of themeListeners) changed(); }); themeObserver.observe(document.documentElement, { attributes: true }); }
  return () => { themeListeners.delete(listener); if (!themeListeners.size) { themeObserver?.disconnect(); themeObserver = undefined; } };
}
function fit(bounds: MapDisplayResult["bounds"]): [number, number, number, number] {
  if (!bounds) return [-180, -90, 180, 90];
  const x = (bounds[0] + bounds[2]) / 2, y = (bounds[1] + bounds[3]) / 2;
  const width = Math.max(.01, bounds[2] - bounds[0]) * 1.2, height = Math.max(.01, bounds[3] - bounds[1]) * 1.2;
  return [x - width / 2, y - height / 2, x + width / 2, y + height / 2];
}

/** Mathematical coordinate canvas; no decorative geography or tile network. */
export function MapResultView({ result }: { result: MapDisplayResult }) {
  const data = useMemo(() => { try { return { ...result, ...checkedMapGeoJson(result.geoJson) }; } catch { return undefined; } }, [result]);
  const [view, setView] = useState<[number, number, number, number]>(() => fit(data?.bounds ?? null)), [drawing, setDrawing] = useState(true), [renderError, setRenderError] = useState("");
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => { setView(fit(data?.bounds ?? null)); }, [data?.geoJson]);
  useEffect(() => {
    const element = canvas.current; if (!element || !data) return; let active = true, frame = 0, visible = typeof IntersectionObserver === "undefined";
    const draw = (): void => {
      cancelAnimationFrame(frame); if (!active || !visible) return; const context = element.getContext("2d");
      if (!context) { setRenderError("Canvas drawing is unavailable; the supplied map data remains retained."); setDrawing(false); return; }
      setRenderError("");
      const ratio = window.devicePixelRatio || 1, width = Math.max(1, element.clientWidth), height = Math.max(1, element.clientHeight);
      element.width = Math.round(width * ratio); element.height = Math.round(height * ratio); context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
      const color = getComputedStyle(element).color, project = (point: number[]): [number, number] => [(point[0]! - view[0]) / (view[2] - view[0]) * width, height - (point[1]! - view[1]) / (view[3] - view[1]) * height];
      context.strokeStyle = color; context.fillStyle = color; context.globalAlpha = .16; context.lineWidth = 1;
      for (let lon = -180; lon <= 180; lon += 30) { const [x] = project([lon, 0]); context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
      for (let lat = -90; lat <= 90; lat += 30) { const [, y] = project([0, lat]); context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
      context.globalAlpha = 1; context.lineWidth = 1.5;
      const work = commands(data.geoJson); setDrawing(true);
      const tick = (): void => {
        if (!active || !visible) return; const started = performance.now();
        do { const next = work.next(); if (next.done) { setDrawing(false); return; } const command = next.value;
          if ("coordinate" in command) { const [x, y] = project(command.coordinate);
            if (command.kind === "point") { context.beginPath(); context.arc(x, y, 4, 0, 2 * Math.PI); context.fill(); }
            else if (command.kind === "move") context.moveTo(x, y); else context.lineTo(x, y);
          } else if (command.kind === "begin") context.beginPath(); else if (command.kind === "close") context.closePath(); else context.stroke();
        } while (performance.now() - started < 5);
        frame = requestAnimationFrame(tick);
      }; frame = requestAnimationFrame(tick);
    };
    const resized = new ResizeObserver(draw); resized.observe(element); const stopTheme = themeChanges(draw);
    const visibility = typeof IntersectionObserver === "undefined" ? undefined : new IntersectionObserver(entries => {
      visible = entries.some(entry => entry.target === element && entry.isIntersecting);
      if (visible) draw(); else { cancelAnimationFrame(frame); element.width = 1; element.height = 1; }
    }); visibility?.observe(element); draw();
    return () => { active = false; cancelAnimationFrame(frame); resized.disconnect(); visibility?.disconnect(); stopTheme(); };
  }, [data?.geoJson, view]);
  if (!data) return <p role="alert">The supplied map geometry could not be displayed.</p>;
  const zoom = (factor: number, dx = 0, dy = 0): void => setView(current => transformedMapView(current, factor, dx, dy));
  const url = safeMapAttributionUrl(data.attributionUrl);
  return <section className="map-result-view" aria-label="Offline coordinate display"><strong>Offline coordinate canvas</strong>
    <p>{data.geoJson.features.length.toLocaleString()} supplied {data.geoJson.features.length === 1 ? "feature" : "features"} · WGS84 longitude/latitude. No street basemap was downloaded.</p>
    <canvas ref={canvas} role="img" aria-label="Supplied WGS84 point, line and polygon coordinates on a mathematical grid">Supplied geographic coordinates. No street map imagery.</canvas>
    <div className="map-result-view__controls"><button type="button" className="button button--ghost" onClick={() => zoom(.5)}>Zoom in</button><button type="button" className="button button--ghost" onClick={() => zoom(2)}>Zoom out</button>
      <button type="button" className="button button--ghost" onClick={() => setView(fit(data.bounds))}>Fit data</button><button type="button" className="button button--ghost" onClick={() => setView([-180, -90, 180, 90])}>World grid</button>
      <button type="button" className="button button--ghost" aria-label="Pan west" onClick={() => zoom(1, -.25)}>←</button><button type="button" className="button button--ghost" aria-label="Pan east" onClick={() => zoom(1, .25)}>→</button>
      <button type="button" className="button button--ghost" aria-label="Pan north" onClick={() => zoom(1, 0, .25)}>↑</button><button type="button" className="button button--ghost" aria-label="Pan south" onClick={() => zoom(1, 0, -.25)}>↓</button></div>
    {renderError ? <p role="alert">{renderError}</p> : <p aria-live="polite">{drawing ? "Rendering supplied geometry…" : "Geometry rendered."} View: {view.map(value => value.toLocaleString(undefined, { maximumFractionDigits: 6 })).join(", ")}.</p>}
    <p className="map-result-view__attribution">{url ? <a href={url} target="_blank" rel="noopener noreferrer">{data.attribution}</a> : data.attribution}</p>
  </section>;
}
