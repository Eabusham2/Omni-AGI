import type { ReactNode } from "react";

export type RendererStartupMode = "desktop" | "preview" | "unavailable";

/** An explicit fixture preview has no IPC, process, file or permission authority. */
export function rendererStartupMode(bridge: unknown, search: string): RendererStartupMode {
  if (bridge !== undefined && bridge !== null) {
    try {
      const value = bridge as { window?: { platform?: unknown }; brain?: { list?: unknown } };
      return typeof value === "object" && typeof value.window?.platform === "function" &&
        typeof value.brain?.list === "function" ? "desktop" : "unavailable";
    } catch { return "unavailable"; }
  }
  const preview = new URLSearchParams(search).getAll("omni-preview");
  return preview.length === 1 && preview[0] === "1" ? "preview" : "unavailable";
}

export function RendererStartupGate({ bridge, search, onReload, children }: {
  bridge: unknown;
  search: string;
  onReload: () => void;
  children: ReactNode;
}) {
  if (rendererStartupMode(bridge, search) !== "unavailable") return <>{children}</>;
  return <main className="content-page" aria-label="Studio startup">
    <section className="surface" role="alert">
      <h1>Desktop connection unavailable</h1>
      <p>Studio’s desktop connection did not initialize. Reload this window to retry, or restart Omni AGI Studio.</p>
      <button className="button" type="button" onClick={onReload}>Reload window</button>
    </section>
  </main>;
}
