/** Preload presence is not a neural worker, hardware or loaded-model receipt. */
export function libraryRuntimePresentation(preview: boolean): { title: string; detail: string; badge?: string } {
  return preview
    ? { title: "No brain connected", detail: "Preview only. Neural work is unavailable." }
    : { title: "Local workspace", detail: "Instances are loaded and checked separately." };
}
