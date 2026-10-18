/** Preload presence is not a neural worker, hardware or loaded-model receipt. */
export function libraryRuntimePresentation(preview: boolean): { title: string; detail: string; badge?: string } {
  return preview
    ? { title: "Interface preview", detail: "No neural engine is connected.", badge: "No trained brain" }
    : { title: "Local workspace", detail: "Instances are loaded and checked separately." };
}
