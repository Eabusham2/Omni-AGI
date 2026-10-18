import type { ActionEvent, ModalityPreview } from "../../shared/types";

function sameActionRevision(left: ActionEvent, right: ActionEvent): boolean {
  return (
    left.id === right.id &&
    left.updatedAt === right.updatedAt &&
    left.state === right.state &&
    left.progress === right.progress &&
    left.statusLabel === right.statusLabel &&
    left.runtimeJobId === right.runtimeJobId &&
    left.cancellationRequested === right.cancellationRequested &&
    left.inlineGenerationOwned === right.inlineGenerationOwned &&
    left.error === right.error &&
    left.evolutionRunId === right.evolutionRunId &&
    left.execution?.id === right.execution?.id &&
    left.execution?.state === right.execution?.state &&
    left.execution?.approvalToken === right.execution?.approvalToken &&
    left.execution?.finishedAt === right.execution?.finishedAt &&
    left.execution?.error === right.execution?.error &&
    left.execution?.output === right.execution?.output &&
    left.preview?.revision === right.preview?.revision
  );
}

/**
 * Merge a lightweight action envelope without erasing a media preview that
 * arrived through the dedicated preview-patch event. Returning the original
 * array for a duplicate revision prevents a redundant React render.
 */
export function mergeChatActionEvent(
  current: ActionEvent[],
  incoming: ActionEvent
): ActionEvent[] {
  const index = current.findIndex((candidate) => candidate.id === incoming.id);
  if (index < 0) return [...current, incoming];
  const existing = current[index]!;
  const merged: ActionEvent = {
    ...existing,
    ...incoming,
    preview: incoming.preview ?? existing.preview,
    execution:
      incoming.execution === undefined
        ? existing.execution
        : existing.execution === undefined
          ? incoming.execution
          : {
              ...existing.execution,
              ...incoming.execution,
              output: incoming.execution.output ?? existing.execution.output
            }
  };
  if (sameActionRevision(existing, merged)) return current;
  const next = [...current];
  next[index] = merged;
  return next;
}

function definedPreviewFields(preview: ModalityPreview): Partial<ModalityPreview> {
  return Object.fromEntries(
    Object.entries(preview).filter(([, value]) => value !== undefined)
  ) as Partial<ModalityPreview>;
}

/** Apply one ordered media patch; stale/duplicate revisions are no-ops. */
export function patchChatActionPreview(
  current: ActionEvent[],
  actionId: string,
  patch: ModalityPreview
): ActionEvent[] {
  const index = current.findIndex((candidate) => candidate.id === actionId);
  if (index < 0) return current;
  const existing = current[index]!;
  if ((existing.preview?.revision ?? -1) >= patch.revision) return current;
  const preview: ModalityPreview = {
    ...(existing.preview ?? { revision: patch.revision }),
    ...definedPreviewFields(patch),
    revision: patch.revision
  };
  const next = [...current];
  next[index] = {
    ...existing,
    preview,
    progress: patch.progress ?? existing.progress,
    statusLabel: patch.statusLabel ?? existing.statusLabel
  };
  return next;
}

export interface TextFrameBatcher {
  push(delta: string): void;
  flush(): void;
  reset(): void;
}

/**
 * Coalesce high-frequency token notifications into one state update per
 * animation frame without pacing ordinary output. A finite explicit budget
 * is reserved for opt-in display pacing; it never throttles native generation.
 * Terminal events always force an immediate complete flush.
 */
export function createTextFrameBatcher(
  commit: (delta: string) => void,
  schedule: (callback: () => void) => number,
  cancel: (handle: number) => void,
  maxCharactersPerFrame = Number.POSITIVE_INFINITY
): TextFrameBatcher {
  if (maxCharactersPerFrame !== Number.POSITIVE_INFINITY &&
      (!Number.isFinite(maxCharactersPerFrame) || maxCharactersPerFrame <= 0)) {
    throw new Error("A display frame budget must be positive or unlimited.");
  }
  let buffer = "";
  let handle: number | null = null;
  const frameLimit = Math.max(1, Math.floor(maxCharactersPerFrame));

  const scheduleDrain = (): void => {
    if (handle !== null || !buffer) return;
    handle = schedule(() => {
      handle = null;
      if (!buffer) return;
      let end = Math.min(buffer.length, frameLimit);
      // Optional pacing must not briefly render half of a Unicode surrogate
      // pair as a replacement glyph. One scalar may exceed the display budget.
      if (end < buffer.length && end > 0 &&
          buffer.charCodeAt(end - 1) >= 0xd800 && buffer.charCodeAt(end - 1) <= 0xdbff &&
          buffer.charCodeAt(end) >= 0xdc00 && buffer.charCodeAt(end) <= 0xdfff) end++;
      const delta = buffer.slice(0, end);
      buffer = buffer.slice(end);
      commit(delta);
      scheduleDrain();
    });
  };

  const flush = (): void => {
    if (handle !== null) {
      cancel(handle);
      handle = null;
    }
    if (!buffer) return;
    const delta = buffer;
    buffer = "";
    commit(delta);
  };

  return {
    push(delta) {
      if (!delta) return;
      buffer += delta;
      scheduleDrain();
    },
    flush,
    reset() {
      if (handle !== null) cancel(handle);
      handle = null;
      buffer = "";
    }
  };
}
