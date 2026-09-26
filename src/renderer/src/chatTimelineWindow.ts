export const CHAT_TIMELINE_PAGE_SIZE = 40;
export const CHAT_TIMELINE_MAX_RENDERED = 120;

export interface ChatTimelineWindow {
  start: number;
  end: number;
  total: number;
}

function boundedTotal(value: number): number {
  return Number.isSafeInteger(value) && value > 0 ? value : 0;
}

export function latestChatTimelineWindow(
  totalValue: number,
  maximum = CHAT_TIMELINE_MAX_RENDERED
): ChatTimelineWindow {
  const total = boundedTotal(totalValue);
  const size = Math.max(1, Math.floor(maximum));
  return { start: Math.max(0, total - size), end: total, total };
}

/** Reconcile count changes without moving an older window to new output. */
export function reconcileChatTimelineWindow(
  current: ChatTimelineWindow,
  totalValue: number,
  followLatest: boolean,
  maximum = CHAT_TIMELINE_MAX_RENDERED
): ChatTimelineWindow {
  const total = boundedTotal(totalValue);
  if (followLatest) return latestChatTimelineWindow(total, maximum);
  const size = Math.max(0, Math.min(maximum, current.end - current.start));
  const end = Math.max(0, Math.min(total, current.end));
  const start = Math.max(0, Math.min(end, current.start, end - size));
  return { start, end, total };
}

export function olderChatTimelineWindow(
  current: ChatTimelineWindow,
  pageSize = CHAT_TIMELINE_PAGE_SIZE,
  maximum = CHAT_TIMELINE_MAX_RENDERED
): ChatTimelineWindow {
  if (current.start <= 0) return current;
  const page = Math.max(1, Math.floor(pageSize));
  const limit = Math.max(page, Math.floor(maximum));
  const move = Math.min(page, current.start);
  const currentSize = current.end - current.start;
  const expansion = Math.min(move, Math.max(0, limit - currentSize));
  const shift = move - expansion;
  return {
    start: current.start - move,
    end: current.end - shift,
    total: current.total
  };
}

export function newerChatTimelineWindow(
  current: ChatTimelineWindow,
  pageSize = CHAT_TIMELINE_PAGE_SIZE,
  maximum = CHAT_TIMELINE_MAX_RENDERED
): ChatTimelineWindow {
  if (current.end >= current.total) return current;
  const page = Math.max(1, Math.floor(pageSize));
  const limit = Math.max(page, Math.floor(maximum));
  const move = Math.min(page, current.total - current.end);
  const currentSize = current.end - current.start;
  const expansion = Math.min(move, Math.max(0, limit - currentSize));
  const shift = move - expansion;
  return {
    start: current.start + shift,
    end: current.end + move,
    total: current.total
  };
}

/** Preserve the exact viewport offset of a surviving variable-height row. */
export function anchoredTimelineScrollTop(
  scrollTop: number,
  anchorOffsetBefore: number,
  anchorOffsetAfter: number
): number {
  if (![scrollTop, anchorOffsetBefore, anchorOffsetAfter].every(Number.isFinite)) {
    return Math.max(0, Number.isFinite(scrollTop) ? scrollTop : 0);
  }
  return Math.max(0, scrollTop + anchorOffsetAfter - anchorOffsetBefore);
}
