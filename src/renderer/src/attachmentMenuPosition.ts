export function attachmentMenuPosition(trigger: { left: number; top: number; bottom: number },
  menu: { width: number; height: number }, viewport: { width: number; height: number }) {
  const gap = 8;
  const above = Math.max(0, trigger.top - gap), below = Math.max(0, viewport.height - trigger.bottom - gap);
  const opensAbove = above >= menu.height || above >= below;
  const maxHeight = Math.max(0, Math.min(viewport.height * 0.6, 384, opensAbove ? above : below));
  const height = Math.min(menu.height, maxHeight);
  return {
    left: Math.max(gap, Math.min(trigger.left, viewport.width - menu.width - gap)),
    top: opensAbove ? Math.max(gap, trigger.top - gap - height) : trigger.bottom + gap,
    maxHeight
  };
}
