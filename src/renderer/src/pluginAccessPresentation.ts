import type { PluginInstanceAccess } from "../../shared/plugins";

/** Learning status is not permission authority and must not erase draft edits. */
export function refreshedPluginAccess(current: PluginInstanceAccess | undefined, saved: PluginInstanceAccess, dirty: boolean): {
  access: PluginInstanceAccess; dirty: boolean; conflict: boolean;
} {
  if (dirty && current && current.brainId === saved.brainId && current.pluginId === saved.pluginId &&
      current.contentSha256 === saved.contentSha256 && current.revision === saved.revision)
    return { access: { ...current, semanticLearning: saved.semanticLearning }, dirty: true, conflict: false };
  return { access: saved, dirty: false, conflict: dirty };
}
