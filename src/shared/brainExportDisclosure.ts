/** Every export preserves the selected saved state; none is sanitized for sharing. */
export const BRAIN_EXPORT_DISCLOSURE =
  ".omni preserves saved chat, neural memory, temporary attention, pending learning, and recovery points without sanitizing their content. Share only with trusted recipients.";

export const BRAIN_EXPORT_CONFIRMATION_DETAIL =
  "Saved content is not redacted: chat, source paths, retained data, tool settings, temporary attention, pending learning, and recovery points may contain private information or secrets. The separate OS credential vault is excluded. Imports start dormant; unfinished dataset jobs still need their original sources and are not automatically resumed. This is a saved-state backup, not a running-process clone.";
