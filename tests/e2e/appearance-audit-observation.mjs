/** Observational only: never changes focus, attributes, popovers or App state. */
export function createAppearanceAuditObservation(options = {}) {
  const capacity = options.capacity ?? 256;
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 4096) throw new Error("Invalid Appearance audit event capacity.");
  const events = [];
  let sequence = 0, dropped = 0, disconnect = () => {};
  const clone = value => JSON.parse(JSON.stringify(value));
  const record = (kind, detail = {}) => {
    const entry = { ...clone(detail), sequence: ++sequence, atMs: performance.now(), kind };
    events.push(entry);
    if (events.length > capacity) { events.shift(); dropped += 1; }
  };
  let current = () => null;
  const snapshot = () => clone(current());
  const read = () => ({ capacity, sequence, dropped, timeOriginMs: performance.timeOrigin,
    events: clone(events), current: snapshot() });

  if (options.install) {
    const doc = document;
    const describe = value => value instanceof Element ? {
      tag: value.tagName.toLowerCase(), classes: [...value.classList].slice(0, 8),
      role: value.getAttribute("role"), ariaModal: value.getAttribute("aria-modal"),
      inAppearance: Boolean(value.closest(".appearance-control,.appearance-panel"))
    } : { tag: value?.nodeName?.toLowerCase() ?? null };
    const visible = node => {
      if (!node?.isConnected || !node.getClientRects().length) return false;
      const style = getComputedStyle(node);
      return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0;
    };
    let previousPanel = null;
    current = () => {
      const root = doc.querySelector(".appearance-panel");
      const bounds = root?.getBoundingClientRect();
      if (root) previousPanel = root;
      return { rootPresent: Boolean(root), rootConnected: root?.isConnected ?? false,
        rootVisible: visible(root), popoverOpen: root?.matches(":popover-open") ?? false,
        expanded: doc.querySelector(".appearance-trigger")?.getAttribute("aria-expanded") ?? null,
        rootBounds: bounds ? { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height } : null,
        activeElement: describe(doc.activeElement),
        visibleModals: [...doc.querySelectorAll('dialog[open],[role="dialog"][aria-modal="true"]')].filter(visible).map(describe) };
    };
    const removers = [];
    for (const type of ["pointerdown", "click", "keydown", "focusin", "focusout", "beforetoggle", "toggle"]) {
      const listener = event => {
        const state = current();
        const appearanceTarget = event.target instanceof Element && Boolean(event.target.closest(".appearance-control,.appearance-panel"));
        if (!state.rootPresent && !appearanceTarget && !["beforetoggle", "toggle"].includes(type)) return;
        if (["beforetoggle", "toggle"].includes(type) && !appearanceTarget) return;
        // No keystroke contents, labels, input values or visible message text.
        const key = type === "keydown" ? ["Escape", "Tab", "Enter", " "].includes(event.key) ? event.key === " " ? "Space" : event.key : "Other" : undefined;
        record(`event:${type}`, { target: describe(event.target), trusted: event.isTrusted,
          ...(key ? { key } : {}), ...(type === "pointerdown" ? { button: event.button, pointerType: event.pointerType } : {}),
          ...(["beforetoggle", "toggle"].includes(type) ? { oldState: event.oldState, newState: event.newState } : {}), state });
      };
      doc.addEventListener(type, listener, true);
      removers.push(() => doc.removeEventListener(type, listener, true));
    }
    const observer = new MutationObserver(records => {
      const relevant = records.flatMap(item => {
        if (item.type === "attributes" && (item.target.matches(".appearance-trigger,.appearance-panel") ||
            ["open", "aria-modal"].includes(item.attributeName))) {
          return [{ type: "attribute", target: describe(item.target), attribute: item.attributeName }];
        }
        const affected = [...item.addedNodes, ...item.removedNodes].filter(node => node instanceof Element &&
          (node === previousPanel || node.contains(previousPanel) || node.matches('.appearance-control,.appearance-panel,dialog,[role="dialog"][aria-modal="true"]') ||
            node.querySelector('.appearance-panel,dialog[open],[role="dialog"][aria-modal="true"]')));
        return affected.length ? [{ type: "child-list", target: describe(item.target),
          added: [...item.addedNodes].filter(node => affected.includes(node)).map(describe),
          removed: [...item.removedNodes].filter(node => affected.includes(node)).map(describe) }] : [];
      });
      if (relevant.length) record("mutation", { mutations: relevant, state: current() });
    });
    const observe = () => observer.observe(doc.body, { childList: true, subtree: true, attributes: true,
      attributeFilter: ["aria-expanded", "open", "aria-modal", "popover"] });
    if (doc.body) observe(); else doc.addEventListener("DOMContentLoaded", observe, { once: true });
    disconnect = () => { observer.disconnect(); for (const remove of removers) remove(); doc.removeEventListener("DOMContentLoaded", observe); };
    Object.defineProperty(window, "__omniAppearanceAuditObservation", { value: { read, snapshot }, configurable: false });
  }
  return { record, read, snapshot, disconnect };
}

export function appearanceAuditFailureMessage(snapshot, phase) {
  if (!snapshot?.rootPresent || !snapshot.rootConnected) return `Appearance panel disappeared during ${phase}; preserve this failed state and its lifecycle observations.`;
  if (!snapshot.rootVisible || !snapshot.popoverOpen) return `Appearance panel stopped being visible in the native top layer during ${phase}; preserve this failed state and its lifecycle observations.`;
  return null;
}
