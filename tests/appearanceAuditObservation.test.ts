import assert from "node:assert/strict";
import { test } from "vitest";
import { appearanceAuditFailureMessage, createAppearanceAuditObservation } from "./e2e/appearance-audit-observation.mjs";

test("Appearance observations retain the newest bounded events with truthful dropped counts", () => {
  const recorder = createAppearanceAuditObservation({ capacity: 2 });
  recorder.record("first", { target: { tag: "button" } });
  recorder.record("second");
  recorder.record("third");
  const snapshot = recorder.read();
  assert.equal(snapshot.capacity, 2);
  assert.equal(snapshot.sequence, 3);
  assert.equal(snapshot.dropped, 1);
  assert.deepEqual(snapshot.events.map(event => [event.sequence, event.kind]), [[2, "second"], [3, "third"]]);
  assert.ok(snapshot.events.every(event => Number.isFinite(event.atMs)));
  assert.ok(Number.isFinite(snapshot.timeOriginMs));
  assert.equal(snapshot.current, null);
});

test("Observation snapshots cannot mutate retained evidence", () => {
  const recorder = createAppearanceAuditObservation();
  const detail = { target: { tag: "button" } };
  recorder.record("event:focusin", detail);
  detail.target.tag = "changed";
  const first = recorder.read();
  (first.events[0]!.target as { tag: string }).tag = "tampered";
  first.events.push({ sequence: -1, atMs: -1, kind: "forged" });
  assert.equal((recorder.read().events[0]!.target as { tag: string }).tag, "button");
  assert.equal(recorder.read().events.length, 1);
});

test("Invalid event bounds fail instead of allowing an unbounded audit buffer", () => {
  for (const capacity of [0, -1, 1.5, NaN, Infinity, 4097]) assert.throws(() => createAppearanceAuditObservation({ capacity }), /Invalid Appearance/);
});

test("Details cannot replace the recorder's measured timestamp, sequence or event kind", () => {
  const recorder = createAppearanceAuditObservation();
  recorder.record("event:focusout", { sequence: -1, atMs: -1, kind: "forged" });
  const event = recorder.read().events[0]!;
  assert.equal(event.sequence, 1);
  assert.equal(event.kind, "event:focusout");
  assert.ok(event.atMs >= 0);
});

test("Detached or hidden native popup states fail explicitly; a real open panel does not", () => {
  const open = { rootPresent: true, rootConnected: true, rootVisible: true, popoverOpen: true,
    expanded: "true", rootBounds: null, activeElement: {}, visibleModals: [] };
  assert.equal(appearanceAuditFailureMessage(open, "control base paint"), null);
  assert.match(appearanceAuditFailureMessage(null, "control base paint") ?? "", /disappeared during control base paint/);
  assert.match(appearanceAuditFailureMessage({ ...open, rootConnected: false }, "scroll") ?? "", /disappeared during scroll/);
  assert.match(appearanceAuditFailureMessage({ ...open, popoverOpen: false }, "hover") ?? "", /stopped being visible.*during hover/);
  assert.match(appearanceAuditFailureMessage({ ...open, rootVisible: false }, "focus") ?? "", /stopped being visible.*during focus/);
});
