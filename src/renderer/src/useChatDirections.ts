import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SteerDirectionRecord, SteerDirectionsApi } from "../../shared/steerDirections";
import { mergeChatDirections } from "./chatDirectionPresentation";
import { visitChatDirectionTurns } from "./chatDirectionQueries";

export function useChatDirections(brainId: string, visibleTurnIds: string[], api?: SteerDirectionsApi) {
  const [records, setRecords] = useState<SteerDirectionRecord[]>([]), [total, setTotal] = useState(0);
  const [before, setBefore] = useState<number>(), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const scope = useRef({ brainId, read: 0 });
  if (scope.current.brainId !== brainId) scope.current = { brainId, read: 0 };
  const recordsRef = useRef(records); recordsRef.current = records;
  const idsKey = JSON.stringify([...new Set(visibleTurnIds)].sort());
  const merge = useCallback((values: SteerDirectionRecord[]) => {
    if (values.some(value => value.brainId !== brainId)) throw new Error("Directions belong to a different instance.");
    // Validate outside a React updater so an async error remains recoverable.
    const next = mergeChatDirections(recordsRef.current.filter(value => value.brainId === brainId), values);
    recordsRef.current = next; setRecords(next);
  }, [brainId]);

  useEffect(() => {
    const owner = scope.current; let active = true;
    recordsRef.current = []; setRecords([]); setTotal(0); setBefore(undefined); setError("");
    if (!api) return;
    const accept = (values: SteerDirectionRecord[]): void => { if (active && scope.current === owner) merge(values); };
    const fail = (failure: unknown): void => { if (active && scope.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); };
    let unsubscribe = () => {};
    try { unsubscribe = api.onEvent(event => {
      if (event.brainId !== brainId || !active) return;
      const request = ++owner.read;
      void Promise.all([api.get(brainId, event.id), api.list(brainId, { limit: 1, includeDismissed: true })]).then(([record, page]) => {
        if (!active || scope.current !== owner) return;
        if (record) accept([record]);
        if (owner.read === request) setTotal(page.total);
      }).catch(fail);
    }); } catch (failure) { fail(failure); }
    const request = ++owner.read;
    setBusy(true);
    void api.list(brainId, { limit: 40, includeDismissed: true }).then(page => {
      if (!active || scope.current !== owner) return;
      accept(page.entries); setBefore(page.nextBeforeSequence);
      if (owner.read === request) setTotal(page.total);
    }).catch(fail).finally(() => { if (active && scope.current === owner) setBusy(false); });
    return () => { active = false; unsubscribe(); };
  }, [api, brainId, merge]);

  useEffect(() => {
    if (!api) return;
    const owner = scope.current; let active = true;
    const ids = JSON.parse(idsKey) as string[];
    void visitChatDirectionTurns(api, brainId, ids, merge, () => active && scope.current === owner)
      .catch(failure => { if (active && scope.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { active = false; };
  }, [api, brainId, idsKey, merge]);

  const older = async (): Promise<void> => {
    if (!api || before === undefined || busy) return;
    const owner = scope.current, request = ++owner.read; setBusy(true); setError("");
    try {
      const page = await api.list(brainId, { beforeSequence: before, limit: 40, includeDismissed: true });
      if (scope.current !== owner) return;
      merge(page.entries); setBefore(page.nextBeforeSequence);
      if (owner.read === request) setTotal(page.total);
    } catch (failure) { if (scope.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (scope.current === owner) setBusy(false); }
  };
  const dismiss = async (id: string): Promise<void> => {
    const current = recordsRef.current.find(value => value.id === id && value.brainId === brainId);
    if (!api || !current) throw new Error("This direction is not available to dismiss.");
    const owner = scope.current;
    const saved = await api.dismiss({ brainId, id, expectedRevision: current.revision, dismissed: true });
    if (scope.current === owner) merge([saved]);
  };
  // Keep the same owner's projection stable between actual record changes.
  // Consumers derive persisted messages and asynchronous prefix verification
  // from this list; allocating it every render can retrigger those effects.
  const visibleRecords = useMemo(() => records.filter(value => value.brainId === brainId), [records, brainId]);
  return { records: visibleRecords, total, busy, error, hasOlder: before !== undefined, older, dismiss };
}
