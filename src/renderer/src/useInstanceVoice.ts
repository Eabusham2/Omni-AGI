import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import type { LiveVoicePreferences } from "../../shared/liveVoice";
import { instanceVoiceSettingsFromLegacy, resolveInstanceVoice, type InstanceVoiceApi,
  type InstanceVoiceSettings, type InstanceVoiceSnapshot } from "../../shared/voiceRouting";
import { LIVE_VOICE_STORAGE_KEY } from "./liveVoicePreferences";
import type { LiveVoiceController } from "./liveVoiceController";

/** Missing/corrupt global values are not explicit preferences for a new brain. */
export function savedLegacyVoice(raw: string | null): LiveVoicePreferences | undefined {
  if (!raw) return;
  try {
    const value: unknown = JSON.parse(raw);
    const migrated = instanceVoiceSettingsFromLegacy(value);
    return { schemaVersion: 1, deliveryMode: migrated.deliveryMode, pace: migrated.pace,
      neuralListening: migrated.input === "native", neuralVoice: migrated.output === "native" };
  } catch { return; }
}

/** Host preferences are independent from neural state and never implicitly Load. */
export function useInstanceVoice(brainId: string, runtimeLoaded: boolean, callOpen: boolean,
  controllerRef: MutableRefObject<LiveVoiceController | null>, api?: InstanceVoiceApi) {
  const [snapshot, setSnapshot] = useState<InstanceVoiceSnapshot>();
  const [error, setError] = useState("");
  const current = useRef({ brainId, runtimeLoaded, request: 0 });
  if (current.current.brainId !== brainId || current.current.runtimeLoaded !== runtimeLoaded) {
    current.current = { brainId, runtimeLoaded, request: current.current.request + 1 };
  }
  const snapshotRef = useRef(snapshot);
  snapshotRef.current = snapshot?.brainId === brainId ? snapshot : undefined;

  const read = useCallback(async (): Promise<InstanceVoiceSnapshot> => {
    if (!api) throw new Error("Instance voice settings require the desktop runtime.");
    const scope = current.current;
    const request = ++scope.request;
    const isCurrent = () => current.current === scope && scope.request === request;
    let next = await api.preferences(brainId);
    if (!isCurrent()) throw new Error("Voice settings changed while this request was pending.");
    if (next.legacyMigrationAllowed) {
      let legacy: LiveVoicePreferences | undefined;
      try { legacy = savedLegacyVoice(window.localStorage.getItem(LIVE_VOICE_STORAGE_KEY)); } catch { /* Storage may be unavailable. */ }
      if (legacy) next = await api.migrateLegacy({ brainId, legacy });
    }
    if (!isCurrent()) throw new Error("Voice settings changed while this request was pending.");
    if (next.brainId !== brainId) throw new Error("Voice settings belong to a different instance.");
    snapshotRef.current = next;
    setSnapshot(next); setError("");
    return next;
  }, [api, brainId]);

  useEffect(() => {
    let active = true;
    if (!api) { setSnapshot(undefined); setError("Instance voice settings require the desktop runtime."); return; }
    void read().catch(failure => { if (active) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { active = false; current.current.request += 1; };
  }, [api, brainId, runtimeLoaded, callOpen, read]);

  const save = useCallback(async (preferences: InstanceVoiceSettings): Promise<void> => {
    const existing = snapshotRef.current;
    const controller = controllerRef.current;
    if (!api || !existing || existing.brainId !== brainId) throw new Error("Saved voice settings are not ready yet.");
    const state = controller?.getState();
    if (state?.activeUtterance) throw new Error("Wait for the current utterance before changing voice settings.");
    if (state?.enabled && (preferences.input !== existing.preferences.input || preferences.output !== existing.preferences.output)) {
      throw new Error("End this call before changing its input or output route.");
    }
    const scope = current.current;
    const request = ++scope.request;
    try {
      const next = await api.update({ brainId, expectedRevision: existing.revision, preferences });
      if (current.current !== scope || scope.request !== request || controllerRef.current !== controller) {
        throw new Error("The loaded instance changed before these voice settings could be applied.");
      }
      snapshotRef.current = next; setSnapshot(next); setError("");
      if (controller) {
        const live = controller.getState();
        if (live.enabled) {
          // A refreshed count can cross the Auto threshold. Keep the current
          // capture/output route until the next call, never mid-recognizer.
          controller.setPreferences({ ...live.preferences, pace: next.preferences.pace, deliveryMode: next.preferences.deliveryMode });
        } else {
          const resolved = resolveInstanceVoice(next.preferences, next.parameterEvidence, live.capabilities);
          if (resolved.controllerPreferences) controller.setPreferences(resolved.controllerPreferences);
        }
      }
    } catch (failure) {
      // Read actual CAS state after a stale save instead of retaining a false choice.
      if (current.current === scope && scope.request === request) await read().catch(() => undefined);
      throw failure;
    }
  }, [api, brainId, controllerRef, read]);

  const prepareCall = useCallback(async (): Promise<void> => {
    const controller = controllerRef.current;
    if (!controller || !current.current.runtimeLoaded) throw new Error("Load the instance before starting a voice call.");
    const scope = current.current;
    const request = scope.request + 1;
    const [next, capabilities] = await Promise.all([read(), controller.readyCapabilities()]);
    if (current.current !== scope || scope.request !== request || controllerRef.current !== controller || !scope.runtimeLoaded) {
      throw new Error("The loaded instance changed before this call started.");
    }
    const resolved = resolveInstanceVoice(next.preferences, next.parameterEvidence, capabilities);
    if (!resolved.ready || !resolved.controllerPreferences) {
      throw new Error([!resolved.input.available ? resolved.input.detail : "", !resolved.output.available ? resolved.output.detail : ""].filter(Boolean).join(" "));
    }
    controller.setPreferences(resolved.controllerPreferences);
  }, [controllerRef, read]);

  return { snapshot: snapshot?.brainId === brainId ? snapshot : undefined, error, save, prepareCall };
}
