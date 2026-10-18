import { lstat } from "node:fs/promises";
import { join } from "node:path";
import type { BrainRepository } from "./brainRepository";
import { instanceVoicePreferences, VoicePreferencesStore } from "./voicePreferencesStore";
import { validateInstanceVoiceSettings, type InstanceVoiceLegacyRequest, type InstanceVoiceSnapshot,
  type InstanceVoiceUpdateRequest } from "../shared/voiceRouting";

/** Metadata/settings only. Never starts, loads, trains or waits on a neural writer. */
export class InstanceVoiceService {
  constructor(private readonly repository: Pick<BrainRepository, "brainDirectory" | "voiceParameterEvidence">) {}
  private async root(brainId: string): Promise<string> {
    const root = this.repository.brainDirectory(brainId), directory = await lstat(root);
    if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("Voice preferences require an owned instance directory.");
    const info = await lstat(join(root, "brain.json"));
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Voice preferences require an existing managed instance.");
    return root;
  }
  async preferences(brainId: string): Promise<InstanceVoiceSnapshot> {
    const preferences = await instanceVoicePreferences(await this.root(brainId), brainId);
    try {
      const evidence = await this.repository.voiceParameterEvidence(brainId);
      return { ...preferences, parameterEvidence: evidence,
        parameterDetail: "Logical parameters measured by the native brain and bound to its current committed runtime card/checkpoint; not an advertised architecture size." };
    } catch {
      return { ...preferences, parameterEvidence: null,
        parameterDetail: "Verified committed native parameter accounting is not available yet. Automatic output remains unresolved; a manual route can be selected without assuming a model size." };
    }
  }
  async update(request: InstanceVoiceUpdateRequest): Promise<InstanceVoiceSnapshot> {
    if (!request || typeof request !== "object" || Object.keys(request).some(key => !["brainId", "expectedRevision", "preferences"].includes(key))) throw new Error("Invalid voice preference update.");
    const root = await this.root(request.brainId), preferences = validateInstanceVoiceSettings(request.preferences);
    const store = (await VoicePreferencesStore.open(root, request.brainId, true))!;
    try { store.update(request.expectedRevision, preferences); } finally { store.close(); }
    return this.preferences(request.brainId);
  }
  async migrateLegacy(request: InstanceVoiceLegacyRequest): Promise<InstanceVoiceSnapshot> {
    if (!request || typeof request !== "object" || Object.keys(request).some(key => !["brainId", "legacy"].includes(key))) throw new Error("Invalid legacy voice migration.");
    const root = await this.root(request.brainId);
    const store = (await VoicePreferencesStore.open(root, request.brainId, true))!;
    try { store.migrateLegacy(request.legacy); } finally { store.close(); }
    return this.preferences(request.brainId);
  }
}
