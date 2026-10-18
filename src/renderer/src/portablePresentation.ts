import { checkedPortablePresentation, type PortablePresentationSettings, type PortableWorkspaceStatus } from "../../shared/portableWorkspace";

/** Apply the exported ordinary preferences once, not on every later launch. */
export function applyPortablePresentationOnce(status: PortableWorkspaceStatus,
  storage: Pick<Storage, "getItem" | "setItem">, apply: (value: PortablePresentationSettings) => void): boolean {
  if (!status.portable || !status.presentation || !status.workspaceId || !status.settingsSha256) return false;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(status.workspaceId) || !/^[a-f0-9]{64}$/.test(status.settingsSha256))
    throw new Error("Portable presentation has no verified workspace/settings binding.");
  const key = `omni.portable.presentation.${status.workspaceId}`;
  if (storage.getItem(key) === status.settingsSha256) return false;
  const presentation = checkedPortablePresentation(status.presentation);
  apply(presentation); storage.setItem(key, status.settingsSha256); return true;
}
