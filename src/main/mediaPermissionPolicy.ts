export interface StudioMediaPermissionRequest {
  trustedWebContentsId?: number;
  requestingWebContentsId?: number;
  permission: string;
  isMainFrame: boolean;
  requestingUrl?: string;
  currentRendererUrl?: string;
  mediaTypes: readonly string[];
}

export type StudioMediaType = "audio" | "video";

export interface StudioDisplayPermissionRequest {
  userGesture: boolean;
  isMainFrame: boolean;
  requestingUrl?: string;
  currentRendererUrl?: string;
}

function sameRendererDocument(requestingUrl?: string, currentUrl?: string): boolean {
  if (!requestingUrl || !currentUrl) return false;
  try {
    const requested = new URL(requestingUrl);
    const current = new URL(currentUrl);
    requested.hash = "";
    current.hash = "";
    return requested.href === current.href;
  } catch {
    return false;
  }
}

/**
 * The studio's default Electron session denies every permission except an
 * explicit microphone/camera request made by the loaded top-level studio
 * renderer. Subframes, navigated pages, empty/unknown device sets, and every
 * other permission stay denied. Chromium and the OS still present their
 * device choices when the person starts Live Voice or Live Perception.
 */
export function allowTrustedStudioMedia(
  request: StudioMediaPermissionRequest
): boolean {
  const mediaTypes = [...new Set(request.mediaTypes)];
  return (
    request.permission === "media" &&
    request.isMainFrame &&
    request.trustedWebContentsId !== undefined &&
    request.requestingWebContentsId === request.trustedWebContentsId &&
    mediaTypes.length > 0 &&
    mediaTypes.length === request.mediaTypes.length &&
    mediaTypes.every((type): type is StudioMediaType =>
      type === "audio" || type === "video"
    ) &&
    sameRendererDocument(request.requestingUrl, request.currentRendererUrl)
  );
}

/** Compatibility helper retained for callers that intentionally request mic only. */
export function allowTrustedAudioMedia(
  request: StudioMediaPermissionRequest
): boolean {
  return (
    request.mediaTypes.length === 1 &&
    request.mediaTypes[0] === "audio" &&
    allowTrustedStudioMedia(request)
  );
}

/** Screen observation must be started by a gesture in the loaded studio page. */
export function allowTrustedDisplayMedia(
  request: StudioDisplayPermissionRequest
): boolean {
  return (
    request.userGesture &&
    request.isMainFrame &&
    sameRendererDocument(request.requestingUrl, request.currentRendererUrl)
  );
}
