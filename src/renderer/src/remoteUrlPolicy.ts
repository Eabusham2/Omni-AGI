/**
 * Renderer-only admission hint. The main process remains authoritative and
 * resolves localhost before fetching; this helper only keeps clearly invalid
 * remote HTTP/private-host spellings from looking submit-ready in the UI.
 */
export function webLearningUrlAllowed(value: string): boolean {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    return false;
  }
  if (url.username || url.password) return false;
  if (url.protocol === "https:") return true;
  if (url.protocol !== "http:") return false;
  const hostname = url.hostname
    .replace(/^\[|\]$/g, "")
    .replace(/\.+$/, "")
    .toLocaleLowerCase();
  if (hostname === "localhost" || hostname === "::1") return true;
  const octets = hostname.split(".");
  return octets.length === 4 &&
    octets.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255) &&
    Number(octets[0]) === 127;
}
