import { createReadStream } from "node:fs";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";
import type { AuthorizedMediaArtifact } from "./mediaArtifactRegistry";

export const OMNI_MEDIA_SCHEME = "omni-media";

/**
 * These privileges are part of the playback contract, not just security
 * metadata. Electron buffers custom schemes by default; Chromium's native
 * audio/video elements only consume a streamed Response when `stream` is set
 * before app readiness.
 */
export const OMNI_MEDIA_SCHEME_PRIVILEGES = Object.freeze({
  standard: true,
  secure: true,
  supportFetchAPI: true,
  corsEnabled: false,
  stream: true
});

interface ByteRange {
  start: number;
  end: number;
}

export type ElectronFileFetcher = (
  url: string,
  init: RequestInit & { bypassCustomProtocolHandlers: true }
) => Promise<Response>;

function requestedByteRange(value: string | null, size: number): ByteRange | null | false {
  if (!value) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return false;
  const first = match[1] ? Number(match[1]) : undefined;
  const last = match[2] ? Number(match[2]) : undefined;
  if (
    (first !== undefined && !Number.isSafeInteger(first)) ||
    (last !== undefined && !Number.isSafeInteger(last))
  ) return false;
  if (first === undefined) {
    const suffix = last ?? 0;
    if (suffix < 1) return false;
    return { start: Math.max(0, size - suffix), end: size - 1 };
  }
  if (first < 0 || first >= size) return false;
  const end = Math.min(size - 1, last ?? size - 1);
  return end < first ? false : { start: first, end };
}

function mediaHeaders(artifact: AuthorizedMediaArtifact): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
    "Content-Type": artifact.mimeType,
    "ETag": `"sha256-${artifact.sha256}"`,
    "X-Content-Type-Options": "nosniff"
  });
}

/**
 * Chromium's native audio/video controls probe custom schemes with byte-range
 * requests. file:// net.fetch can answer those probes with a full 200 response,
 * which images tolerate but media demuxers reject. Stream the exact authorized
 * range and advertise seek semantics explicitly.
 */
export function authorizedMediaResponse(
  request: Request,
  artifact: AuthorizedMediaArtifact
): Response {
  const headers = mediaHeaders(artifact);
  if (request.method !== "GET" && request.method !== "HEAD") {
    headers.set("Allow", "GET, HEAD");
    return new Response(null, { status: 405, headers });
  }
  const range = requestedByteRange(request.headers.get("range"), artifact.size);
  if (range === false) {
    headers.set("Content-Range", `bytes */${artifact.size}`);
    headers.set("Content-Length", "0");
    return new Response(null, { status: 416, headers });
  }
  const start = range?.start ?? 0;
  const end = range?.end ?? artifact.size - 1;
  const length = end - start + 1;
  headers.set("Content-Length", String(length));
  if (range) headers.set("Content-Range", `bytes ${start}-${end}/${artifact.size}`);
  const body = request.method === "HEAD"
    ? null
    : Readable.toWeb(createReadStream(artifact.path, { start, end })) as unknown as BodyInit;
  return new Response(body, {
    status: range ? 206 : 200,
    headers
  });
}

/**
 * Serve playback through Chromium's native file URL loader after capability
 * authorization. Electron 41+ on macOS can reject audio/video backed by a
 * manually constructed protocol.handle Response even when its bytes and Range
 * headers are correct. Returning net.fetch(file://...) keeps the opaque lease
 * at the renderer boundary while letting Chromium own the media data pipe.
 */
export function authorizedNativeMediaResponse(
  request: Request,
  artifact: AuthorizedMediaArtifact,
  fetchFile: ElectronFileFetcher
): Response | Promise<Response> {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return authorizedMediaResponse(request, artifact);
  }
  const range = requestedByteRange(request.headers.get("range"), artifact.size);
  if (range === false) return authorizedMediaResponse(request, artifact);
  const headers = new Headers();
  if (range) headers.set("Range", `bytes=${range.start}-${range.end}`);
  return fetchFile(pathToFileURL(artifact.path).toString(), {
    method: request.method,
    headers,
    bypassCustomProtocolHandlers: true
  });
}
