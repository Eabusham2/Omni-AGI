/**
 * Bounded renderer fallback for media that Chromium cannot play through an
 * Electron custom protocol. This is a transport threshold, never a generation
 * or model-output limit. Larger artifacts remain available through capability
 * URLs and download flows.
 */
export const INLINE_MEDIA_BINARY_BYTE_LIMIT = 512 * 1024;

/** Base64 expansion plus a bounded data MIME prefix. */
export const INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT =
  Math.ceil(INLINE_MEDIA_BINARY_BYTE_LIMIT / 3) * 4 + 128;
