/** Header-only admission before the renderer's raster decoder allocates pixels. */
export function mapRasterImageInfo(bytes: Uint8Array): { mime: "image/png" | "image/jpeg" | "image/webp"; width: number; height: number } {
  const data = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let mime: "image/png" | "image/jpeg" | "image/webp", width = 0, height = 0;
  if (data.length >= 24 && data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) && data.toString("ascii", 12, 16) === "IHDR") {
    mime = "image/png"; width = data.readUInt32BE(16); height = data.readUInt32BE(20);
  } else if (data.length >= 12 && data.toString("ascii", 0, 4) === "RIFF" && data.toString("ascii", 8, 12) === "WEBP") {
    mime = "image/webp";
    if (data.readUInt32LE(4) + 8 !== data.length) throw new Error("Map WebP container length is invalid.");
    for (let at = 12; at + 8 <= data.length;) {
      const type = data.toString("ascii", at, at + 4), size = data.readUInt32LE(at + 4), start = at + 8;
      if (start + size > data.length) throw new Error("Map WebP chunk is incomplete.");
      if (type === "VP8X" && size >= 10) { width = 1 + data.readUIntLE(start + 4, 3); height = 1 + data.readUIntLE(start + 7, 3); break; }
      if (type === "VP8L" && size >= 5 && data[start] === 0x2f) { const bits = data.readUInt32LE(start + 1); width = 1 + (bits & 0x3fff); height = 1 + ((bits >>> 14) & 0x3fff); break; }
      if (type === "VP8 " && size >= 10 && data[start + 3] === 0x9d && data[start + 4] === 0x01 && data[start + 5] === 0x2a) { width = data.readUInt16LE(start + 6) & 0x3fff; height = data.readUInt16LE(start + 8) & 0x3fff; break; }
      at = start + size + (size & 1);
    }
  } else if (data.length >= 4 && data[0] === 0xff && data[1] === 0xd8) {
    mime = "image/jpeg";
    for (let at = 2; at + 1 < data.length;) {
      if (data[at++] !== 0xff) throw new Error("Map JPEG marker is invalid.");
      while (data[at] === 0xff) at++;
      const marker = data[at++];
      if (marker === 0xd9 || marker === 0xda) break;
      if (marker === 0x01 || marker !== undefined && marker >= 0xd0 && marker <= 0xd7) continue;
      if (at + 2 > data.length) break;
      const length = data.readUInt16BE(at); if (length < 2 || at + length > data.length) throw new Error("Map JPEG segment is incomplete.");
      if (marker !== undefined && [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
        if (length < 8) throw new Error("Map JPEG frame is invalid."); height = data.readUInt16BE(at + 3); width = data.readUInt16BE(at + 5); break;
      }
      at += length;
    }
  } else throw new Error("Map service did not return a supported PNG, JPEG or WebP raster. SVG, HTML and scripts are not accepted.");
  if (!width || !height || !Number.isSafeInteger(width * height * 4)) throw new Error("Map raster dimensions are unavailable or exceed exact pixel admission.");
  return { mime, width, height };
}
