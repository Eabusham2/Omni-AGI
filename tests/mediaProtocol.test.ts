import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import {
  authorizedNativeMediaResponse,
  authorizedMediaResponse,
  OMNI_MEDIA_SCHEME,
  OMNI_MEDIA_SCHEME_PRIVILEGES
} from "../src/main/mediaProtocol";

const roots: string[] = [];
const leasedMediaUrl =
  `omni-media://artifact/${"0".repeat(48)}/${"a".repeat(64)}`;

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ));
});

function playableWavBytes(): Buffer {
  const pcm = Buffer.alloc(320, 0);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(16_000, 24);
  header.writeUInt32LE(32_000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

describe("native generated-media protocol", () => {
  it("registers a privileged streaming scheme for packaged audio and video", () => {
    expect(OMNI_MEDIA_SCHEME).toBe("omni-media");
    expect(OMNI_MEDIA_SCHEME_PRIVILEGES).toEqual({
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: false,
      stream: true
    });
  });

  it("answers Chromium audio probes with exact seekable byte ranges", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-media-protocol-"));
    roots.push(root);
    const path = join(root, "generated.wav");
    const bytes = playableWavBytes();
    await writeFile(path, bytes);
    const artifact = {
      path,
      mimeType: "audio/wav",
      size: bytes.length,
      sha256: "a".repeat(64)
    };

    const headerProbe = authorizedMediaResponse(new Request(leasedMediaUrl, {
      headers: { Range: "bytes=0-43" }
    }), artifact);
    expect(headerProbe.status).toBe(206);
    expect(headerProbe.headers.get("content-type")).toBe("audio/wav");
    expect(headerProbe.headers.get("accept-ranges")).toBe("bytes");
    expect(headerProbe.headers.get("content-range")).toBe(
      `bytes 0-43/${bytes.length}`
    );
    expect(Buffer.from(await headerProbe.arrayBuffer())).toEqual(bytes.subarray(0, 44));

    const tailProbe = authorizedMediaResponse(new Request(leasedMediaUrl, {
      headers: { Range: "bytes=-16" }
    }), artifact);
    expect(tailProbe.status).toBe(206);
    expect(Buffer.from(await tailProbe.arrayBuffer())).toEqual(bytes.subarray(-16));

    const full = authorizedMediaResponse(
      new Request(leasedMediaUrl),
      artifact
    );
    expect(full.status).toBe(200);
    expect(full.headers.get("content-length")).toBe(String(bytes.length));
    expect(Buffer.from(await full.arrayBuffer())).toEqual(bytes);

    const openEnded = authorizedMediaResponse(new Request(leasedMediaUrl, {
      headers: { Range: "bytes=44-" }
    }), artifact);
    expect(openEnded.status).toBe(206);
    expect(openEnded.headers.get("content-range")).toBe(
      `bytes 44-${bytes.length - 1}/${bytes.length}`
    );
    expect(openEnded.headers.get("content-length")).toBe(String(bytes.length - 44));
    expect(Buffer.from(await openEnded.arrayBuffer())).toEqual(bytes.subarray(44));
  });

  it("delegates authorized playback to Electron's native file loader", async () => {
    const path = join(tmpdir(), "Omni media", "generated audio.wav");
    const calls: Array<{ url: string; init: RequestInit & { bypassCustomProtocolHandlers: true } }> = [];
    const native = new Response(Buffer.from("native file response"), {
      status: 206,
      headers: { "Content-Type": "audio/wav" }
    });
    const response = await authorizedNativeMediaResponse(
      new Request(leasedMediaUrl, { headers: { Range: "bytes=12-47" } }),
      {
        path,
        mimeType: "audio/wav",
        size: 100,
        sha256: "e".repeat(64)
      },
      async (url, init) => {
        calls.push({ url, init });
        return native;
      }
    );

    expect(response).toBe(native);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe(pathToFileURL(path).href);
    expect(calls[0]?.init.method).toBe("GET");
    expect(calls[0]?.init.bypassCustomProtocolHandlers).toBe(true);
    expect(new Headers(calls[0]?.init.headers).get("range")).toBe("bytes=12-47");
  });

  it("keeps invalid capability methods and ranges out of the native file loader", async () => {
    let fetches = 0;
    const fetchFile = async () => {
      fetches += 1;
      return new Response();
    };
    const artifact = {
      path: "/not-opened.wav",
      mimeType: "audio/wav",
      size: 364,
      sha256: "f".repeat(64)
    };
    const invalidRange = await authorizedNativeMediaResponse(
      new Request(leasedMediaUrl, { headers: { Range: "bytes=364-" } }),
      artifact,
      fetchFile
    );
    const invalidMethod = await authorizedNativeMediaResponse(
      new Request(leasedMediaUrl, { method: "POST" }),
      artifact,
      fetchFile
    );
    expect(invalidRange.status).toBe(416);
    expect(invalidMethod.status).toBe(405);
    expect(fetches).toBe(0);
  });

  it("answers GET and HEAD probes with identical metadata but no HEAD body", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-media-head-"));
    roots.push(root);
    const path = join(root, "generated.wav");
    const bytes = playableWavBytes();
    await writeFile(path, bytes);
    const artifact = {
      path,
      mimeType: "audio/wav",
      size: bytes.length,
      sha256: "c".repeat(64)
    };

    const head = authorizedMediaResponse(new Request(leasedMediaUrl, {
      method: "HEAD"
    }), artifact);
    expect(head.status).toBe(200);
    expect(head.headers.get("content-type")).toBe("audio/wav");
    expect(head.headers.get("content-length")).toBe(String(bytes.length));
    expect(head.headers.get("accept-ranges")).toBe("bytes");
    expect((await head.arrayBuffer()).byteLength).toBe(0);

    const rangedHead = authorizedMediaResponse(new Request(leasedMediaUrl, {
      method: "HEAD",
      headers: { Range: "bytes=0-43" }
    }), artifact);
    expect(rangedHead.status).toBe(206);
    expect(rangedHead.headers.get("content-range")).toBe(
      `bytes 0-43/${bytes.length}`
    );
    expect(rangedHead.headers.get("content-length")).toBe("44");
    expect((await rangedHead.arrayBuffer()).byteLength).toBe(0);
  });

  it("rejects malformed and unsatisfiable media ranges deterministically", () => {
    const artifact = {
      path: "/not-opened-for-invalid-ranges.wav",
      mimeType: "audio/wav",
      size: 364,
      sha256: "b".repeat(64)
    };
    for (const range of ["bytes=364-", "bytes=9-2", "bytes=0-1,4-5", "items=0-2"]) {
      const response = authorizedMediaResponse(new Request(leasedMediaUrl, {
        headers: { Range: range }
      }), artifact);
      expect(response.status).toBe(416);
      expect(response.headers.get("content-range")).toBe("bytes */364");
    }
  });

  it("rejects non-playback methods without opening the artifact", () => {
    const response = authorizedMediaResponse(new Request(leasedMediaUrl, {
      method: "POST"
    }), {
      path: "/not-opened-for-method-rejection.wav",
      mimeType: "audio/wav",
      size: 364,
      sha256: "d".repeat(64)
    });
    expect(response.status).toBe(405);
    expect(response.headers.get("allow")).toBe("GET, HEAD");
    expect(response.headers.get("content-length")).toBeNull();
  });
});
