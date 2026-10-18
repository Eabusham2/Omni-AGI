import type { NativeAudioMessageInfo } from "../../shared/nativeAudioChat";
import { Icon } from "./icons";

/** Presentation of a real receipt; never synthesizes a transcript or prompt. */
export function NativeAudioInputSummary({ input }: { input: NativeAudioMessageInfo }) {
  return <details className="native-audio-input">
    <summary><Icon name="volume" size={15} /> Voice message</summary>
    <p>Sent directly as audio, without a platform transcript.</p>
    <p>{input.mimeType} · {input.sourceBytes.toLocaleString()} bytes · {input.decodedWindows.toLocaleString()} decoded audio windows</p>
    <p>{input.modelSamples.toLocaleString()} model-rate samples at {input.modelSampleRate.toLocaleString()} Hz. All source audio was traversed.</p>
    <p className="native-audio-input__hash">Source SHA-256: {input.sourceSha256}</p>
  </details>;
}
