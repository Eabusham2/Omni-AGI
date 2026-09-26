import type { LiveVoiceState } from "../../shared/liveVoice";

export interface LiveVoiceStatusCopy {
  title: string;
  detail: string;
  tone: "neutral" | "active" | "warning";
}

export function liveVoiceStatusCopy(
  state: Readonly<LiveVoiceState>
): LiveVoiceStatusCopy {
  const skipped = state.speechBackpressure.skippedCharacters;
  const pressureDetail = skipped > 0
    ? ` ${skipped} audio characters were skipped to keep live delivery bounded; visible text is complete.`
    : "";
  if (state.error) {
    return {
      title:
        state.phase === "unavailable" ? "Live voice unavailable" : "Live voice needs attention",
      detail: state.error.message,
      tone: "warning"
    };
  }
  switch (state.phase) {
    case "starting":
      return {
        title: "Starting live voice",
        detail: "Waiting for explicit microphone permission…",
        tone: "neutral"
      };
    case "listening":
      return {
        title: state.interimTranscript || "Listening continuously",
        detail: state.interimTranscript
          ? "The live recognizer is forming this utterance."
          : state.preferences.deliveryMode === "live"
            ? `Speak naturally. A new speech start interrupts any current reply.${pressureDetail}`
            : "Speak naturally. Buffered replies finish before another utterance; use Stop to cancel.",
        tone: "active"
      };
    case "pondering":
      return {
        title: "Pondering the utterance",
        detail: state.preferences.deliveryMode === "live"
          ? `Speech begins incrementally. Speak again or choose Barge in to replace this turn.${pressureDetail}`
          : "Collecting the full reply before speech. Voice-triggered barge-in is off; Stop remains available.",
        tone: "active"
      };
    case "speaking":
      return {
        title: "Speaking the brain reply",
        detail: state.preferences.deliveryMode === "live"
          ? `The microphone remains live; speaking now immediately barges in.${pressureDetail}`
          : "Buffered delivery is speaking the complete reply. Use Stop to cancel it.",
        tone: "active"
      };
    case "stopping":
      return {
        title: "Stopping live voice",
        detail: "Closing microphone, recognition, speech, and the active turn…",
        tone: "neutral"
      };
    default:
      return {
        title: "Live voice ready",
        detail: "Microphone access begins only when you start it.",
        tone: "neutral"
      };
  }
}
