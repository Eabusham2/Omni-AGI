"""Typed warm-steer completion; no model, prompt or fabricated response text."""


def generation_completion(token_ids, decode, stopped_by_steer=False, private_steered=False, native_stopped=False):
    steered = bool(stopped_by_steer or private_steered)
    text = "" if private_steered else decode(token_ids)
    interrupted = steered or native_stopped
    result = {"text": text if interrupted else text.strip(), "steered": steered,
              "nativeStopped": bool(native_stopped and not steered),
              "zeroTokenYield": interrupted and not text}
    if not interrupted and not any(
        character.isprintable() and not character.isspace() for character in text
    ):
        # This is an observation about the decoder's output, not evidence of
        # intent, refusal, cancellation or an ethical decision. Do not decode
        # hidden special-token labels or invent prose to fill this completion.
        result["noReply"] = True
        result["text"] = ""
        result["noReplyReason"] = (
            "whitespace-only" if text and not text.strip() else
            "no-printable-text" if text else
            "no-decoded-text" if len(token_ids) else "no-generated-tokens"
        )
    return result


def mark_interrupted_turn(human, assistant, receipt, disposition="steered"):
    if disposition not in {"steered", "native-stop"}:
        raise ValueError("invalid generation disposition")
    human["generation_end"] = disposition
    assistant["generation_end"] = disposition
    receipt["generationEnd"] = disposition


def mark_no_reply_turn(human, assistant, receipt):
    """Bind a zero-text, completed turn without pretending an interruption."""
    if assistant.get("content") != "":
        raise ValueError("no-reply completion must have zero assistant text")
    human["generation_end"] = "no-reply"
    assistant["generation_end"] = "no-reply"
    receipt["generationEnd"] = "no-reply"


def validate_no_reply_turn(human, assistant, trace, receipt):
    """Require exact saved evidence; an empty row alone is not a completion."""
    markers = (
        human.get("generation_end"), assistant.get("generation_end"),
        receipt.get("generationEnd"), trace.get("generation_stop_reason"),
    )
    if "no-reply" not in markers:
        return False
    if (
        any(marker != "no-reply" for marker in markers)
        or assistant.get("content") != ""
        or trace.get("generation_no_reply_reason") not in {
            "no-generated-tokens", "no-decoded-text", "whitespace-only", "no-printable-text"
        }
        or type(trace.get("generation_printable_text_characters")) is not int
        or trace["generation_printable_text_characters"] != 0
        or type(trace.get("generated_token_count")) is not int
        or trace["generated_token_count"] < 0
        or (trace["generation_no_reply_reason"] == "no-generated-tokens") != (trace["generated_token_count"] == 0)
        or not isinstance(trace.get("generation_decoder_stop_reason"), str)
        or not trace["generation_decoder_stop_reason"]
    ):
        raise ValueError("no-reply completion is not bound to saved zero-text evidence")
    return True
