"""Typed warm-steer completion; no model, prompt or fabricated response text."""
import hashlib


def public_reply_text_binding(raw, canonical, *, interrupted=False):
    """Exact presentation normalization measured at its canonical boundary."""
    prefix = 0 if interrupted else len(raw) - len(raw.lstrip())
    suffix = 0 if interrupted else len(raw) - len(raw.rstrip())
    # An all-whitespace completion is entirely prefix, not two overlapping
    # removals. Nonprintable no-reply output has no canonical text to anchor.
    prefix = min(prefix, len(raw))
    suffix = min(suffix, len(raw) - prefix)
    if raw[prefix:len(raw) - suffix if suffix else len(raw)] != canonical:
        return None
    utf16 = lambda value: len(value.encode("utf-16-le")) // 2
    return {"format": "omni-public-reply-text-binding", "version": 1,
        "normalization": "identity" if interrupted else "strip",
        "rawSha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "canonicalSha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "rawUtf16Length": utf16(raw), "canonicalUtf16Length": utf16(canonical),
        "removedPrefixUtf16": utf16(raw[:prefix]), "removedSuffixUtf16": utf16(raw[len(raw) - suffix:]) if suffix else 0}


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
    binding = public_reply_text_binding(text, result["text"], interrupted=interrupted)
    if binding is not None: result["publicTextBinding"] = binding
    return result


def mark_interrupted_turn(human, assistant, receipt, disposition="steered"):
    if disposition not in {"steered", "native-stop"}:
        raise ValueError("invalid generation disposition")
    if not human.get("input_accepted_before_reply"):
        human["generation_end"] = disposition
    assistant["generation_end"] = disposition
    receipt["generationEnd"] = disposition


def mark_no_reply_turn(human, assistant, receipt):
    """Bind a zero-text, completed turn without pretending an interruption."""
    if assistant.get("content") != "":
        raise ValueError("no-reply completion must have zero assistant text")
    if not human.get("input_accepted_before_reply"):
        human["generation_end"] = "no-reply"
    assistant["generation_end"] = "no-reply"
    receipt["generationEnd"] = "no-reply"


def validate_no_reply_turn(human, assistant, trace, receipt):
    """Require exact saved evidence; an empty row alone is not a completion."""
    markers = (
        (assistant.get("generation_end") if human.get("input_accepted_before_reply") is True
         and trace.get("input_accepted_before_reply") is True else human.get("generation_end")), assistant.get("generation_end"),
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
