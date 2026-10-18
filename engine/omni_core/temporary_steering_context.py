"""Current-request warm handoff input, never history lookup or learned reply."""
import hashlib
import re
from collections.abc import Mapping

from .offload import NeuralStateResourcePause


def temporary_steering_inputs(value, *, brain_id, turn_id, attention_epoch, policy):
    if value is None: return [], None
    required = {"format", "brainId", "successorTurnId", "attentionEpoch", "inputs"}
    if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - {"audioInputs"} or \
            value.get("format") != "omni-temporary-steering-input-v1" or value.get("brainId") != brain_id or \
            value.get("successorTurnId") != turn_id or type(value.get("attentionEpoch")) is not int or value["attentionEpoch"] < 0 or \
            not isinstance(value.get("inputs"), list) or not isinstance(value.get("audioInputs", []), list) or not (value["inputs"] or value.get("audioInputs")):
        raise ValueError("temporary steering input has invalid brain/turn/attention ownership")
    inputs = value["inputs"]
    charge = 131072 + len(inputs) * 1024 + len(value.get("audioInputs", [])) * 4096 + max((len(item.get("content", "")) * 4
        for item in inputs if isinstance(item, Mapping) and isinstance(item.get("content"), str)), default=0)
    status = policy.status(estimated_ram_bytes=charge)
    if status.get("memoryPressure"):
        raise NeuralStateResourcePause("temporary user-input handoff awaits admitted hash/metadata RAM", {**status,
            "paused": True, "recoverable": True, "stage": "temporary-steering-input-binding", "estimatedRamBytes": charge})
    seen = set()
    for item in inputs:
        if not isinstance(item, Mapping) or set(item) != {"turnId", "content", "inputSha256"} or \
                not isinstance(item.get("turnId"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item["turnId"]) or \
                item["turnId"] == turn_id or item["turnId"] in seen or not isinstance(item.get("content"), str) or \
                not item["content"] or item["content"].replace("\x00", "").strip() != item["content"] or \
                not isinstance(item.get("inputSha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["inputSha256"]) or \
                hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != item["inputSha256"]:
            raise ValueError("temporary steering input lost its exact human identity/hash")
        seen.add(item["turnId"])
    audio_inputs = []
    if value.get("audioInputs"):
        from .native_audio_chat import validate_native_audio_input
        for item in value["audioInputs"]:
            audio = validate_native_audio_input(item, brain_id=brain_id)
            owner = audio["attachment"]["turnId"]
            if owner == turn_id or owner in seen:
                raise ValueError("temporary audio handoff repeats or replaces its original owner")
            seen.add(owner); audio_inputs.append(audio)
    audit = {"format": value["format"], "attentionEpoch": value["attentionEpoch"],
        "inputs": [{"turnId": item["turnId"], "inputSha256": item["inputSha256"]} for item in inputs],
        "freshAttentionDiscarded": value["attentionEpoch"] != attention_epoch,
        "durablePredecessorLearning": False, "longTermHistoryRead": False}
    if audio_inputs:
        audit["audioInputs"] = [{"turnId": item["attachment"]["turnId"], "inputSha256": item["inputSha256"]} for item in audio_inputs]
    # Fresh is an explicit ownership boundary. Even an already queued carry
    # cannot reintroduce its retired raw task input into the next native turn.
    return ([] if audit["freshAttentionDiscarded"] else [item["content"] for item in inputs]), audit


def prompt_with_temporary_user_inputs(tokenizer, human, inputs, recent_tokens, *, capacity, policy):
    """Real role-token segments, not fabricated prose or a behavior prompt."""
    estimated = 131072 + (len(human) + sum(len(text) for text in inputs)) * 4 * 48 + len(inputs) * 1024
    status = policy.status(estimated_ram_bytes=estimated)
    if status.get("memoryPressure"):
        raise NeuralStateResourcePause("unfinished user requests and successor need admitted temporary token RAM", {**status,
            "paused": True, "recoverable": True, "stage": "temporary-steering-prompt", "estimatedRamBytes": estimated})
    current = []
    for text in (*inputs, human):
        current.append(tokenizer.human_id)
        current.extend(tokenizer.encode(text))
    current.append(tokenizer.brain_id)
    required = 1 + len(current)
    if required > capacity:
        raise NeuralStateResourcePause("complete unfinished user input and successor exceed the selected native context", {
            "paused": True, "recoverable": True, "stage": "temporary-steering-context-capacity",
            "requiredTokens": required, "selectedContextTokens": capacity, "inputTruncated": False,
            "temporaryInputDiscarded": False})
    history_budget = capacity - required
    history = list(recent_tokens[-history_budget:]) if history_budget else []
    while history and history[0] not in {tokenizer.human_id, tokenizer.brain_id}: history.pop(0)
    return [tokenizer.bos_id, *history, *current], history
