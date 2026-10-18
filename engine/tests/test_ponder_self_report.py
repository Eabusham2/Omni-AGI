import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("owned_ponder_self_report", Path(__file__).resolve().parents[1] / "omni_core/ponder_self_report.py")
PONDER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(PONDER)
PRIVATE_CANDIDATE = PONDER.PRIVATE_CANDIDATE
decode_ponder_self_report = PONDER.decode_ponder_self_report
emit_registered_action = PONDER.emit_registered_action
finalize_registered_action = PONDER.finalize_registered_action
publication_frames = PONDER.publication_frames
validate_publication_frame = PONDER.validate_publication_frame
PonderPublicationTracker = PONDER.PonderPublicationTracker


class FakeHead:
    def __init__(self, arguments, *, grounded=1):
        self.arguments = arguments
        self.grounded_steps = torch.tensor(grounded)
        self.calls = []

    @staticmethod
    def schema_features(tool_id, action, schema):
        assert (tool_id, action) == ("brain.ponder", "self-report")
        assert schema["actionInputSchemas"][action]["required"] == ["emit"]
        return torch.zeros(64)

    def decode(self, neural_state, features, **options):
        self.calls.append((neural_state.clone(), features.clone(), dict(options)))
        return {"arguments": self.arguments, "meanTokenProbability": 0.99}


class FakeDecoder:
    def __init__(self, head):
        self.action_argument_head = head


def candidate(text="I can share a tentative summary of what changed."):
    head = FakeHead({"emit": True, "text": text})
    value = decode_ponder_self_report(
        FakeDecoder(head), torch.zeros((1, 8)), phase="pre-speech",
        max_output_bytes=100_000,
    )
    assert value is not None
    assert head.calls[0][2]["allow_declared_training"] is True
    assert head.calls[0][2]["minimum_mean_probability"] == pytest.approx(0.90)
    return value


def test_grounded_model_can_choose_emit_or_silence_without_canned_text():
    value = candidate("my own exact report")
    assert value["text"] == "my own exact report"
    assert value["hiddenPromptUsed"] is False
    assert value["selfReportNotChainOfThought"] is True
    assert decode_ponder_self_report(
        FakeDecoder(FakeHead({"emit": False}, grounded=1)),
        torch.zeros((1, 8)), phase="pre-speech", max_output_bytes=100,
    ) is None
    assert decode_ponder_self_report(
        FakeDecoder(FakeHead({"emit": True, "text": "not grounded"}, grounded=0)),
        torch.zeros((1, 8)), phase="pre-speech", max_output_bytes=100,
    ) is None


def test_publication_is_utf8_safe_ordered_and_finally_hash_bound():
    text = "é" * 10_000
    action = {"kind": "ponder", "actionId": "a" * 32, "arguments": {PRIVATE_CANDIDATE: candidate(text)}}
    emitted = []
    emit_registered_action(lambda kind, value: emitted.append((kind, value)), action)
    assert emitted[0][0] == "action"
    frames = [validate_publication_frame(value) for kind, value in emitted[1:] if kind == "ponder-self-report"]
    assert len(frames) > 1
    assert "".join(frame["delta"] for frame in frames) == text
    assert [frame["chunkIndex"] for frame in frames] == list(range(len(frames)))
    assert frames[-1]["textSha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert frames[-1]["totalBytes"] == len(text.encode())
    assert action["arguments"]["ponderSelfReport"]["state"] == "complete"
    assert PRIVATE_CANDIDATE not in action["arguments"]


def test_stopped_publication_exposes_only_frames_accepted_before_stop():
    action = {"kind": "ponder", "actionId": "b" * 32, "arguments": {PRIVATE_CANDIDATE: candidate("z" * 40_000)}}
    emitted = []

    def stop_after_first(kind, value):
        emitted.append((kind, value))
        if kind == "ponder-self-report":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        emit_registered_action(stop_after_first, action)
    frames = [value for kind, value in emitted if kind == "ponder-self-report"]
    assert len(frames) == 1
    assert frames[0]["publicationComplete"] is False
    assert "ponderSelfReport" not in action["arguments"]
    assert PRIVATE_CANDIDATE in action["arguments"]


def test_nonstream_finalization_retains_the_actual_model_report():
    action = {"kind": "ponder", "actionId": "c" * 32, "arguments": {PRIVATE_CANDIDATE: candidate("saved report")}}
    report = finalize_registered_action(action)
    assert report is not None
    assert report["text"] == "saved report"
    assert report["textSha256"] == hashlib.sha256(b"saved report").hexdigest()
    assert list(publication_frames(report))[-1]["publicationComplete"] is True


def test_worker_tracker_requires_preceding_action_exact_order_and_only_one_report():
    action = {"kind": "ponder", "actionId": "d" * 32}
    report = finalize_registered_action({"kind": "ponder", "actionId": "d" * 32,
        "arguments": {PRIVATE_CANDIDATE: candidate("tracked report")}})
    frames = list(publication_frames(report))
    tracker = PonderPublicationTracker()
    with pytest.raises(ValueError, match="preceding"):
        tracker.accept(frames[0], [])
    assert tracker.accept(frames[0], [action])["publicationComplete"] is True
    with pytest.raises(ValueError, match="order or identity"):
        tracker.accept(frames[0], [action])
