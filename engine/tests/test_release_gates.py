import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


def write_text_pdf(path: Path, text: str) -> None:
    """Write a minimal, valid PDF with extractable Helvetica text."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/Resources")] = resources
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(
        ("BT /F1 12 Tf 72 720 Td (%s) Tj ET" % escaped).encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(404)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-release-gates-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, brain_id: str, **overrides) -> AdaptiveBrain:
        config = OmniConfig.micro(
            name=brain_id,
            parallel_thoughts=1,
            max_seq_len=40,
            learn_from_own_messages=False,
            **overrides,
        )
        return AdaptiveBrain.create(
            brain_id, self.root / brain_id, config
        )

    def test_real_pdf_text_ingestion_changes_parameters_and_concepts(self):
        brain = self.make_brain("pdf-brain")
        pdf = self.root / "knowledge.pdf"
        write_text_pdf(
            pdf,
            "PdfSentinel neurons remember compositional cobalt geometry.",
        )
        before = brain.parameter_checksum()
        result = brain.ingest(path=str(pdf), kind="pdf", policy="encode")
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        self.assertGreater(result["source"]["learned_ideas"], 0)
        labels = {node["label"] for node in brain.memory.concepts.values()}
        self.assertIn("pdfsentinel", labels)
        self.assertFalse(result["source"]["raw_text_retained"])
        brain.events.close()

    def test_real_png_ingestion_trains_image_and_vision_packs(self):
        brain = self.make_brain("png-brain")
        png = self.root / "fixture.png"
        image = Image.new("RGB", (12, 10), (12, 80, 220))
        for x in range(2, 10):
            image.putpixel((x, x % 10), (240, 40, 20))
        image.save(png, format="PNG")
        before = brain.parameter_checksum()
        result = brain.ingest(path=str(png), kind="image", policy="encode")
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        vision = brain.generate_modality("vision", input_path=str(png), seed=9)
        self.assertEqual(len(vision["embedding"]), brain.config.idea_dim)
        brain.events.close()

    def test_real_animated_gif_ingestion_trains_video_pack(self):
        brain = self.make_brain("gif-brain")
        gif = self.root / "motion.gif"
        frames = [
            Image.new("RGB", (10, 10), color)
            for color in ((240, 20, 20), (20, 240, 20), (20, 20, 240))
        ]
        frames[0].save(
            gif,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=40,
            loop=0,
        )
        before = brain.parameter_checksum()
        result = brain.ingest(path=str(gif), kind="video", policy="encode")
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        brain.events.close()

    @unittest.skipUnless(
        importlib.util.find_spec("soundfile") is not None,
        "soundfile is installed from engine requirements in release CI",
    )
    def test_flac_ingestion_trains_audio_pack(self):
        import numpy as np
        import soundfile

        brain = self.make_brain("flac-brain")
        flac = self.root / "tone.flac"
        samples = np.sin(np.linspace(0, 8 * np.pi, 800)).astype("float32")
        soundfile.write(str(flac), samples, 8000, format="FLAC")
        result = brain.ingest(path=str(flac), kind="audio", policy="encode")
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertGreater(brain.modality_training["audio"], 0)
        brain.events.close()

    @unittest.skipUnless(
        importlib.util.find_spec("imageio_ffmpeg") is not None,
        "imageio-ffmpeg is installed from engine requirements in release CI",
    )
    def test_mp4_ingestion_trains_video_pack(self):
        import numpy as np
        import imageio_ffmpeg

        brain = self.make_brain("mp4-brain")
        mp4 = self.root / "motion.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(mp4),
            (16, 16),
            fps=4,
            codec="libx264",
            pix_fmt_in="rgb24",
            output_params=["-pix_fmt", "yuv420p"],
        )
        writer.send(None)
        try:
            for index in range(4):
                frame = np.zeros((16, 16, 3), dtype="uint8")
                frame[:, :, index % 3] = 220
                frame[index : index + 4, :, :] = 80
                writer.send(frame.tobytes())
        finally:
            writer.close()
        result = brain.ingest(path=str(mp4), kind="video", policy="encode")
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertGreater(brain.modality_training["video"], 0)
        brain.events.close()

    def test_memory_recipes_have_distinct_source_retention(self):
        text = "RetentionSentinel links amber concepts into one durable idea."
        cases = (
            ("human", "human-consolidation", True, False),
            ("total", "total-recall", True, True),
            ("synapses", "synapses-only", True, False),
        )
        for brain_id, recipe, retain_flag, expected_raw in cases:
            with self.subTest(recipe=recipe):
                brain = self.make_brain(
                    brain_id,
                    memory_recipe=recipe,
                    retain_source_text=retain_flag,
                )
                result = brain.ingest(
                    text=text, name=brain_id + ".txt", policy="encode"
                )
                source = result["source"]
                self.assertEqual(source["raw_text_retained"], expected_raw)
                self.assertEqual("raw_text" in source, expected_raw)
                idea_has_text = any(
                    idea.get("source_text") == text for idea in brain.memory.ideas
                )
                self.assertEqual(idea_has_text, expected_raw)
                metadata = (brain.engine_path / "brain.json").read_text("utf-8")
                self.assertEqual(text in metadata, expected_raw)
                brain.events.close()

    def test_expert_growth_is_persisted_and_reloadable(self):
        brain = self.make_brain(
            "growth-brain",
            growth_policy="elastic",
            max_experts=2,
            growth_novelty_threshold=0.0,
            growth_patience=1,
        )
        self.assertEqual(brain.decoder.expert_count, 0)
        learned = brain.learn_experience(
            "A completely novel expert-growth experience.", importance=0.9
        )
        self.assertTrue(learned["grew_expert"])
        self.assertEqual(brain.decoder.expert_count, 1)
        brain.save()
        brain.events.close()
        reloaded = AdaptiveBrain.load(
            self.root / "growth-brain", "growth-brain"
        )
        self.assertEqual(reloaded.decoder.expert_count, 1)
        self.assertEqual(
            reloaded.runtime_card()["growth"]["experts"], 1
        )
        reloaded.events.close()

    def test_replay_threshold_decay_and_seeded_trace_are_deterministic(self):
        brain = self.make_brain(
            "memory-brain",
            long_term_threshold=0.8,
            forgetting_rate=0.25,
        )
        brain.learn_experience("A low salience transient.", importance=0.2)
        self.assertEqual(len(brain.replay), 0)
        brain.learn_experience(
            "A durable high salience memory.", importance=0.95
        )
        self.assertEqual(len(brain.replay), 1)
        concept_id = next(iter(brain.memory.concepts))
        brain.memory.concepts[concept_id]["activation"] = 1.0
        result = brain.consolidate(steps=1)
        self.assertTrue(result["promoted"])
        self.assertLess(brain.memory.concepts[concept_id]["activation"], 1.0)
        brain.events.close()

        first = self.make_brain("deterministic-a")
        second = self.make_brain("deterministic-b")
        torch.manual_seed(991)
        left = first.chat("Trace the same signal.", max_new_tokens=5, seed=77)
        torch.manual_seed(991)
        right = second.chat("Trace the same signal.", max_new_tokens=5, seed=77)
        self.assertEqual(left["text"], right["text"])
        self.assertEqual(
            left["trace"]["selected_branch"],
            right["trace"]["selected_branch"],
        )
        self.assertEqual(left["trace"]["branches"], right["trace"]["branches"])
        self.assertEqual(
            left["trace"]["ponder_factors"],
            right["trace"]["ponder_factors"],
        )
        first.events.close()
        second.events.close()


if __name__ == "__main__":
    unittest.main()
