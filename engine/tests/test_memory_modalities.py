import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig
from omni_core.modalities import ModalityHub
from omni_core.vsa import ConceptMemory, HypervectorSpace


class MemoryAndModalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_vsa_binding_unbinding_and_similar_recall(self):
        space = HypervectorSpace(128, seed=2)
        left = space.symbol("left")
        right = space.symbol("right")
        bound = space.bind(left, right)
        recovered = space.bind(bound, right)
        self.assertGreater(space.similarity(left, recovered), 0.99)
        sequenced = space.permute(left, steps=7)
        unsequenced = space.inverse_permute(sequenced, steps=7)
        self.assertTrue(torch.equal(left, unsequenced))
        self.assertLess(space.similarity(left, sequenced), 0.5)

        memory = ConceptMemory(128, seed=2, max_relations=4)
        learned = memory.learn(
            "liquid neurons adapt through continuous time",
            retain_source_text=False,
        )
        cue = memory.vector_for_text("liquid neurons")
        _, recalled = memory.recall_vector(cue)
        self.assertTrue(recalled)
        self.assertNotIn("source_text", memory.ideas[0])
        self.assertIn(learned["idea_id"], memory.idea_vectors)

    def test_relation_capacity_is_enforced(self):
        memory = ConceptMemory(64, max_relations=2)
        memory.learn("alpha beta gamma delta epsilon")
        self.assertLessEqual(len(memory.relations), 2)

    def test_all_modality_baselines_forward_generate_and_backpropagate(self):
        config = OmniConfig.micro()
        hub = ModalityHub(config)
        idea = torch.randn(1, config.idea_dim)

        image = torch.randn(1, 3, config.image_size, config.image_size).clamp(-1, 1)
        image_result = hub.image(image, idea)
        self.assertEqual(tuple(image_result["reconstruction"].shape), tuple(image.shape))
        self.assertGreater(float(image_result["diffusion_loss"].item()), 0.0)

        audio = torch.randn(1, 1, config.audio_samples).clamp(-1, 1)
        audio_result = hub.audio(audio, idea)
        self.assertEqual(tuple(audio_result["reconstruction"].shape), tuple(audio.shape))

        video = torch.randn(
            1,
            3,
            config.video_frames,
            config.image_size,
            config.image_size,
        ).clamp(-1, 1)
        video_result = hub.video(video, idea)
        self.assertEqual(tuple(video_result["reconstruction"].shape), tuple(video.shape))
        self.assertGreater(float(video_result["diffusion_loss"].item()), 0.0)

        loss = (
            image_result["loss"] + audio_result["loss"] + video_result["loss"]
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in hub.parameters()
            )
        )
        for kind, expected in (
            ("image", (1, 3, config.image_size, config.image_size)),
            ("audio", (1, config.audio_samples)),
            (
                "video",
                (
                    1,
                    3,
                    config.video_frames,
                    config.image_size,
                    config.image_size,
                ),
            ),
        ):
            generated = hub.generate(kind, idea, seed=1)
            self.assertEqual(tuple(generated.shape), expected)
            self.assertTrue(torch.isfinite(generated).all())

    def test_each_modality_overfits_a_fixture_and_safe_reload_is_exact(self):
        config = OmniConfig.micro(learning_rate=0.01)
        hub = ModalityHub(config)
        idea = torch.randn(1, config.idea_dim)
        fixtures = {
            "image": torch.linspace(
                -1, 1, 3 * config.image_size * config.image_size
            ).reshape(1, 3, config.image_size, config.image_size),
            "audio": torch.sin(
                torch.linspace(0, 8, config.audio_samples)
            ).reshape(1, 1, config.audio_samples),
            "video": torch.linspace(
                -1,
                1,
                3
                * config.video_frames
                * config.image_size
                * config.image_size,
            ).reshape(
                1,
                3,
                config.video_frames,
                config.image_size,
                config.image_size,
            ),
        }
        for name, target in fixtures.items():
            module = getattr(hub, name)
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
            with torch.no_grad():
                initial = float(module(target, idea)["loss"].item())
            for _ in range(18):
                optimizer.zero_grad(set_to_none=True)
                loss = module(target, idea)["loss"]
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final = float(module(target, idea)["loss"].item())
            self.assertLess(final, initial, msg=name)

        with tempfile.TemporaryDirectory(prefix="omni-modalities-") as folder:
            path = Path(folder) / "modalities.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in hub.state_dict().items()
                },
                str(path),
            )
            reloaded = ModalityHub(config)
            reloaded.load_state_dict(load_file(str(path)))
            for name in ("image", "audio", "video"):
                expected = hub.generate(name, idea, seed=91)
                actual = reloaded.generate(name, idea, seed=91)
                self.assertTrue(torch.equal(expected, actual), msg=name)


if __name__ == "__main__":
    unittest.main()
