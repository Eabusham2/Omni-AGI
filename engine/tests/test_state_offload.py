import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.offload import (
    GIB,
    MIB,
    HotStateResidencyPlanner,
    DurableReplayBuffer,
    NeuralStateResourcePause,
    ResourcePolicy,
    ResourceReading,
    _apply_memory_limit,
    _mac_memory_pressure_available,
    _mac_process_rusage,
    _parse_mac_process_footprint,
    copy_mutable_state_snapshot,
)
from omni_core.persistence import (
    copy_substrate_snapshot,
)


def optimizer_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value.detach().cpu()
    elif isinstance(value, dict):
        for item in value.values():
            yield from optimizer_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from optimizer_tensors(item)


class DurableStateOffloadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-state-offload-"
        )
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, **overrides):
        return AdaptiveBrain.create(
            "offload-brain",
            self.root,
            OmniConfig.micro(
                max_seq_len=40,
                learn_from_own_messages=False,
                **overrides,
            ),
        )

    def test_mac_memory_pressure_reclaimability_is_bounded(self):
        self.assertEqual(
            _mac_memory_pressure_available(
                "System-wide memory free percentage: 33%", 16 * GIB
            ),
            int(16 * GIB * 0.33),
        )
        self.assertIsNone(
            _mac_memory_pressure_available("unavailable", 16 * GIB)
        )
        self.assertIsNone(
            _mac_memory_pressure_available(
                "System-wide memory free percentage: 101%", 16 * GIB
            )
        )

    def test_mac_physical_footprint_tracks_activity_monitor_and_peak(self):
        current, peak = _parse_mac_process_footprint(
            """
            Auxiliary data:
                phys_footprint: 2147483648 B
                phys_footprint_peak: 6442450944 B
            """
        )
        self.assertEqual(current, 2 * GIB)
        self.assertEqual(peak, 6 * GIB)
        self.assertEqual(
            _parse_mac_process_footprint("unavailable"), (None, None)
        )

    @unittest.skipUnless(sys.platform == "darwin", "macOS libproc only")
    def test_mac_libproc_physical_footprint_is_live_and_bounded(self):
        current, peak = _mac_process_rusage()
        self.assertIsInstance(current, int)
        self.assertIsInstance(peak, int)
        self.assertGreater(current or 0, 0)
        self.assertGreaterEqual(peak or 0, current or 0)

    def test_resource_status_exposes_physical_memory_high_water(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=8 * GIB,
            process_memory_bytes=2 * GIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
            process_peak_memory_bytes=6 * GIB,
        )
        status = ResourcePolicy(
            self.root,
            reading_provider=lambda: reading,
        ).status()
        self.assertEqual(status["processMemoryBytes"], 2 * GIB)
        self.assertEqual(status["processPeakMemoryBytes"], 6 * GIB)

    def test_replay_never_thins_at_the_old_capacity_and_reloads_without_loss(self):
        brain = self.make_brain()
        expected = []
        # A tiny legacy-looking capacity in a loaded configuration must not
        # become a replay cardinality ceiling in stable v1.
        for index in range(17):
            value = torch.linspace(
                float(index), float(index + 1), brain.config.idea_dim
            )
            expected.append(value)
            brain._append_replay(value, importance=1.0)
        brain.save()
        self.assertEqual(len(brain.replay), len(expected))
        self.assertNotIn(
            "state.replay",
            load_file(str(brain.engine_path / "plasticity.safetensors")),
        )
        self.assertTrue(
            (brain.engine_path / "state" / "replay.sqlite3").is_file()
        )
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "offload-brain")
        self.assertEqual(len(reloaded.replay), len(expected))
        for actual, wanted in zip(reloaded.replay, expected):
            self.assertTrue(torch.equal(actual, wanted))
        card = reloaded.runtime_card()["state_offload"]
        self.assertEqual(card["replay"]["count"], len(expected))
        self.assertFalse(card["replay"]["silentEviction"])
        reloaded.events.close()

    def test_optimizer_moments_are_safe_tensor_checkpointed_and_restored(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Durable optimizer moments preserve slowly learned neural state.",
            steps=1,
            importance=1.0,
        )
        before = copy.deepcopy(brain._optimizer.state_dict())
        self.assertTrue(before["state"])
        brain.save()
        metadata = json.loads(
            (brain.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertEqual(
            metadata["mutable_state"]["format"], "omni-mutable-state"
        )
        self.assertFalse(
            list((brain.engine_path / "state").rglob("*.pt"))
        )
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "offload-brain")
        after = reloaded._optimizer.state_dict()
        self.assertEqual(before["param_groups"], after["param_groups"])
        before_tensors = list(optimizer_tensors(before))
        after_tensors = list(optimizer_tensors(after))
        self.assertEqual(len(before_tensors), len(after_tensors))
        self.assertTrue(
            all(
                torch.equal(expected, actual)
                for expected, actual in zip(before_tensors, after_tensors)
            )
        )
        reloaded.events.close()

    def test_memory_pressure_spills_optimizer_and_activation_scratch_then_rehydrates(self):
        brain = self.make_brain()
        brain.learn_experience(
            "A pressure spill must be exact and resumable.",
            steps=1,
            importance=1.0,
        )
        expected = copy.deepcopy(brain._optimizer.state_dict())
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=64 * 1024 * 1024,
            process_memory_bytes=4 * GIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(
            brain.engine_path,
            reading_provider=lambda: reading,
        )
        brain.resource_policy = policy
        brain.state_store.policy = policy
        brain.replay.policy = policy

        status = brain._maintain_neural_state_resources()
        self.assertTrue(brain._optimizer_offloaded)
        self.assertFalse(brain._optimizer.state)
        self.assertFalse(status["optimizer"]["resident"])
        self.assertTrue((brain.engine_path / "state" / "scratch.json").is_file())
        scratch = json.loads(
            (brain.engine_path / "state" / "scratch.json").read_text("utf-8")
        )
        scratch_manifest = brain.engine_path / "state" / scratch["manifest"]
        scratch_value = json.loads(scratch_manifest.read_text("utf-8"))
        self.assertIn("activations", scratch_value)

        brain._ensure_optimizer_resident()
        actual = brain._optimizer.state_dict()
        expected_tensors = list(optimizer_tensors(expected))
        actual_tensors = list(optimizer_tensors(actual))
        self.assertTrue(
            all(
                torch.equal(wanted, value)
                for wanted, value in zip(expected_tensors, actual_tensors)
            )
        )
        brain.events.close()

    def test_low_disk_pauses_before_replay_mutation(self):
        brain = self.make_brain()
        before = len(brain.replay)
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=8 * GIB,
            process_memory_bytes=1 * GIB,
            disk_total_bytes=100 * GIB,
            disk_free_bytes=3 * GIB,
        )
        policy = ResourcePolicy(
            brain.engine_path,
            disk_reserve_bytes=4 * GIB,
            reading_provider=lambda: reading,
        )
        brain.resource_policy = policy
        brain.state_store.policy = policy
        brain.replay.policy = policy
        with self.assertRaises(NeuralStateResourcePause):
            brain._append_replay(
                torch.ones(brain.config.idea_dim), importance=1.0
            )
        self.assertEqual(len(brain.replay), before)
        self.assertIsNotNone(brain.resource_pause)
        self.assertTrue(brain.runtime_card()["state_offload"]["paused"])
        brain.events.close()

    def test_disk_reserve_never_falls_below_twenty_gib(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=8 * GIB,
            process_memory_bytes=1 * GIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=25 * GIB,
        )
        policy = ResourcePolicy(
            self.root,
            disk_reserve_bytes=2 * GIB,
            reading_provider=lambda: reading,
        )
        status = policy.status()
        self.assertEqual(status["diskReserveBytes"], 20 * GIB)
        self.assertEqual(status["mandatoryFreeDiskBytes"], 20 * GIB)

    def test_system_ram_envelope_counts_existing_process_memory_without_collapsing(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=6 * GIB,
            process_memory_bytes=2 * GIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(
            self.root,
            system_ram_share_percent=50,
            reading_provider=lambda: reading,
        )
        status = policy.status()
        self.assertEqual(status["safeRamPoolBytes"], 15 * GIB)
        self.assertEqual(status["systemRamBudgetBytes"], int(7.5 * GIB))
        self.assertFalse(status["memoryPressure"])
        projected = policy.status(estimated_ram_bytes=6 * GIB)
        self.assertTrue(projected["memoryPressure"])
        self.assertEqual(projected["projectedProcessMemoryBytes"], 8 * GIB)
        self.assertEqual(
            projected["systemRamBudgetBytes"], status["systemRamBudgetBytes"]
        )
        self.assertTrue(projected["capacityPersistsAcrossPressure"])
        self.assertFalse(projected["contextPagedToStorage"])

    def test_packed_update_scratch_is_not_charged_as_adam_moments(self):
        reading = ResourceReading(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=5 * GIB,
            process_memory_bytes=256 * MIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(self.root, reading_provider=lambda: reading)
        plan = policy.training_plan(
            max_window_tokens=64,
            requested_batch_size=1,
            requested_gradient_accumulation=1,
            trainable_parameter_bytes=0,
            packed_update_scratch_bytes=32 * MIB,
            activation_bytes_per_token=MIB,
        )
        self.assertEqual(plan["memory"]["optimizerAndGradientBytes"], 0)
        self.assertEqual(plan["memory"]["packedUpdateScratchBytes"], 32 * MIB)

    def test_slow_drive_biases_auto_toward_ram_and_rate_limits_scratch(self):
        reading = ResourceReading(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=5 * GIB,
            process_memory_bytes=512 * 1024 * 1024,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(
            self.root,
            storage_bytes_per_second=40 * 1024 * 1024,
            hardware_tier="micro",
            reading_provider=lambda: reading,
        )
        plan = policy.training_plan(
            max_window_tokens=1024,
            requested_batch_size=4,
            requested_gradient_accumulation=2,
            trainable_parameter_bytes=256 * 1024 * 1024,
            activation_bytes_per_token=4 * 1024 * 1024,
            storage_bytes_per_second=40 * 1024 * 1024,
        )
        self.assertEqual(plan["memory"]["systemRamSharePercent"], 65.0)
        self.assertLess(plan["physicalBatchRecords"], 4)
        self.assertLess(plan["windowTokens"], 1024)
        self.assertEqual(plan["scratch"]["storageClass"], "slow-storage")
        self.assertEqual(plan["scratch"]["minimumWriteIntervalSeconds"], 3600)
        self.assertFalse(plan["scratch"]["usedAsVirtualRam"])
        self.assertTrue(plan["allSourceBytesVisited"])

    def test_training_plan_chooses_largest_safe_logical_batch_divisor(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=12 * GIB,
            process_memory_bytes=256 * MIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(
            self.root,
            reading_provider=lambda: reading,
        )
        plan = policy.training_plan(
            max_window_tokens=64,
            requested_batch_size=4,
            requested_gradient_accumulation=4,
            effective_batch_target=16,
            require_physical_batch_divisor=True,
            trainable_parameter_bytes=0,
            activation_bytes_per_token=MIB,
            resource_mode="manual",
            manual_ram_budget_bytes=350 * MIB,
        )

        # The raw headroom fits three samples. Three does not divide the
        # frozen logical target, so the planner selects two and restores eight
        # accumulation rounds instead of silently expanding 16 records to 18.
        self.assertEqual(plan["physicalBatchRecords"], 2)
        self.assertEqual(plan["gradientAccumulation"], 8)
        self.assertEqual(plan["effectiveBatchTarget"], 16)
        self.assertEqual(
            plan["physicalBatchRecords"] * plan["gradientAccumulation"],
            16,
        )

    def test_constrained_memory_limits_and_reserves_match_desktop_preflight(self):
        self.assertEqual(
            _apply_memory_limit(64 * GIB, 40 * GIB, 4 * GIB, int(1.5 * GIB)),
            (4 * GIB, int(2.5 * GIB)),
        )
        self.assertEqual(ResourcePolicy._adaptive_ram_reserve(2 * GIB), 192 * MIB)
        self.assertEqual(ResourcePolicy._adaptive_ram_reserve(4 * GIB), 384 * MIB)
        self.assertEqual(ResourcePolicy._adaptive_ram_reserve(8 * GIB), 512 * MIB)
        self.assertEqual(ResourcePolicy._adaptive_ram_reserve(16 * GIB), GIB)
        self.assertEqual(ResourcePolicy._adaptive_ram_reserve(32 * GIB), 2 * GIB)

    def test_manual_ram_share_is_a_ceiling_inside_live_safety(self):
        reading = ResourceReading(
            total_memory_bytes=32 * GIB,
            available_memory_bytes=24 * GIB,
            process_memory_bytes=2 * GIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        policy = ResourcePolicy(
            self.root,
            system_ram_share_percent=30,
            reading_provider=lambda: reading,
        )
        status = policy.status()
        self.assertEqual(status["systemRamMode"], "manual")
        self.assertEqual(status["systemRamSharePercent"], 30.0)
        self.assertLess(status["systemRamBudgetBytes"], status["safeRamPoolBytes"])

    def test_auto_ram_base_follows_hardware_tier_not_total_ram_bucket(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=13 * GIB,
            process_memory_bytes=256 * MIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        for tier, expected in {
            "micro": 55.0,
            "personal": 65.0,
            "gpu": 75.0,
            "workstation": 80.0,
        }.items():
            with self.subTest(tier=tier):
                status = ResourcePolicy(
                    self.root,
                    hardware_tier=tier,
                    storage_bytes_per_second=700 * MIB,
                    reading_provider=lambda: reading,
                ).status()
                self.assertEqual(status["systemRamMode"], "auto")
                self.assertEqual(status["hardwareTier"], tier)
                self.assertEqual(status["systemRamSharePercent"], expected)

    def test_auto_expands_for_safe_model_allocation_but_manual_cap_blocks(self):
        reading = ResourceReading(
            total_memory_bytes=16 * GIB,
            available_memory_bytes=13 * GIB,
            process_memory_bytes=256 * MIB,
            disk_total_bytes=500 * GIB,
            disk_free_bytes=300 * GIB,
        )
        estimate = 10 * GIB
        automatic = ResourcePolicy(
            self.root,
            reading_provider=lambda: reading,
        ).status(estimated_ram_bytes=estimate)
        manual = ResourcePolicy(
            self.root,
            system_ram_share_percent=65,
            reading_provider=lambda: reading,
        ).status(estimated_ram_bytes=estimate)

        self.assertEqual(automatic["systemRamMode"], "auto")
        self.assertGreater(automatic["systemRamSharePercent"], 65.0)
        self.assertLessEqual(automatic["systemRamSharePercent"], 90.0)
        self.assertFalse(automatic["memoryPressure"])
        self.assertEqual(manual["systemRamSharePercent"], 65.0)
        self.assertTrue(manual["memoryPressure"])

    def test_transient_pressure_wait_preserves_saved_cortex_capacity(self):
        # Resource-admission unit fixture: no origin or learned capability is
        # exercised, so a real Build curriculum is unnecessary here.
        brain = AdaptiveBrain(
            "offload-brain",
            self.root,
            OmniConfig.micro(
                max_seq_len=4096,
                learn_from_own_messages=False,
            ),
        )
        configured = brain.config.max_seq_len
        policy = ResourcePolicy(
            brain.engine_path,
            system_ram_share_percent=70,
            reading_provider=lambda: ResourceReading(
                total_memory_bytes=16 * GIB,
                available_memory_bytes=64 * MIB,
                process_memory_bytes=4 * GIB,
                disk_total_bytes=500 * GIB,
                disk_free_bytes=300 * GIB,
            ),
        )
        brain.resource_policy = policy
        brain.state_store.policy = policy
        brain.replay.policy = policy
        brain.paged_working_memory.policy = policy

        status = brain._maintain_neural_state_resources()

        self.assertEqual(brain.config.max_seq_len, configured)
        self.assertEqual(
            brain.decoder.blocks[0].attention.rotary.max_seq_len,
            configured,
        )
        self.assertTrue(status["paused"])
        self.assertTrue(status["resources"]["waitForMemory"])
        self.assertGreater(status["resources"]["retryAfterSeconds"], 0)
        self.assertFalse(status["resources"]["contextWindowShrunk"])
        self.assertTrue(status["resources"]["activeCortexResident"])
        brain.events.close()

    def test_hot_residency_prioritizes_firing_rooted_and_unfinished_state(self):
        planner = HotStateResidencyPlanner()
        status = planner.update(
            neurons={
                "firing": {"activation": 1.0, "exposures": 2},
                "cold": {"activation": 0.0, "exposures": 0},
            },
            assemblies=[
                {"id": "rooted", "importance": 0.8, "source": "starter"},
                {"id": "unfinished", "importance": 0.0},
            ],
            synapses={
                "frequent": {"uses": 100, "stability": 0.8},
                "unused": {"uses": 0, "stability": 0.0},
            },
            unfinished_ids=["unfinished"],
            resident_budget=4,
        )
        self.assertIn("firing", planner.hot_ids)
        self.assertIn("rooted", planner.hot_ids)
        self.assertIn("unfinished", planner.hot_ids)
        self.assertIn("frequent", planner.hot_ids)
        self.assertNotIn("cold", planner.hot_ids)
        self.assertEqual(status["hotUnfinishedEntities"], 1)
        self.assertTrue(status["updatedContinuously"])
        self.assertTrue(status["dynamicTransitions"])
        self.assertFalse(status["physicalSubstratePaging"])
        self.assertTrue(status["noCardinalityLimit"])

        planner.note_access(["cold"])
        planner.note_access(["cold"])
        changed = planner.update(
            neurons={
                "firing": {"activation": 0.0, "exposures": 2},
                "cold": {"activation": 0.0, "exposures": 0},
            },
            assemblies=[
                {"id": "rooted", "importance": 0.8, "source": "starter"},
                {"id": "unfinished", "importance": 0.0},
            ],
            synapses={
                "frequent": {"uses": 100, "stability": 0.8},
                "unused": {"uses": 0, "stability": 0.0},
            },
            unfinished_ids=["unfinished"],
            resident_budget=4,
            paged_assembly_ids=["cold"],
        )
        self.assertIn("cold", planner.hot_ids)
        self.assertIn("firing", planner.became_cold_ids)
        self.assertIn("cold", planner.page_in_candidate_ids)
        self.assertGreater(changed["becameColdEntities"], 0)
        self.assertEqual(changed["pageInCandidates"], 1)

    def test_working_patterns_page_to_storage_and_survive_restart(self):
        brain = self.make_brain(
            working_memory_slots=5,
            memory_resident_items=2,
        )
        for index in range(7):
            vector = torch.zeros(brain.config.idea_dim, dtype=torch.float32)
            vector[index] = 1.0
            brain._append_working_memory(
                vector,
                assembly_id="working-%d" % index,
                source="test",
                salience=float(index + 1) / 10.0,
            )
        snapshot = brain.workspace_snapshot()["latentWorkspace"]
        self.assertEqual(snapshot["resident"], 2)
        self.assertEqual(snapshot["paged"], 3)
        self.assertEqual(snapshot["occupancy"], 5)
        self.assertTrue(
            (brain.engine_path / "state" / "working-memory.sqlite3").is_file()
        )
        brain.save()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "offload-brain")
        restored = reloaded.workspace_snapshot()["latentWorkspace"]
        self.assertEqual(restored["resident"], 2)
        self.assertEqual(restored["paged"], 3)
        self.assertEqual(restored["occupancy"], 5)
        self.assertFalse(
            reloaded.runtime_card()["state_offload"]["workingMemoryPaging"][
                "denseAttention"
            ]
        )
        self.assertTrue(
            reloaded.runtime_card()["state_offload"]["workingMemoryPaging"][
                "pageInSupported"
            ]
        )
        reloaded.events.close()

    def test_working_pages_are_verified_readable_and_can_page_back_into_ram(self):
        brain = self.make_brain(
            working_memory_slots=4,
            memory_resident_items=1,
        )
        pages = brain.paged_working_memory
        vector = torch.linspace(-1.0, 1.0, brain.config.idea_dim)
        page_id = pages.append(
            vector,
            {
                "assemblyId": "reactivate-me",
                "source": "test",
                "salience": 0.9,
                "retentionScore": 0.8,
                "rehearsals": 3,
            },
        )

        read_vector, read_metadata = pages.read(page_id)
        self.assertTrue(torch.equal(read_vector, vector))
        self.assertEqual(read_metadata["assemblyId"], "reactivate-me")
        self.assertEqual(pages.count(), 1)
        self.assertEqual(pages.status()["reads"], 1)

        restored = pages.page_in_hot(
            hot_assembly_ids=["reactivate-me"], limit=1
        )
        self.assertEqual(len(restored), 1)
        self.assertTrue(torch.equal(restored[0][0], vector))
        self.assertEqual(restored[0][1]["pageId"], page_id)
        self.assertEqual(pages.count(), 0)
        status = pages.status()
        self.assertEqual(status["pageIns"], 1)
        self.assertGreaterEqual(status["bytesRead"], vector.numel() * 8)
        self.assertTrue(status["runtimeReadable"])
        self.assertTrue(status["dynamicHotCold"])
        brain.events.close()

    def test_working_page_trim_removes_lowest_live_priority_not_oldest(self):
        brain = self.make_brain(
            working_memory_slots=3,
            memory_resident_items=1,
        )
        pages = brain.paged_working_memory
        kept_id = pages.append(
            torch.ones(brain.config.idea_dim),
            {
                "assemblyId": "older-important",
                "salience": 1.0,
                "retentionScore": 1.0,
                "rehearsals": 4,
            },
        )
        removed_id = pages.append(
            torch.zeros(brain.config.idea_dim),
            {
                "assemblyId": "newer-cold",
                "salience": 0.0,
                "retentionScore": 0.0,
                "rehearsals": 0,
            },
        )
        self.assertEqual(pages.trim_to(1), 1)
        self.assertEqual(pages.page_ids(), (kept_id,))
        with self.assertRaises(KeyError):
            pages.read(removed_id)
        brain.events.close()

    def test_repeated_saves_reclaim_unreachable_checkpoint_generations(self):
        brain = self.make_brain()
        first_state_generation = brain.mutable_state_manifest[
            "activeGeneration"
        ]
        first_substrate_generation = brain.memory.persistence_manifest[
            "activeGeneration"
        ]
        for index in range(4):
            brain.memory.learn(
                "checkpoint garbage collection marker %d" % index,
                source="test",
            )
            with torch.no_grad():
                next(brain.decoder.parameters()).reshape(-1)[0].add_(0.0001)
            brain.save()

        state_generations = {
            item.name
            for item in (brain.engine_path / "state" / "generations").iterdir()
            if item.is_dir()
        }
        substrate_generations = {
            item.name
            for item in (
                brain.engine_path / "substrate" / "generations"
            ).iterdir()
            if item.is_dir()
        }
        self.assertLessEqual(len(state_generations), 2)
        self.assertLessEqual(len(substrate_generations), 2)
        self.assertNotIn(first_state_generation, state_generations)
        self.assertNotIn(first_substrate_generation, substrate_generations)
        runtime = brain.runtime_card()["state_offload"]["checkpoint"]
        self.assertTrue(
            runtime["garbageCollection"]["mutableState"]["completed"]
        )
        self.assertTrue(
            runtime["garbageCollection"]["substrate"]["completed"]
        )
        self.assertGreater(
            runtime["garbageCollection"]["mutableState"]["bytesReclaimed"],
            0,
        )
        self.assertGreater(
            runtime["garbageCollection"]["substrate"]["bytesReclaimed"],
            0,
        )
        expected_checksum = brain.parameter_checksum()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "offload-brain")
        self.assertEqual(reloaded.parameter_checksum(), expected_checksum)
        self.assertEqual(
            reloaded.memory.persistence_manifest["activeGeneration"],
            brain.memory.persistence_manifest["activeGeneration"],
        )
        reloaded.events.close()

    def test_corrupt_active_pointer_and_runtime_copy_recover_from_brain_generation(self):
        brain = self.make_brain()
        brain._append_replay(
            torch.arange(brain.config.idea_dim, dtype=torch.float32),
            importance=1.0,
        )
        brain.save()
        expected_checksum = brain.parameter_checksum()
        engine = brain.engine_path
        # Simulate a crash that left the convenience active pointer and runtime
        # copy damaged. brain.json still atomically names the complete prior
        # immutable generation.
        (engine / "state" / "manifest.json").write_text(
            "{incomplete", encoding="utf-8"
        )
        (engine / "state" / "generations" / "incomplete").mkdir(
            parents=True, exist_ok=True
        )
        (engine / "core.safetensors").write_bytes(b"partial checkpoint")
        brain.events.close()

        recovered = AdaptiveBrain.load(self.root, "offload-brain")
        self.assertEqual(recovered.parameter_checksum(), expected_checksum)
        self.assertEqual(len(recovered.replay), 1)
        recovery = recovered.runtime_card()["state_offload"]["checkpoint"][
            "lastRecovery"
        ]
        self.assertTrue(recovery["recovered"])
        self.assertIn("repaired", recovery["reason"])
        repaired_pointer = json.loads(
            (engine / "state" / "manifest.json").read_text("utf-8")
        )
        brain_pointer = json.loads(
            (engine / "brain.json").read_text("utf-8")
        )["mutable_state"]
        self.assertEqual(repaired_pointer, brain_pointer)
        recovered.events.close()

    def test_packaged_style_fork_copies_every_generation_before_load(self):
        brain = self.make_brain()
        brain.learn_experience(
            "A fork keeps its complete copy-on-write neural generation.",
            steps=1,
            importance=1.0,
        )
        brain.save()
        expected_parameters = brain.parameter_checksum()
        expected_replay = len(brain.replay)
        source_engine = brain.engine_path
        fork_root = Path(self.temporary.name) / "packaged-fork"
        fork_engine = fork_root / "engine"
        fork_engine.mkdir(parents=True)
        for filename in ("core.safetensors", "plasticity.safetensors"):
            shutil.copy2(source_engine / filename, fork_engine / filename)
        copy_substrate_snapshot(source_engine, fork_engine)
        copy_mutable_state_snapshot(source_engine, fork_engine)
        metadata = json.loads((source_engine / "brain.json").read_text("utf-8"))
        metadata["brain_id"] = "packaged-fork"
        metadata["name"] = "Packaged fork"
        metadata["config"]["name"] = "Packaged fork"
        (fork_engine / "brain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        brain.events.close()

        forked = AdaptiveBrain.load(fork_root, "packaged-fork")
        self.assertEqual(forked.parameter_checksum(), expected_parameters)
        self.assertEqual(len(forked.replay), expected_replay)
        substrate_pointer = metadata["substrate"]["persistence"]
        mutable_pointer = metadata["mutable_state"]
        self.assertTrue(
            (fork_engine / "substrate" / substrate_pointer["generationManifest"]).is_file()
        )
        self.assertTrue(
            (fork_engine / "state" / mutable_pointer["generationManifest"]).is_file()
        )
        forked.events.close()

        committed_pointer = (fork_engine / "state" / "manifest.json").read_bytes()
        committed_replay = (fork_engine / "state" / "replay.sqlite3").read_bytes()
        with patch(
            "omni_core.offload.shutil.copy2",
            side_effect=OSError("simulated staged COW interruption"),
        ):
            with self.assertRaisesRegex(OSError, "staged COW interruption"):
                copy_mutable_state_snapshot(source_engine, fork_engine)
        self.assertEqual(
            (fork_engine / "state" / "manifest.json").read_bytes(),
            committed_pointer,
        )
        self.assertEqual(
            (fork_engine / "state" / "replay.sqlite3").read_bytes(),
            committed_replay,
        )


if __name__ == "__main__":
    unittest.main()
