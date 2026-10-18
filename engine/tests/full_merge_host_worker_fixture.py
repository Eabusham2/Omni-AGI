"""Opt-in tiny real worker for TypeScript host integration; never user instances."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.shared_resource_ledger import SharedResourceLedger

root = Path(os.environ["OMNI_NATIVE_MERGE_FIXTURE_ROOT"]).resolve()
if not root.name.startswith("omni-native-host-"):
    raise ValueError("fixture may only use its dedicated temporary root")
quota = SharedResourceLedger(root / "fixture-quota.sqlite3")


def policy(path, **kwargs):
    kwargs.pop("device_resource_settings", None)
    return ResourcePolicy(path, **kwargs, shared_ledger=quota,
        reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
            available_memory_bytes=6 * 1024**3, process_memory_bytes=128 * 1024**2,
            disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
            managed_process_memory_bytes=128 * 1024**2, managed_memory_verified=True))


with patch("omni_core.brain.ResourcePolicy", policy):
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        identifier, name = sys.argv[2:4]
        config = OmniConfig.micro(name=name, d_model=8, idea_dim=8, n_heads=1, n_layers=1,
            d_ff=16, vsa_dim=16, router_neurons=4, max_seq_len=16,
            working_memory_slots=4, memory_resident_items=4, modality_channels=4,
            image_size=8, audio_samples=16, video_frames=2,
            train_batch_size=1, gradient_accumulation=1)
        brain = AdaptiveBrain(identifier, root / identifier, config)
        try:
            brain.conversation.append("message", ({"id": identifier + "-message", "role": "human",
                "content": name + " saved native experience", "createdAt": "2026-10-01T00:00:00.000Z"},))
            brain.save()
        finally:
            brain.close()
    else:
        import worker
        raise SystemExit(worker.main())
