"""Opt-in structural host/native worker; never creates or trains an origin."""
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omni_core.brain import AdaptiveBrain
from omni_core.offload import ResourcePolicy

root = Path(os.environ["OMNI_NATIVE_LIFECYCLE_HOST_ROOT"]).resolve()
if not root.name.startswith("omni-native-lifecycle-host-"):
    raise ValueError("lifecycle host proof requires its own temporary root")

torch.set_num_threads(1)
with patch.object(ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024), \
        patch.object(AdaptiveBrain, "_select_structured_actions", return_value=({"talk": 1., "ponder": 0.}, [])):
    # Only action choice is controlled. Real packed decoder, learning flow,
    # persistence, RPC scheduling, ownership and cache restoration are used.
    import worker
    original_notify = worker.Worker.notify
    def paced_notify(self, kind, *args, **kwargs):
        original_notify(self, kind, *args, **kwargs)
        if kind == "chat-token":
            time.sleep(.05)  # Deterministic control-lane handoff, not a TPS benchmark.
    worker.Worker.notify = paced_notify
    raise SystemExit(worker.main())
