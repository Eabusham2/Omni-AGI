#!/usr/bin/env python3
"""Standalone torchrun entry point for ground-up OmniCortex training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import torch

from omni_core.config import OmniConfig
from omni_core.distributed_runtime import (
    DatasetManifest,
    DistributedLaunchConfig,
    DistributedRunStore,
    initialize_distributed,
    read_distributed_status,
)
from omni_core.distributed_training import (
    DistributedGroundUpTrainer,
    DistributedTrainingOptions,
)
from omni_core.ground_up import resolve_ground_up_curriculum_manifest


_INITIAL_ORIGIN_METADATA_MAX_BYTES = 16 * 1024 * 1024
_INITIAL_ORIGIN_REQUIRED_FILES = (
    "brain.json",
    "core.safetensors",
    "plasticity.safetensors",
    "substrate/manifest.json",
    "state/manifest.json",
    "packed-ternary/manifest.json",
    "packed-ternary/manifest.sha256",
)


def _sha256_identifier(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_integer(value: Any, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and int(value) >= int(minimum)
    )


def _default_run_path(output: Path) -> Path:
    target = Path(output).resolve()
    return target.parent / (".%s.distributed-run" % target.name)


def _read_bounded_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        size = int(path.stat().st_size)
    except OSError as error:
        raise ValueError("%s metadata is unavailable" % label) from error
    if size < 1 or size > _INITIAL_ORIGIN_METADATA_MAX_BYTES:
        raise ValueError("%s metadata size is invalid" % label)
    try:
        with path.open("rb") as stream:
            payload = stream.read(_INITIAL_ORIGIN_METADATA_MAX_BYTES + 1)
    except OSError as error:
        raise ValueError("%s metadata is unavailable" % label) from error
    if len(payload) > _INITIAL_ORIGIN_METADATA_MAX_BYTES:
        raise ValueError("%s metadata size is invalid" % label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("%s metadata is invalid" % label) from error
    if not isinstance(value, dict):
        raise ValueError("%s metadata must be an object" % label)
    return value


def _resolve_initial_ground_up_locator(value: Optional[Path]) -> Optional[Path]:
    """Resolve a live brain only as a locator for its immutable origin.

    This is a cheap CLI boundary check. The trainer copies only the returned
    root's ``engine/origin`` and then authenticates those copied tensors and
    receipts through the normal brain loader. No mutable live checkpoint file
    is admitted as training state here.
    """

    if value is None:
        return None
    try:
        locator = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("--initial-ground-up brain path does not exist") from error
    if not locator.is_dir():
        raise ValueError("--initial-ground-up must name a live brain directory")
    direct_engine = locator.name == "engine" and (locator / "brain.json").is_file()
    direct_origin = (
        locator.name == "origin"
        and locator.parent.name == "engine"
        and (locator / "brain.json").is_file()
    )
    if direct_engine or direct_origin:
        raise ValueError(
            "--initial-ground-up must name the live brain root, not engine/origin"
        )

    engine = locator / "engine"
    current_metadata = engine / "brain.json"
    origin_entry = engine / "origin"
    if (
        not engine.is_dir()
        or engine.is_symlink()
        or not current_metadata.is_file()
        or current_metadata.is_symlink()
        or not origin_entry.is_dir()
        or origin_entry.is_symlink()
    ):
        raise ValueError(
            "--initial-ground-up must name a brain with a distinct immutable "
            "engine/origin"
        )
    try:
        engine_path = engine.resolve(strict=True)
        origin = origin_entry.resolve(strict=True)
        origin.relative_to(engine_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "--initial-ground-up engine/origin cannot escape the live brain"
        ) from error
    if origin == engine_path:
        raise ValueError(
            "--initial-ground-up mutable engine and immutable origin must be distinct"
        )

    resolved_files: Dict[str, Path] = {}
    for relative in _INITIAL_ORIGIN_REQUIRED_FILES:
        candidate = origin / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(
                "--initial-ground-up immutable origin is missing %s" % relative
            )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(origin)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                "--initial-ground-up immutable origin contains an escaping file"
            ) from error
        resolved_files[relative] = resolved
    for relative, immutable_file in resolved_files.items():
        mutable_file = engine_path / relative
        if not mutable_file.is_file():
            continue
        try:
            aliases_mutable_state = os.path.samefile(
                mutable_file, immutable_file
            )
        except OSError as error:
            raise ValueError(
                "--initial-ground-up checkpoint identity could not be verified"
            ) from error
        if aliases_mutable_state:
            raise ValueError(
                "--initial-ground-up mutable checkpoint and immutable origin "
                "must be distinct: %s" % relative
            )

    packed_manifest_path = resolved_files["packed-ternary/manifest.json"]
    packed_checksum_path = resolved_files["packed-ternary/manifest.sha256"]
    try:
        packed_manifest_size = int(packed_manifest_path.stat().st_size)
        packed_checksum_size = int(packed_checksum_path.stat().st_size)
        if (
            packed_manifest_size < 1
            or packed_manifest_size > _INITIAL_ORIGIN_METADATA_MAX_BYTES
            or packed_checksum_size < 64
            or packed_checksum_size > 128
        ):
            raise ValueError(
                "--initial-ground-up packed ternary manifest size is invalid"
            )
        with packed_manifest_path.open("rb") as stream:
            packed_manifest_bytes = stream.read(
                _INITIAL_ORIGIN_METADATA_MAX_BYTES + 1
            )
        if len(packed_manifest_bytes) > _INITIAL_ORIGIN_METADATA_MAX_BYTES:
            raise ValueError(
                "--initial-ground-up packed ternary manifest size is invalid"
            )
        with packed_checksum_path.open("rb") as stream:
            packed_checksum_bytes = stream.read(129)
        if len(packed_checksum_bytes) > 128:
            raise ValueError(
                "--initial-ground-up packed ternary manifest size is invalid"
            )
        packed_checksum = packed_checksum_bytes.decode("ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError(
            "--initial-ground-up packed ternary manifest checksum is unavailable"
        ) from error
    if (
        not _sha256_identifier(packed_checksum)
        or hashlib.sha256(packed_manifest_bytes).hexdigest() != packed_checksum
    ):
        raise ValueError(
            "--initial-ground-up packed ternary manifest checksum is invalid"
        )

    metadata = _read_bounded_json_object(
        resolved_files["brain.json"], "initial ground-up origin"
    )
    config = metadata.get("config")
    manifest = metadata.get("ground_up_training_manifest")
    packed = metadata.get("packed_ternary_manifest")
    expected_curriculum = resolve_ground_up_curriculum_manifest(manifest)
    random_initialization = (
        manifest.get("randomInitialization")
        if isinstance(manifest, Mapping)
        else None
    )
    architecture = (
        manifest.get("architectureScale")
        if isinstance(manifest, Mapping)
        else None
    )
    training_receipt = (
        manifest.get("trainingReceipt")
        if isinstance(manifest, Mapping)
        else None
    )
    conversation = metadata.get("conversation")
    current_context = metadata.get("current_context")
    paged_working_memory = metadata.get("paged_working_memory")
    memory_lifecycle = metadata.get("memory_lifecycle")
    if (
        not isinstance(config, Mapping)
        or config.get("origin_kind") != "ground-up"
        or "foundation_model_id" in config
        or not isinstance(manifest, Mapping)
        or expected_curriculum is None
        or any(
            manifest.get(key) != expected
            for key, expected in expected_curriculum.items()
        )
        or manifest.get("originKind") != "ground-up"
        or manifest.get("externalWeightFiles") != []
        or manifest.get("pretrainedTextCortex") is not None
        or manifest.get("baseFrozen") is not False
        or not isinstance(random_initialization, Mapping)
        or random_initialization.get("algorithm")
        != "torch-seeded-module-initialization-v1"
        or random_initialization.get("seed") != config.get("seed")
        or not _sha256_identifier(random_initialization.get("parameterChecksum"))
        or not _strict_integer(
            random_initialization.get("exactParameterCount"), minimum=1
        )
        or not isinstance(architecture, Mapping)
        or architecture.get("hardwareTier") != config.get("hardware_tier")
        or architecture.get("dimensions") != config.get("d_model")
        or architecture.get("layers") != config.get("n_layers")
        or not _strict_integer(
            architecture.get("denseParameterCount"), minimum=1
        )
        or not isinstance(training_receipt, Mapping)
        or training_receipt.get("format") != "omni-ground-up-training-receipt"
        or training_receipt.get("formatVersion") != 2
        or training_receipt.get("completeCoverage") is not True
        or training_receipt.get("parametersChanged") is not True
        or training_receipt.get("baseFrozen") is not False
        or not _sha256_identifier(training_receipt.get("parameterChecksumBefore"))
        or not _sha256_identifier(training_receipt.get("parameterChecksumAfter"))
        or training_receipt.get("parameterChecksumBefore")
        == training_receipt.get("parameterChecksumAfter")
        or not isinstance(packed, Mapping)
        or packed.get("originKind") != "ground-up"
        or packed.get("pretrainedTextCortex") is not None
        or packed.get("baseFrozen") is not False
        or not _sha256_identifier(packed.get("contentSha256"))
        or packed.get("parameterChecksum")
        != training_receipt.get("parameterChecksumAfter")
        or metadata.get("training_sources") not in (None, [])
        or metadata.get("ingestion_checkpoints") not in (None, {})
        or metadata.get("completed_ingestions") not in (None, [])
        or metadata.get("completed_chat_turns") not in (None, [])
        or metadata.get("workspace_items") not in (None, [])
        or metadata.get("installed_modality_packs") not in (None, [])
        or metadata.get("recent_token_context") not in (None, [])
        or metadata.get("fresh_attention_boundary") is not None
        or (
            paged_working_memory is not None
            and (
                not isinstance(paged_working_memory, Mapping)
                or not _strict_integer(paged_working_memory.get("count"))
                or int(paged_working_memory.get("count", -1)) != 0
            )
        )
        or (
            current_context is not None
            and (
                not isinstance(current_context, Mapping)
                or not _strict_integer(current_context.get("tokenCount", 0))
                or int(current_context.get("tokenCount", 0)) != 0
                or not _strict_integer(current_context.get("recentTokenCount", 0))
                or int(current_context.get("recentTokenCount", 0)) != 0
                or not _strict_integer(current_context.get("sensorySlots", 0))
                or int(current_context.get("sensorySlots", 0)) != 0
                or current_context.get("tokenHash", "") not in (None, "")
            )
        )
        or (
            memory_lifecycle is not None
            and (
                not isinstance(memory_lifecycle, Mapping)
                or memory_lifecycle.get("scratchItems") not in (None, [])
                or memory_lifecycle.get("activeFocus") not in (None, [])
            )
        )
        or not isinstance(conversation, Mapping)
        or not _strict_integer(conversation.get("totalEntries"))
        or int(conversation.get("totalEntries", -1)) != 0
        or not _strict_integer(conversation.get("messageCount"))
        or int(conversation.get("messageCount", -1)) != 0
        or not _strict_integer(conversation.get("actionCount"))
        or int(conversation.get("actionCount", -1)) != 0
        or not _strict_integer(conversation.get("traceCount"))
        or int(conversation.get("traceCount", -1)) != 0
        or not isinstance(metadata.get("counters"), Mapping)
        or not _strict_integer(metadata["counters"].get("inference_count"))
        or int(metadata["counters"].get("inference_count", -1)) != 0
    ):
        raise ValueError(
            "--initial-ground-up immutable origin provenance is missing or invalid"
        )
    return locator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train OmniCortex from local random initialization "
            "under torchrun DDP/FSDP."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start or resume training")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--brain-id", default="distributed-ground-up")
    train.add_argument(
        "--initial-ground-up",
        type=Path,
        help=(
            "optional ground-up brain used only to locate and verify its "
            "pristine immutable engine/origin; mutable current state is never copied"
        ),
    )
    train.add_argument("--dataset-kind", default="")
    train.add_argument(
        "--profile", choices=("micro", "personal", "gpu", "workstation"), default="gpu"
    )
    train.add_argument(
        "--config",
        type=Path,
        help="optional OmniConfig JSON; native origin and device are enforced",
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--global-batch-records", type=int, default=16)
    train.add_argument("--micro-batch-records", type=int, default=0, help="Physical learner microbatch ceiling; 0 uses admitted CPU/CUDA capacity")
    train.add_argument("--gradient-accumulation", type=int, default=0)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--strategy", choices=("auto", "ddp", "fsdp"), default="auto")
    train.add_argument(
        "--fsdp-min-parameter-bytes", type=int, default=2 * 1024**3
    )
    train.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    train.add_argument("--checkpoint-steps", type=int, default=1)
    train.add_argument("--keep-checkpoints", type=int, default=2)
    train.add_argument(
        "--capability-rehearsal-waves",
        type=int,
        default=128,
        help="rehearse and gate tools/actions/imagination every N committed global waves",
    )
    train.add_argument("--resume", choices=("auto", "required", "never"), default="auto")
    train.add_argument("--replace-output", action="store_true")
    train.add_argument(
        "--inject-failure",
        default="",
        metavar="RANK:STEP",
        help=argparse.SUPPRESS,
    )
    train.add_argument("--cpu-threads", type=int, default=0)
    # torchrun 1.x may inject this spelling. Modern torchrun exports only the
    # environment variable, but accepting both keeps Windows launchers safe.
    train.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)

    status = commands.add_parser("status", help="print live training status")
    status.add_argument("--run-dir", type=Path, required=True)

    cancel = commands.add_parser("cancel", help="request graceful cancellation")
    cancel.add_argument("--run-dir", type=Path, required=True)
    cancel.add_argument("--reason", default="user requested cancellation")

    clear = commands.add_parser(
        "clear-cancel", help="clear an old cancellation before an explicit resume"
    )
    clear.add_argument("--run-dir", type=Path, required=True)

    manifest = commands.add_parser(
        "manifest", help="build and verify a source-free DatasetManifest"
    )
    manifest.add_argument("--dataset", type=Path, required=True)
    manifest.add_argument("--dataset-kind", default="")
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace, device: str) -> OmniConfig:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("config JSON root must be an object")
        config = OmniConfig.from_dict(value)
    else:
        config = OmniConfig.from_external(
            {
                "name": "Distributed Ground-up OmniCortex",
                "hardwareTier": args.profile,
                "device": device,
            }
        )
    config.origin_kind = "ground-up"
    config.device = device
    config.validate()
    return config


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        _print(read_distributed_status(args.run_dir))
        return 0
    if args.command == "cancel":
        store = DistributedRunStore(args.run_dir)
        store.initialize()
        store.request_cancel(args.reason)
        _print({"state": "cancel-requested", "runPath": str(store.path)})
        return 0
    if args.command == "clear-cancel":
        store = DistributedRunStore(args.run_dir)
        store.clear_cancel()
        _print({"state": "cancel-cleared", "runPath": str(store.path)})
        return 0
    if args.command == "manifest":
        manifest = DatasetManifest.build(
            args.dataset,
            args.dataset_kind,
            database_path=args.output.resolve().with_suffix(".sqlite3"),
        )
        manifest.write(args.output)
        _print(
            {
                "manifest": str(args.output.resolve()),
                "contentSha256": manifest.content_sha256,
                "validRecords": len(manifest.entries),
                "coverage": manifest.coverage,
            }
        )
        return 0

    launch = DistributedLaunchConfig.from_environment()
    if args.local_rank is not None and int(args.local_rank) != launch.local_rank:
        raise ValueError("command-line local rank disagrees with torchrun environment")
    if args.cpu_threads > 0:
        torch.set_num_threads(max(1, int(args.cpu_threads)))
    elif launch.world_size > 1 and not torch.cuda.is_available():
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // launch.world_size))
    initial_ground_up = _resolve_initial_ground_up_locator(
        args.initial_ground_up
    )

    context = initialize_distributed(requested_device=args.device)
    try:
        output = args.output.resolve()
        run_path = (
            args.run_dir.resolve()
            if args.run_dir is not None
            else _default_run_path(output)
        )
        options = DistributedTrainingOptions(
            epochs=args.epochs,
            global_batch_records=args.global_batch_records,
            micro_batch_records=args.micro_batch_records,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            strategy=args.strategy,
            fsdp_min_parameter_bytes=args.fsdp_min_parameter_bytes,
            amp=args.amp,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoints=args.keep_checkpoints,
            resume=args.resume,
            replace_output=args.replace_output,
            failure_injection=args.inject_failure,
            capability_rehearsal_waves=args.capability_rehearsal_waves,
        )
        trainer = DistributedGroundUpTrainer(
            context=context,
            store=DistributedRunStore(run_path),
            dataset_path=args.dataset,
            output_path=output,
            config=_config(args, str(context.device)),
            options=options,
            brain_id=args.brain_id,
            requested_kind=args.dataset_kind,
            initial_brain_path=initial_ground_up,
        )
        result = trainer.run()
        if context.is_rank_zero:
            _print(result)
        return 0 if result.get("state") in {"complete", "cancelled"} else 1
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
