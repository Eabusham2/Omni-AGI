import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import PackedAdaptiveBitConv2d, PackedAdaptiveBitLinear
from omni_core.spiking import STDPSynapses
from omni_core.ternary_packing import (
    TernaryCoverageError,
    TernaryIntegrityError,
    TernaryPackingError,
    TernaryTensorSpec,
    collect_module_ternary_tensors,
    decode_ternary_2bit,
    encode_ternary_2bit,
    export_module_ternary_shards,
    export_ternary_shards,
    inspect_module_ternary_layout,
    verify_ternary_shards,
)


class TinyTernaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = PackedAdaptiveBitLinear(5, 3, bias=False)
        self.image = PackedAdaptiveBitConv2d(1, 2, 3, padding=1, bias=False)
        self.recurrent = STDPSynapses(3, 3)


class DenseButEligible(nn.Linear):
    ternary_eligible = True


class TernaryPackingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_cold_layout_inspection_never_evaluates_projection_weights(self):
        model = TinyTernaryNetwork()
        grown = torch.tensor([1, -1, 0], dtype=torch.int8)
        with mock.patch.object(
            model.input,
            "effective_weight",
            side_effect=AssertionError("projection evaluation during cold load"),
        ):
            layout = inspect_module_ternary_layout(
                {"cortex": model},
                dynamic_synapses={"substrate.grown.weights": grown},
            )
        self.assertEqual(
            layout,
            {
                "cortex.image.weight": ((2, 1, 3, 3), "projection"),
                "cortex.input.weight": ((3, 5), "projection"),
                "cortex.recurrent.weights": ((3, 3), "dynamic-synapse"),
                "substrate.grown.weights": ((3,), "dynamic-synapse"),
            },
        )

    def test_two_bit_codec_is_exact_deterministic_and_strict(self):
        values = torch.tensor(
            [-1, 0, 1, -1, 1, 0, 0, -1, 1], dtype=torch.int8
        )
        first = encode_ternary_2bit(values)
        second = encode_ternary_2bit(values.clone().to(torch.float32))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(
            torch.equal(decode_ternary_2bit(first, values.shape), values)
        )

        with self.assertRaises(TernaryPackingError):
            encode_ternary_2bit(torch.tensor([-1.0, 0.25, 1.0]))
        with self.assertRaises(TernaryIntegrityError):
            decode_ternary_2bit(bytes([0b00000011]), (1,))
        # One active -1 code followed by non-canonical -1 padding codes.
        with self.assertRaises(TernaryIntegrityError):
            decode_ternary_2bit(bytes([0]), (1,))
        with self.assertRaises(TernaryIntegrityError):
            decode_ternary_2bit(first[:-1], values.shape)

    def test_module_and_dynamic_synapses_export_exact_roundtrip(self):
        model = TinyTernaryNetwork()
        model.recurrent.set_effective_weights(
            torch.tensor(
                [
                    [-1, 0, 1],
                    [0, -1, 1],
                    [1, -1, 0],
                ],
                dtype=torch.int8,
            )
        )
        grown = torch.tensor([[1, 0, -1], [0, 1, 0]], dtype=torch.int8)
        specs = collect_module_ternary_tensors(
            {"cortex": model},
            dynamic_synapses={"substrate.grown.weights": grown},
        )
        names = [spec.name for spec in specs]
        self.assertEqual(names, sorted(names))
        self.assertEqual(
            set(names),
            {
                "cortex.image.weight",
                "cortex.input.weight",
                "cortex.recurrent.weights",
                "substrate.grown.weights",
            },
        )

        with tempfile.TemporaryDirectory() as first_folder, tempfile.TemporaryDirectory() as second_folder:
            first_path = Path(first_folder)
            second_path = Path(second_folder)
            first_manifest = export_module_ternary_shards(
                first_path,
                {"cortex": model},
                dynamic_synapses={"substrate.grown.weights": grown},
                expected_names=names,
                metadata={"brainId": "deterministic-fixture"},
            )
            second_manifest = export_module_ternary_shards(
                second_path,
                {"cortex": model},
                dynamic_synapses={"substrate.grown.weights": grown},
                expected_names=reversed(names),
                metadata={"brainId": "deterministic-fixture"},
            )
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (first_path / "manifest.json").read_bytes(),
                (second_path / "manifest.json").read_bytes(),
            )

            verified = verify_ternary_shards(
                first_path, expected_names=names
            )
            for spec in specs:
                self.assertEqual(verified.tensors[spec.name].dtype, torch.int8)
                self.assertTrue(
                    torch.equal(
                        verified.tensors[spec.name],
                        spec.values.detach().cpu().to(torch.int8),
                    )
                )
            entries = {
                entry["name"]: entry
                for entry in verified.manifest["tensors"]
            }
            expected_scale = float(model.input._packed_forward_scale.item())
            self.assertAlmostEqual(
                entries["cortex.input.weight"]["scale"], expected_scale
            )
            self.assertEqual(
                entries["cortex.recurrent.weights"]["kind"],
                "dynamic-synapse",
            )
            self.assertEqual(
                entries["substrate.grown.weights"]["sourceDtype"], "int8"
            )
            manifest_bytes = (first_path / "manifest.json").read_bytes()
            self.assertEqual(
                (first_path / "manifest.sha256").read_text().strip(),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )

    def test_invalid_packed_projection_scale_fails_closed(self):
        model = TinyTernaryNetwork()
        with torch.no_grad():
            model.input._packed_forward_scale.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(
                ValueError, "packed-authoritative scale must be finite"
            ):
                export_module_ternary_shards(Path(folder), {"cortex": model})

    def test_selective_retention_checks_every_shard_without_keeping_every_tensor(self):
        specs = [
            TernaryTensorSpec(
                name="cortex.projection.weight",
                values=torch.tensor([[1, 0, -1, 1]], dtype=torch.int8),
            ),
            TernaryTensorSpec(
                name="substrate.dynamic_synapses.weights",
                values=torch.tensor([-1, 0, 1], dtype=torch.int8),
                kind="dynamic-synapse",
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = export_ternary_shards(path, specs)
            all_values = verify_ternary_shards(path)
            dynamic_only = verify_ternary_shards(
                path, retain_names=("substrate.dynamic_synapses.weights",)
            )
            manifest_only = verify_ternary_shards(path, retain_names=())
            with self.assertRaisesRegex(ValueError, "iterable of strings"):
                verify_ternary_shards(path, retain_names="substrate.dynamic_synapses.weights")
            self.assertEqual(all_values.manifest, manifest)
            self.assertEqual(dynamic_only.manifest, manifest)
            self.assertEqual(manifest_only.manifest, manifest)
            self.assertEqual(set(all_values.tensors), {spec.name for spec in specs})
            self.assertEqual(
                set(dynamic_only.tensors), {"substrate.dynamic_synapses.weights"}
            )
            self.assertEqual(manifest_only.tensors, {})
            self.assertTrue(
                torch.equal(
                    dynamic_only.tensors["substrate.dynamic_synapses.weights"],
                    all_values.tensors["substrate.dynamic_synapses.weights"],
                )
            )

            # An unretained projection must still be hashed and rejected.
            projection = manifest["tensors"][0]
            shard = path / projection["shard"]
            payload = bytearray(shard.read_bytes())
            payload[0] ^= 0x01
            shard.write_bytes(payload)
            with self.assertRaisesRegex(TernaryIntegrityError, "shard checksum mismatch"):
                verify_ternary_shards(path, retain_names=())
            with self.assertRaisesRegex(TernaryIntegrityError, "shard checksum mismatch"):
                verify_ternary_shards(
                    path, retain_names=("substrate.dynamic_synapses.weights",)
                )

            # Even a valid ternary change with repaired outer hashes must fail
            # the unretained tensor's decoded-content checksum.
            payload[0] ^= 0x03  # Original +1 lane becomes a valid -1 lane.
            shard.write_bytes(payload)
            packed_sha = hashlib.sha256(payload).hexdigest()
            projection["packedSha256"] = packed_sha
            next(
                entry for entry in manifest["shards"]
                if entry["file"] == projection["shard"]
            )["sha256"] = packed_sha
            def canonical(value):
                return json.dumps(
                    value, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
            body = dict(manifest)
            body.pop("contentSha256")
            manifest["contentSha256"] = hashlib.sha256(canonical(body)).hexdigest()
            manifest_bytes = canonical(manifest)
            (path / "manifest.json").write_bytes(manifest_bytes)
            (path / "manifest.sha256").write_text(
                hashlib.sha256(manifest_bytes).hexdigest() + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(TernaryIntegrityError, "decoded tensor checksum mismatch"):
                verify_ternary_shards(path, retain_names=())

    def test_duplicate_manifest_names_still_fail_when_nothing_is_retained(self):
        specs = [
            TernaryTensorSpec(
                name="cortex.first.weight",
                values=torch.tensor([[1, 0, -1, 1]], dtype=torch.int8),
            ),
            TernaryTensorSpec(
                name="cortex.second.weight",
                values=torch.tensor([[0, 1, 0, -1]], dtype=torch.int8),
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = export_ternary_shards(path, specs)
            manifest["tensors"][1]["name"] = manifest["tensors"][0]["name"]
            manifest["coverage"]["eligibleTensorNames"][1] = (
                manifest["tensors"][0]["name"]
            )

            def canonical(value):
                return json.dumps(
                    value, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")

            body = dict(manifest)
            body.pop("contentSha256")
            manifest["contentSha256"] = hashlib.sha256(canonical(body)).hexdigest()
            manifest_bytes = canonical(manifest)
            (path / "manifest.json").write_bytes(manifest_bytes)
            (path / "manifest.sha256").write_text(
                hashlib.sha256(manifest_bytes).hexdigest() + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(TernaryIntegrityError, "duplicate tensor names"):
                verify_ternary_shards(path, retain_names=())

    def test_coverage_contract_and_dense_eligible_projection_fail_closed(self):
        model = TinyTernaryNetwork()
        specs = collect_module_ternary_tensors({"cortex": model})
        names = [spec.name for spec in specs]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(TernaryCoverageError, "missing="):
                export_ternary_shards(
                    Path(folder),
                    specs,
                    expected_names=names + ["cortex.new_projection.weight"],
                )

        model.dense = DenseButEligible(3, 2)
        with self.assertRaisesRegex(
            TernaryCoverageError, "unsupported/dense projection"
        ):
            collect_module_ternary_tensors({"cortex": model})

        broken = PackedAdaptiveBitLinear(2, 2)
        broken.ternary = False
        with self.assertRaisesRegex(TernaryCoverageError, "not marked ternary"):
            collect_module_ternary_tensors({"broken": broken})

    def test_corrupt_shards_and_manifest_tampering_are_rejected(self):
        spec = TernaryTensorSpec(
            name="substrate.synapses",
            values=torch.tensor([[-1, 0, 1]], dtype=torch.int8),
            kind="dynamic-synapse",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = export_ternary_shards(path, [spec])
            shard_path = path / manifest["tensors"][0]["shard"]
            payload = bytearray(shard_path.read_bytes())
            payload[0] ^= 0x01
            shard_path.write_bytes(payload)
            with self.assertRaisesRegex(
                TernaryIntegrityError, "shard checksum mismatch"
            ):
                verify_ternary_shards(path)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            export_ternary_shards(path, [spec])
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["coverage"]["complete"] = False
            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            manifest_path.write_bytes(manifest_bytes)
            # Even when an attacker updates the outer checksum, the signed
            # content checksum inside the manifest must still fail.
            (path / "manifest.sha256").write_text(
                hashlib.sha256(manifest_bytes).hexdigest() + "\n"
            )
            with self.assertRaisesRegex(
                TernaryIntegrityError, "content checksum mismatch"
            ):
                verify_ternary_shards(path)

    def test_verifier_rechecks_expected_architecture_names(self):
        spec = TernaryTensorSpec(
            name="cortex.projection.weight",
            values=torch.tensor([[0, 1], [-1, 0]], dtype=torch.int8),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            export_ternary_shards(path, [spec])
            with self.assertRaisesRegex(
                TernaryIntegrityError, "coverage mismatch"
            ):
                verify_ternary_shards(
                    path,
                    expected_names=[
                        "cortex.projection.weight",
                        "cortex.missing.weight",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
