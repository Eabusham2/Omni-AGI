"""Pure arithmetic inventory and canonical trusted native shape descriptors.

No Torch, model constructor, data read, or quality/throughput measurement.
Packed bytes use the actual projection row layout, including skinny biases.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def native_core_inventory(shape: Mapping[str, Any]) -> dict[str, int]:
    d, layers, ff = (int(shape[key]) for key in ("dModel", "layers", "feedForward"))
    vsa, router, channels = (int(shape[key]) for key in ("vsaDimensions", "routerNeurons", "modalityChannels"))
    workspace = int(shape["workspaceLatents"])
    side, audio, frames = int(shape["imageSize"]) // 4, int(shape["audioSamples"]) // 4, int(shape["videoFrames"])
    vocab = int(shape.get("vocabSize", 261))
    counts = {"logicalParameters": 0, "projectionParameters": 0, "tableParameters": 0,
              "routerSynapseParameters": router * router, "packedWeightBytes": 0,
              "resistanceBytes": 0, "packedOwners": 0, "maximumProjectionBytes": 0}

    def matrix(width, rows, bias=0, *, table=False, repeat=1):
        parameters = width * rows + bias
        packed = rows * ((width + 3) // 4) + (bias + 3) // 4
        counts["logicalParameters"] += repeat * parameters
        counts["tableParameters" if table else "projectionParameters"] += repeat * parameters
        counts["packedWeightBytes"] += repeat * packed
        counts["resistanceBytes"] += repeat * (rows + bool(bias))
        counts["packedOwners"] += repeat
        counts["maximumProjectionBytes"] = max(counts["maximumProjectionBytes"], packed + rows + bool(bias) + 8)

    def linear(width, rows, bias=False, **kwargs):
        matrix(width, rows, rows if bias else 0, **kwargs)

    def norm(width, repeat=1):
        matrix(width, 1, table=True, repeat=repeat)

    def convolution(inputs, outputs, kernel, *, transposed=False, repeat=1):
        # No initial grouped convolutions exist. Transposed storage rows are
        # outputs*kernel, with inputs packed per row (model.py's exact layout).
        matrix(inputs if transposed else inputs * kernel,
               outputs * kernel if transposed else outputs,
               outputs, repeat=repeat)

    def latent_transformer(tokens):
        linear(tokens, channels, table=True)
        linear(d, channels, True)
        linear(1, channels, True)
        linear(channels, channels, True)
        linear(channels, channels * 3, True, repeat=2)
        linear(channels, channels, True, repeat=2)
        linear(channels, channels * 3, True, repeat=2)
        linear(channels * 3, channels, True, repeat=2)
        linear(channels, channels, True)

    # OmniDecoder: token/workspace embeddings, packed gains and scalar digits.
    matrix(d, vocab, table=True)
    matrix(d, workspace, table=True)
    linear(d, d, repeat=5)
    norm(d)
    linear(d, d)
    matrix(16, 1, table=True, repeat=2)
    norm(d, repeat=2 * layers)
    linear(d, 3 * d, repeat=layers)
    linear(d, d, repeat=layers)
    linear(d, 2 * ff, repeat=layers)
    linear(ff, d, repeat=layers)
    norm(d)
    linear(d, vocab)
    norm(d, repeat=2)
    linear(d, 4 * d, True, repeat=2)
    linear(4 * d, 8, True, repeat=2)
    linear(1024, 96, True)
    linear(d, 96, True)
    linear(1024, 96)
    linear(d + 1024, d, True)
    matrix(d, 258, table=True)
    linear(2 * d, d, True)
    linear(d, 257, True)
    # Bridge, adapter, router and continuous controller.
    linear(vsa, d, True)
    linear(d, 2 * d, True)
    linear(2 * d, d, True)
    linear(d, router, True)
    linear(router, d, True)
    if shape.get("liquidMode", "cfc") == "ltc":
        linear(2 * d, d, True, repeat=2)
        linear(d, d, True)
        linear(1, d, repeat=2)
    else:
        linear(2 * d, d, True, repeat=3)
    linear(d, 4, True)
    # Vision, image VQ/DiT, audio RVQ/DiT and factorized video.
    convolution(3, channels, 9)
    convolution(channels, 2 * channels, 9)
    linear(2 * channels, d, True)
    convolution(3, channels, 16)
    convolution(channels, channels, 16)
    linear(32, channels, table=True)
    convolution(channels, channels, 16, transposed=True)
    convolution(channels, 3, 16, transposed=True)
    linear(d, channels, True)
    linear(channels, d, True)
    latent_transformer(side * side)
    convolution(1, channels, 4)
    convolution(channels, channels, 4)
    linear(32, channels, table=True)
    linear(16, channels, table=True)
    convolution(channels, channels, 4, transposed=True)
    convolution(channels, 1, 4, transposed=True)
    linear(d, channels * audio, True)
    linear(channels, d, True)
    latent_transformer(audio)
    linear(d, channels * frames * side * side, True)
    convolution(channels, channels, 3, repeat=2)
    convolution(channels, channels, 9, repeat=2)
    linear(2 * channels, channels, True, repeat=2)
    convolution(channels, channels, 16, transposed=True)
    convolution(channels, 3, 16, transposed=True)
    convolution(3, channels, 16)
    convolution(channels, channels, 16)
    linear(channels, d, True)
    linear(d, 3, True)
    counts["logicalParameters"] += router * router
    counts["packedWeightBytes"] += router * ((router + 3) // 4)
    counts["gainAndRateBytes"] = 8 * counts["packedOwners"]
    # STDP scalar counters (16), active-prefix scalar (8), first region end (8).
    counts["routerNonweightTensorBytes"] = 10 * router * router + 16 * router + 32
    counts["freshRouteControlBytes"] = 4096 + 24 + 16
    counts["liquidStateBytes"] = 4 * d
    counts["staticNonweightTensorBytes"] = (counts["resistanceBytes"] + counts["gainAndRateBytes"]
        + counts["routerNonweightTensorBytes"] + counts["freshRouteControlBytes"] + counts["liquidStateBytes"])
    return counts


def native_architecture_sha256(descriptor: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in descriptor.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_native_architecture(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping) or descriptor.get("format") != "omni-main-selected-native-architecture" or descriptor.get("formatVersion") != 1:
        raise ValueError("invalid main-selected native architecture descriptor")
    if descriptor.get("architecture") != "OmniCortex" or descriptor.get("externalPretrainedWeights") is not False:
        raise ValueError("native architecture cannot name external/foundation weights")
    if "foundationModelId" in descriptor or "foundation_model_id" in descriptor:
        raise ValueError("native architecture cannot name external/foundation weights")
    shape = descriptor.get("shape")
    inventory = descriptor.get("inventory")
    if not isinstance(shape, Mapping) or not isinstance(inventory, Mapping):
        raise ValueError("native architecture shape/inventory is missing")
    required = ("dModel", "layers", "feedForward", "nHeads", "vsaDimensions", "routerNeurons",
                "modalityChannels", "imageSize", "audioSamples", "videoFrames", "workingMemoryItems",
                "workspaceLatents", "vocabSize")
    if any(isinstance(shape.get(key), bool) or not isinstance(shape.get(key), int)
           or not 1 <= shape[key] <= (1 << 53) - 1 for key in required):
        raise ValueError("native architecture shape must contain positive safe integers")
    if set(shape) != {*required, "liquidMode"}:
        raise ValueError("native architecture shape contains an unsupported dimension")
    if (shape["dModel"] % shape["nHeads"] or (shape["dModel"] // shape["nHeads"]) % 2
        or shape["vocabSize"] != 261 or shape["imageSize"] % 4 or shape["audioSamples"] % 4
        or shape["audioSamples"] < 4 or shape["vsaDimensions"] <= 8
        or shape["imageSize"] < 8 or shape["videoFrames"] < 2 or shape["routerNeurons"] < 2
        or shape.get("liquidMode") not in {"cfc", "ltc"}
        or shape["workspaceLatents"] != max(8, shape["workingMemoryItems"] // 4)):
        raise ValueError("native architecture shape violates the constructor contract")
    actual = native_core_inventory(shape)
    if dict(inventory) != actual or any(value > (1 << 53) - 1 for value in actual.values()):
        raise ValueError("native architecture exact inventory does not match its shape")
    if descriptor.get("sha256") != native_architecture_sha256(descriptor):
        raise ValueError("native architecture descriptor hash mismatch")
    if descriptor.get("qualityEvidence") != "unmeasured-native-quality-deferred":
        raise ValueError("native architecture cannot claim unmeasured quality")
    if descriptor.get("hardwareTier") not in {"micro", "personal", "gpu", "workstation"}:
        raise ValueError("native architecture hardware tier is invalid")
    sizing = descriptor.get("sizing")
    if not isinstance(sizing, Mapping) or any(
        value is not None and not isinstance(value, (str, bool))
        and not (isinstance(value, int) and abs(value) <= (1 << 53) - 1)
        for value in sizing.values()
    ):
        raise ValueError("native architecture sizing must contain flat canonical JSON primitives")
    return dict(descriptor)
