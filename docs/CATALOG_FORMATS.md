# Declarative catalog formats

Omni AGI Studio accepts two catalog artifacts in addition to complete `.omni`
brains. Both are data-only formats. They cannot declare commands, packages,
plugins, post-install hooks, or repository setup scripts.

## Build recipe JSON

A version-1 recipe is a UTF-8 JSON object no larger than 1 MiB. Unknown fields,
invalid ranges, non-HTTPS remote URLs, and undeclared starter URLs are rejected.
The desktop validates the document before applying it to the Build wizard; the
user can still review every resolved setting before creating a brain.

```json
{
  "schemaVersion": 1,
  "id": "whole-brain-micro",
  "name": "Whole Brain Micro",
  "description": "A small complete OmniCortex recipe.",
  "origin": "blank",
  "hardwareProfile": "micro",
  "architecture": {
    "preset": "whole-brain",
    "text": {
      "vocabularySize": 261,
      "contextLength": 64,
      "dimension": 32,
      "layers": 1,
      "heads": 4,
      "ternary": true
    },
    "spiking": {
      "enabled": true,
      "neurons": 2048,
      "stdp": true,
      "metaplasticity": true,
      "structuralGrowth": "elastic"
    },
    "liquid": {
      "enabled": true,
      "kind": "cfc",
      "units": 32
    },
    "ideas": {
      "enabled": true,
      "hypervectorDimensions": 128
    },
    "modalities": {
      "vision": true,
      "imageGeneration": true,
      "audio": true,
      "video": true
    }
  },
  "memoryRecipe": "human-consolidation",
  "toolPermission": "ask",
  "license": "PolyForm Noncommercial 1.0.0",
  "provenanceUrl": "https://example.org/whole-brain-micro"
}
```

Allowed values are:

- `origin`: `blank` or `starter`. A starter also requires an HTTPS
  `starterUrl` pointing to a compatible `.omni`.
- `hardwareProfile`: `micro`, `personal`, `gpu`, or `workstation`.
- `architecture.preset`: `whole-brain`, `ternary`, `neuromorphic`, `liquid`,
  `symbolic`, or `custom`.
- `memoryRecipe`: `human-consolidation`, `total-recall`, or `synapses-only`.
- `toolPermission`: `off`, `ask`, `auto`, or `full`; the selected initial level
  is expanded across the known tool protocols and remains visible in Build.

Shape fields are range-validated for forward compatibility. The selected
hardware profile owns the final tensor shapes in format version 1; a recipe
does not allocate arbitrary shapes or execute code.

Recipes may be bundled in `catalog/recipes`, opened from a local file, or
downloaded from HTTPS. A catalog entry can pin its SHA-256 digest. Remote
downloads use the same public-network/redirect guard and bounded reader as
other catalog imports.

## `.omnipack` modality pack

An `.omnipack` is a ZIP file containing exactly four paths:

```text
manifest.json
model-card.md
checksums.sha256
tensors/modality.safetensors
```

The compressed file is limited to 512 MiB and expanded content to 1 GiB.
Encrypted entries, symbolic links, duplicate/extra paths, unsupported ZIP
methods, traversal names, and missing checksum records are rejected.
`checksums.sha256` must cover the other three files exactly once.

The manifest contract is:

```json
{
  "format": "omni-modality-pack",
  "formatVersion": 1,
  "architecture": "OmniCortex",
  "architectureSchemaVersion": 1,
  "pack": {
    "id": "example-vision-pack",
    "name": "Example vision pack",
    "version": "1.0.0",
    "modalities": ["vision"]
  },
  "compatibility": {
    "dModel": 64,
    "modalityChannels": 16,
    "imageSize": 16,
    "audioSamples": 256,
    "videoFrames": 4
  },
  "licenseLedger": {
    "license": "MIT",
    "provenanceUrl": "https://example.org/model-card",
    "sourceUrl": "https://example.org/training-source"
  },
  "files": {
    "model-card.md": {
      "sha256": "64 lowercase hexadecimal characters",
      "bytes": 1234
    },
    "tensors/modality.safetensors": {
      "sha256": "64 lowercase hexadecimal characters",
      "bytes": 5678
    }
  }
}
```

`modalities` may contain one or more unique values from `vision`, `image`,
`audio`, and `video`. Every tensor name must begin with the matching
`modalities.<kind>.` namespace, and every declared modality must contribute at
least one tensor.

Installation has two validation boundaries:

1. Electron validates the ZIP directory, checksums, manifest, license ledger,
   safe-tensor header, dtypes, offsets, sizes, and namespace inventory, then
   stages only the tensor and inspectable metadata below that brain.
2. The Python worker independently checks architecture/schema, the complete
   compatibility record, enabled modality set, exact expected tensor inventory
   and shapes, finite values, and namespace. It transactionally restores the
   old modality state if any load or save step fails.

An installed pack changes only the declared modality parameters. It cannot
replace the text decoder, spiking router, liquid controller, idea graph,
permissions, credentials, or application code. Its digest, source label,
license, provenance, model card, and installed time remain inspectable.

## GitHub and third-party distribution

GitHub is treated as a transport, not an installer. A URL may point directly to
a recipe JSON, `.omnipack`, or complete `.omni`, but Omni AGI Studio never
clones and executes an arbitrary repository as part of import. Publishers
should provide immutable release assets, SHA-256 values, a model card, dataset
provenance, and a license declaration. License metadata is informative and
does not prove that a publisher owns redistribution rights.
