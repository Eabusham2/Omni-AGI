"""Forward viewport geometry from an existing native owner, without execution."""

from typing import Any, Dict, Optional


def describe_forward_output(module: Any) -> Optional[Dict[str, Any]]:
    """Report only dimensions declared by the loaded projection object.

    Paths, packed row counts and repeated-contact slots cannot establish a
    forward output axis. A storage-only owner therefore has no descriptor.
    """
    if isinstance(getattr(module, "num_embeddings", None), int):
        axis, size = "embedding-column", getattr(module, "embedding_dim", None)
    elif getattr(module, "dimensions", None) in (1, 2, 3) and hasattr(module, "out_channels"):
        axis, size = "output-channel", module.out_channels
    elif hasattr(module, "out_features"):
        axis, size = "output-row", module.out_features
    else:
        return None
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return None
    return {"outputAxis": axis, "outputSize": size}
