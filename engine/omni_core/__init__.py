"""OmniCortex: a small, local-first adaptive neural research engine.

The package intentionally ships without pretrained weights.  A new brain is
randomly initialized and only changes through explicit experience, training,
or consolidation calls.
"""

from .brain import AdaptiveBrain
from .config import OmniConfig
from .model import BitLinear, OmniDecoder, RMSNorm, ternary_quantize
from .tokenizer import ByteTokenizer

__all__ = [
    "AdaptiveBrain",
    "BitLinear",
    "ByteTokenizer",
    "OmniConfig",
    "OmniDecoder",
    "RMSNorm",
    "ternary_quantize",
]

__version__ = "0.1.0"
