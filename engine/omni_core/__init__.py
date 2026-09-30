"""OmniCortex: a local-first adaptive neural research engine.

Every brain begins from native random OmniCortex weights. Model weights are not
hidden inside this Python package.
"""

from .brain import AdaptiveBrain
from .config import OmniConfig
from .model import OmniDecoder, RMSNorm
from .tokenizer import ByteTokenizer

__all__ = [
    "AdaptiveBrain",
    "ByteTokenizer",
    "OmniConfig",
    "OmniDecoder",
    "RMSNorm",
]

__version__ = "1.1.1"
