import struct

import torch

from omni_core.brain import AdaptiveBrain


def test_video_apng_fallback_declares_automatic_infinite_replay():
    payload = AdaptiveBrain._apng_bytes(
        torch.zeros((3, 4, 2, 2), dtype=torch.float32),
        fps=8,
    )
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    position = 8
    animation_control = None
    while position + 12 <= len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        chunk_type = payload[position + 4 : position + 8]
        chunk_data = payload[position + 8 : position + 8 + length]
        if chunk_type == b"acTL":
            animation_control = struct.unpack(">II", chunk_data)
            break
        position += 12 + length

    # APNG num_plays=0 means replay forever. The gallery can therefore label
    # this verified video fallback as automatically looping animation without
    # inventing HTMLVideoElement controls that APNG does not support.
    assert animation_control == (4, 0)
