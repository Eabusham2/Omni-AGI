"""Exact byte-producer timeline declarations without neural/runtime imports."""


def encoded_media_timeline(unit, count=1, rate=1, *, complete=False,
                           stable_count=None, container_audio=None):
    if unit not in {"image", "samples", "frames"}:
        raise ValueError("Unsupported encoded media timeline unit")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (count, rate)):
        raise ValueError("Encoded media count and timebase must be positive integers")
    if unit == "image" and (count != 1 or rate != 1):
        raise ValueError("A delivered image is one snapshot, not timed generated frames")
    result = {"format": "omni-native-media-timeline", "version": 1,
              "producer": "native-byte-encoder", "unit": unit, "start": 0,
              "count": count, "timebaseNumerator": 1,
              "timebaseDenominator": rate,
              "layout": "snapshot" if unit == "image" else
                        "complete-from-zero" if complete else "replacement-from-zero"}
    if stable_count is not None:
        if isinstance(stable_count, bool) or not isinstance(stable_count, int) or not 0 <= stable_count <= count:
            raise ValueError("The producer's stable prefix is outside the encoded range")
        result["stableCount"] = stable_count
    if container_audio is not None:
        if unit != "frames" or not complete or set(container_audio) != {"sampleRate", "sampleCount", "startSample"} or container_audio["startSample"] != 0:
            raise ValueError("Container audio requires an actual completed paired video")
        if any(isinstance(container_audio[key], bool) or not isinstance(container_audio[key], int) or container_audio[key] < 1 for key in ("sampleRate", "sampleCount")):
            raise ValueError("Container audio count/rate must be measured")
        result["containerAudio"] = dict(container_audio)
    return result
