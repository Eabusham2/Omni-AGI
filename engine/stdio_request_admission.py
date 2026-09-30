"""Selected-budget stdio allocations and explicit retained request ownership.

Pure protocol helpers. No model loading, OS pinning or hard RSS guarantee.
"""
import json
import threading


class RequestMemoryPressure(RuntimeError):
    def __init__(self, message, *, status=None, prefix=b""):
        super().__init__(message)
        self.status, self.prefix = status or {}, prefix[:4096]


def reserve_request_bytes(callback, size, operation):
    if callback is None:
        return None
    try:
        return callback(max(0, int(size)), operation)
    except Exception as error:
        raise RequestMemoryPressure("request allocation waits for selected RAM budget",
            status=getattr(error, "status", {})) from error


def decoded_request_bound(raw):
    """Conservative JSON allocation bound scanned BEFORE json.loads.

    Long strings are proportional, not charged a worst-case 64x object ratio.
    Container/entry overhead covers small highly nested JSON objects instead.
    UTF-8 decoding, raw/str copies and parser temporary storage are included.
    """
    size, quoted, escaped, scalar = 4096 + 12 * len(raw), False, False, False
    for byte in raw:
        if quoted:
            if escaped: escaped = False
            elif byte == 92: escaped = True
            elif byte == 34: quoted = False
            continue
        if byte == 34:
            quoted = True; size += 96; scalar = False
        elif byte == 123: size += 384; scalar = False
        elif byte == 91: size += 128; scalar = False
        elif byte == 58: size += 192; scalar = False
        elif byte == 44: size += 32; scalar = False
        elif byte in b" \t\r\n]}": scalar = False
        elif not scalar: size += 64; scalar = True
    return size


def request_id_hint(prefix):
    """Only a complete root-level ID value; never a nested observation ID."""
    try:
        text = prefix.decode("utf-8")
    except UnicodeError:
        return None
    decoder, offset = json.JSONDecoder(), 0
    while offset < len(text) and text[offset].isspace(): offset += 1
    if offset >= len(text) or text[offset] != "{": return None
    offset += 1
    while offset < len(text):
        while offset < len(text) and text[offset].isspace(): offset += 1
        try:
            key, end = decoder.raw_decode(text, offset)
        except (ValueError, RecursionError): return None
        if not isinstance(key, str): return None
        offset = end
        while offset < len(text) and text[offset].isspace(): offset += 1
        if offset >= len(text) or text[offset] != ":": return None
        offset += 1
        while offset < len(text) and text[offset].isspace(): offset += 1
        if key == "id":
            try: value, end = decoder.raw_decode(text, offset)
            except (ValueError, RecursionError): return None
            return value if type(value) in (str, int) else None
        # Skipping an earlier params object must not allocate its nested JSON
        # merely to correlate a rejected frame. Balanced lexical scan only.
        quoted, escaped, depth, end = False, False, 0, offset
        while end < len(text):
            char = text[end]
            if quoted:
                if escaped: escaped = False
                elif char == "\\": escaped = True
                elif char == '"': quoted = False
            elif char == '"': quoted = True
            elif char in "[{": depth += 1
            elif char in "]}":
                if not depth: break
                depth -= 1
            elif char == "," and not depth: break
            end += 1
        if quoted or depth: return None
        offset = end
        while offset < len(text) and text[offset].isspace(): offset += 1
        if offset >= len(text) or text[offset] != ",": return None
        offset += 1
    return None


class RequestAllocationReceipt:
    def __init__(self, leases=()):
        self._leases = [lease for lease in leases if lease is not None]
        self._count, self._lock = 1, threading.Lock()

    def retain(self):
        with self._lock:
            if self._count < 1: raise RuntimeError("request allocation already released")
            self._count += 1
        return _RetainedRequestAllocation(self)

    def release(self):
        with self._lock:
            if self._count < 1: return
            self._count -= 1
            if self._count: return
            leases, self._leases = self._leases, []
        for lease in leases:
            # Objects existed: don't falsely credit delayed Python allocator
            # release before a fresh family RSS sample observes it.
            lease.mark_allocated()
            lease.release()


class _RetainedRequestAllocation:
    def __init__(self, receipt): self.receipt = receipt
    def release(self):
        receipt, self.receipt = self.receipt, None
        if receipt is not None: receipt.release()


class AdmittedRequest(dict):
    def __init__(self, value, receipt):
        super().__init__(value)
        self._request_allocation = receipt  # Attribute, never serialized JSON.


def retain_request_allocation(request):
    receipt = getattr(request, "_request_allocation", None)
    return receipt.retain() if receipt is not None else None
