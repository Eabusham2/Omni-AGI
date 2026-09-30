"""UTF-8 wire syntax only; no language, persona, or response-content policy."""


class Utf8TextBoundary:
    """Track one byte-token stream without publishing an incomplete scalar.

    EOS is valid at every complete boundary, including the initial boundary.
    Overlong encodings, surrogates and values beyond U+10FFFF are invalid wire
    syntax. The saved-message transport forbids NUL; other UTF-8 scalars,
    including other ASCII controls, are permitted without content preferences.
    """

    byte_offset = 3
    eos_id = 2

    def __init__(self):
        self.remaining = 0
        self.minimum = 0x80
        self.maximum = 0xBF
        self.ended = False

    @property
    def complete(self):
        return self.remaining == 0

    def allowed_token_ids(self, token_budget):
        if self.ended:
            return [self.eos_id]
        if self.remaining:
            return [value + self.byte_offset for value in range(self.minimum, self.maximum + 1)]
        values = list(range(1, 0x80))
        # This is an encoding-boundary constraint, not a response-length or
        # language preference. Never start a scalar the caller's remaining
        # byte-token budget cannot finish.
        if token_budget >= 2: values.extend(range(0xC2, 0xE0))
        if token_budget >= 3: values.extend(range(0xE0, 0xF0))
        if token_budget >= 4: values.extend(range(0xF0, 0xF5))
        return [self.eos_id, *(value + self.byte_offset for value in values)]

    def accept(self, token_id):
        token_id = int(token_id)
        if token_id == self.eos_id:
            if not self.complete:
                raise ValueError("EOS inside an incomplete UTF-8 scalar")
            self.ended = True
            return
        if self.ended:
            raise ValueError("text byte after EOS")
        value = token_id - self.byte_offset
        if value == 0:
            raise ValueError("NUL is invalid in the saved-message transport")
        if self.remaining:
            if not self.minimum <= value <= self.maximum:
                raise ValueError("invalid UTF-8 continuation byte")
            self.remaining -= 1
            self.minimum, self.maximum = 0x80, 0xBF
            return
        if 0 <= value <= 0x7F: return
        if 0xC2 <= value <= 0xDF:
            self.remaining = 1
        elif 0xE0 <= value <= 0xEF:
            self.remaining = 2
            if value == 0xE0: self.minimum = 0xA0
            elif value == 0xED: self.maximum = 0x9F
        elif 0xF0 <= value <= 0xF4:
            self.remaining = 3
            if value == 0xF0: self.minimum = 0x90
            elif value == 0xF4: self.maximum = 0x8F
        else:
            raise ValueError("invalid UTF-8 leading byte")
