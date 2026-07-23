"""A deterministic UTF-8 byte tokenizer with no external model files."""

from typing import Iterable, List

import torch


class ByteTokenizer:
    """Maps UTF-8 bytes to ids while reserving three special tokens."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3
    human_id = 259
    brain_id = 260
    vocab_size = 261

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = 0,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(int(value) + self.byte_offset for value in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        values = []
        for token_id in ids:
            token = int(token_id)
            if token < self.byte_offset:
                if skip_special:
                    continue
                continue
            if token < self.human_id:
                values.append(token - self.byte_offset)
        return bytes(values).decode("utf-8", errors="replace")

    def dialogue(self, human: str, brain: str = "", complete: bool = True) -> List[int]:
        ids = [self.bos_id, self.human_id]
        ids.extend(value + self.byte_offset for value in human.encode("utf-8"))
        ids.append(self.brain_id)
        ids.extend(value + self.byte_offset for value in brain.encode("utf-8"))
        if complete:
            ids.append(self.eos_id)
        return ids

    def tensor(
        self,
        text: str,
        device: torch.device,
        max_length: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                self.encode(
                    text,
                    add_bos=add_bos,
                    add_eos=add_eos,
                    max_length=max_length,
                )
            ],
            dtype=torch.long,
            device=device,
        )
