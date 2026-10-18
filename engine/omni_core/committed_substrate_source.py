"""Internal verified regional source capability, never accepted from RPC JSON."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Any


@dataclass(frozen=True)
class VerifiedCommittedSubstrateSource:
    engine_directory: Path
    brain_id: str
    metadata: Mapping[str, Any]
    assert_current: Callable[[], None]

    def resolve(self, engine_directory, brain_id):
        self.assert_current()
        if Path(engine_directory).resolve() != self.engine_directory.resolve() or brain_id != self.brain_id:
            raise ValueError("verified regional substrate source owner/path differs")
        if self.metadata.get("brain_id") != brain_id:
            raise ValueError("verified regional substrate metadata owner differs")
        return self.metadata
