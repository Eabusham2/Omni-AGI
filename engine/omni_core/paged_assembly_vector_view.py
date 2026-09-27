"""Assembly subset of one packed neuron store without resident member IDs.

Assembly membership is the structural SQLite index, not a second Python set
or a duplicated learned vector table. Iteration reads bounded index pages;
all vector reads and adaptation address the same authoritative neuron row.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping

import torch

from .paged_assembly_index import PagedAssemblyIndex
from .paged_packed_vectors import PagedPackedVectors


class PagedAssemblyVectorView(MutableMapping[str, torch.Tensor]):
    """Mutable vector values with append-only indexed assembly membership."""

    def __init__(self, index: PagedAssemblyIndex, backing: PagedPackedVectors):
        if not isinstance(index, PagedAssemblyIndex):
            raise TypeError("paged assembly vector view needs an assembly index")
        if not isinstance(backing, PagedPackedVectors):
            raise TypeError("paged assembly vector view needs packed neuron rows")
        if index._vectors is not None and index._vectors is not backing:
            raise ValueError("assembly vector view cannot use a second row authority")
        self.index = index
        self.backing = backing

    def __len__(self) -> int:
        return self.index.count()

    def __iter__(self) -> Iterator[str]:
        for page in self.index.iter_pages(page_size=128):
            for record in page.records:
                yield str(record["id"])

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and self.index.get_by_id(key) is not None
            and key in self.backing
        )

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self:
            raise KeyError(key)
        return self.backing[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        # A new assembly has its structural metadata appended first. The
        # vector is its shared neuron row, not a separate stored VSA object.
        if self.index.get_by_id(key) is None:
            raise KeyError("assembly vector requires indexed metadata")
        self.backing[key] = value

    def __delitem__(self, key: str) -> None:
        raise TypeError("assembly membership cannot be unlinked from a paged view")

    def link(self, key: str) -> None:
        """Check an already-loaded assembly alias without storing its ID."""

        if key not in self:
            raise ValueError("assembly vector has no indexed shared neuron row")

    def adapt(self, key: str, target: torch.Tensor, rate: float) -> torch.Tensor:
        if key not in self:
            raise KeyError(key)
        return self.backing.adapt(key, target, rate)
