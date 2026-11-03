from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from faro.graph.ct_types import (
    FREE,
    InterfaceKey,
    InterfaceValue,
    ModeDict
)

@dataclass
class ModeSpace:
    """
    Defines:
      - ordered interface keys
      - per-key domain values (including FREE)
      - integer encoding of each value per interface

    Internal node representation:
      node = tuple[int, ...]  (value IDs per interface index)
    """
    keys: List[InterfaceKey]
    per_key_values: Mapping[InterfaceKey, List[tuple]]  # non-free tuples only

    def __post_init__(self) -> None:
        self.keys = list(sorted(self.keys))
        self.key_to_i: Dict[InterfaceKey, int] = {k: i for i, k in enumerate(self.keys)}

        # For each key, build domain list: [FREE] + allowed tuples
        self.domains: List[List[InterfaceValue]] = []
        self.value_to_id: List[Dict[InterfaceValue, int]] = []

        for k in self.keys:
            # include FREE as ()
            dom = [FREE, *self.per_key_values.get(k, [])]
            self.domains.append(dom)
            self.value_to_id.append({val: idx for idx, val in enumerate(dom)})

    @property
    def n_interfaces(self) -> int:
        return len(self.keys)

    def decode(self, node: tuple[int, ...]) -> ModeDict:
        return {self.keys[i]: self.domains[i][vid] for i, vid in enumerate(node)}

    def encode(self, mode: ModeDict) -> tuple[int, ...]:
        out = [0] * self.n_interfaces
        for k, v in mode.items():
            i = self.key_to_i[k]
            out[i] = self.value_to_id[i][v]
        return tuple(out)

    def decode_contact_mode(self, node: tuple[int, ...]) -> ModeDict:
        """Decode with canonical ownership-preserving interface keys."""
        return self.decode(node)

    def decode_contact_mode_seq(self, encoded_path: Sequence[tuple[int, ...]]) -> dict[InterfaceKey, list[tuple]]:
        out = {key: [] for key in self.keys}
        for node in encoded_path:
            for key, value in self.decode(node).items():
                out[key].append(value)
        return out if encoded_path else {}

    def encode_contact_mode(self, contact_mode: Mapping[InterfaceKey, InterfaceValue]) -> tuple[int, ...]:
        return self.encode(contact_mode)

    def encode_contact_mode_seq(self, contact_mode_seq: Mapping[InterfaceKey, Sequence[InterfaceValue]]) -> tuple[tuple[int, ...], ...]:
        if not contact_mode_seq:
            return ()
        lengths = {len(values) for values in contact_mode_seq.values()}
        if len(lengths) != 1:
            raise ValueError("All contact schedules must have the same number of modes.")
        keys = tuple(contact_mode_seq)
        return tuple(self.encode(dict(zip(keys, values)))
                     for values in zip(*contact_mode_seq.values()))
