from __future__ import annotations

from itertools import product
from typing import Any


def values_for(specification: tuple) -> list[Any]:
    if specification[0] == "choice":
        return list(specification[1])
    if specification[0] == "int":
        low, high = specification[1], specification[2]
        return sorted(set([low, (low + high) // 2, high]))
    raise ValueError(f"Unsupported search-space kind: {specification[0]!r}")


def decode_vector(search_space: dict[str, tuple], vector: list[Any]) -> dict[str, Any]:
    parameters = {}
    for value, (name, specification) in zip(vector, search_space.items()):
        if specification[0] == "int":
            parameters[name] = int(round(value))
        elif specification[0] == "choice":
            parameters[name] = specification[1][int(round(value))]
        else:
            raise ValueError(f"Unsupported search-space kind: {specification[0]!r}")
    return parameters


def pairwise_configurations(search_space: dict[str, tuple], limit: int) -> list[dict[str, Any]]:
    names = list(search_space)
    levels = [values_for(search_space[name]) for name in names]
    candidates = [dict(zip(names, combination)) for combination in product(*levels)]
    pairs = {
        (i, j, left, right)
        for i in range(len(names))
        for j in range(i + 1, len(names))
        for left in levels[i]
        for right in levels[j]
    }
    selected = []
    covered = set()
    while candidates and len(selected) < limit and covered != pairs:
        candidate = max(
            candidates,
            key=lambda config: sum(
                (i, j, config[names[i]], config[names[j]]) not in covered
                for i in range(len(names))
                for j in range(i + 1, len(names))
            ),
        )
        candidates.remove(candidate)
        selected.append(candidate)
        covered.update(
            (i, j, candidate[names[i]], candidate[names[j]])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )
    return selected

