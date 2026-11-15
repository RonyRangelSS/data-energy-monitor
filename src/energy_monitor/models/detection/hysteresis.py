from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class HysteresisState:
    active: bool = False
    above: int = 0
    below: int = 0


def step(state: HysteresisState, opening: bool, closing: bool, consecutive: int) -> HysteresisState:
    if state.active:
        below = state.below + 1 if closing else 0
        if below >= consecutive:
            return HysteresisState()
        return replace(state, below=below)
    above = state.above + 1 if opening else 0
    if above >= consecutive:
        return HysteresisState(active=True)
    return replace(state, above=above)


def apply_hysteresis(
    groups: np.ndarray, opening: np.ndarray, closing: np.ndarray, consecutive: int
) -> np.ndarray:
    active = np.zeros(len(groups), dtype=bool)
    opening_flags, closing_flags = opening.tolist(), closing.tolist()
    is_active, above, below, previous = False, 0, 0, None
    for position, group in enumerate(groups.tolist()):
        if group != previous:
            is_active, above, below, previous = False, 0, 0, group
        if is_active:
            below = below + 1 if closing_flags[position] else 0
            if below >= consecutive:
                is_active, above, below = False, 0, 0
        else:
            above = above + 1 if opening_flags[position] else 0
            if above >= consecutive:
                is_active, above, below = True, 0, 0
        active[position] = is_active
    return active
