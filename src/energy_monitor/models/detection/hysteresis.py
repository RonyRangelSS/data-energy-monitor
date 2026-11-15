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
    state = HysteresisState()
    previous = None
    for position, group in enumerate(groups.tolist()):
        if group != previous:
            state, previous = HysteresisState(), group
        state = step(state, bool(opening[position]), bool(closing[position]), consecutive)
        active[position] = state.active
    return active
