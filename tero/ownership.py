"""Finite ownership states for zero, one, and two group deletions."""
from collections import deque
from functools import lru_cache
import numpy as np

def initial_state(m):
    return (1 << m)-1, (), ()

@lru_cache(maxsize=None)
def transition(state, mask):
    uncovered, single, double = state
    residual = {part: part & ~mask for part in single}
    new_single = uncovered & mask
    next_single = tuple(sorted([part for part in residual.values() if part] +
                               ([new_single] if new_single else [])))
    next_double = []
    for part, endpoints in double:
        remain = part & ~mask
        if remain:
            next_double.append((remain, tuple(sorted(residual[owner] for owner in endpoints if residual[owner]))))
    for part in single:
        moved = part & mask
        if moved:
            endpoints = tuple(sorted(owner for owner in (residual[part], new_single) if owner))
            next_double.append((moved, endpoints))
    return uncovered & ~mask, next_single, tuple(sorted(next_double))

def validate_state(state, m):
    uncovered, single, double = state
    limit = (1 << m)-1
    assert 0 <= uncovered <= limit and single == tuple(sorted(single)) and double == tuple(sorted(double))
    occupied = uncovered
    for part in single:
        assert 0 < part <= limit and not occupied & part
        occupied |= part
    pairs = set()
    for part, endpoints in double:
        assert 0 < part <= limit and not occupied & part
        occupied |= part
        assert endpoints == tuple(sorted(set(endpoints))) and len(endpoints) <= 2
        assert all(owner in single for owner in endpoints)
        if len(endpoints) == 2:
            assert endpoints not in pairs
            pairs.add(endpoints)

def enumerate_states(m):
    first = initial_state(m)
    depths = {first: 0}
    pending = deque([first])
    while pending:
        state = pending.popleft()
        validate_state(state, m)
        for mask in range(1 << m):
            after = transition(state, mask)
            if after not in depths:
                depths[after] = depths[state]+1
                pending.append(after)
    return depths

def _reward(state, m):
    """Worst losses need only positive owner weights; zero padding is harmless."""
    uncovered, single, double = state
    nominal = m - uncovered.bit_count()
    sole = sorted((part.bit_count() for part in single), reverse=True)
    worst_one_loss = sole[0] if sole else 0
    worst_two_loss = sum(sole[:2])
    for part, endpoints in double:
        worst_two_loss = max(worst_two_loss, part.bit_count() + sum(owner.bit_count() for owner in endpoints))
    value = (nominal - worst_two_loss, nominal - worst_one_loss, nominal)
    assert 0 <= value[0] <= value[1] <= value[2] <= m
    return value

@lru_cache(maxsize=5)
def _tables(m):
    states = tuple(enumerate_states(m))
    index = {state: i for i, state in enumerate(states)}
    transitions = np.asarray([[index[transition(state, mask)] for mask in range(1 << m)]
                              for state in states], dtype=np.intp)
    rewards = np.asarray([_reward(state, m) for state in states], dtype=np.float64)
    assert len(states) == (1, 4, 20, 126, 992)[m]
    assert index[initial_state(m)] == 0
    assert np.array_equal(transitions[:, 0], np.arange(len(states)))
    transitions.setflags(write=False)
    rewards.setflags(write=False)
    return states, transitions, rewards
