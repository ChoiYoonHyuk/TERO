"""Finite mixtures of signed-partition copulas with unchanged role marginals.

Every partition block owns an independent uniform latent variable. Each role
uses that variable or its reflection. Mixture weights are nonnegative. Shared
latents couple all selected subsets to the same complete-pool realization.
"""
from itertools import product
import numpy as np


def partitions(n):
    def rec(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for j in range(len(blocks)):
            blocks[j].append(i)
            yield from rec(i + 1, blocks)
            blocks[j].pop()
        blocks.append([i])
        yield from rec(i + 1, blocks)
        blocks.pop()
    yield from rec(0, [])


def components(n=4):
    result = []
    for partition in partitions(n):
        variable_roles = [r for block in partition for r in block[1:]]
        for signs in product((1, -1), repeat=len(variable_roles)):
            orientation = [1] * n
            for r, sign in zip(variable_roles, signs):
                orientation[r] = sign
            result.append((partition, tuple(orientation)))
    return tuple(result)


COMPONENTS = components()
INDEPENDENT = next(i for i, (partition, _) in enumerate(COMPONENTS) if len(partition) == 4)
assert len(COMPONENTS) == 49


def _probabilities(q):
    q = np.asarray(q, dtype=float)
    assert q.ndim == 2 and q.shape[1] == 4
    assert np.isfinite(q).all() and np.logical_and(q >= 0, q <= 1).all()
    return q


def component_observed_likelihood(q, masks):
    """N by 49 exact probabilities for N observed four-bit masks."""
    q = _probabilities(q)
    masks = np.asarray(masks, dtype=int)
    assert masks.shape == (len(q),) and ((masks >= 0) & (masks < 16)).all()
    bits = (masks[:, None] >> np.arange(4)) & 1
    result = np.ones((len(q), len(COMPONENTS)))
    for c, (partition, signs) in enumerate(COMPONENTS):
        for block in partition:
            lower, upper = np.zeros(len(q)), np.ones(len(q))
            for r in block:
                threshold = q[:, r] if signs[r] == 1 else 1 - q[:, r]
                below = bits[:, r] == (1 if signs[r] == 1 else 0)
                upper = np.minimum(upper, np.where(below, threshold, 1))
                lower = np.maximum(lower, np.where(below, 0, threshold))
            result[:, c] *= np.maximum(upper - lower, 0)
    return result


class SignedPartitionMixture:
    def __init__(self, weights):
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == (49,) and np.isfinite(weights).all()
        assert (weights >= 0).all() and abs(weights.sum() - 1) < 1e-10
        self.weights = weights / weights.sum()

    def observed_likelihood(self, q, masks):
        return component_observed_likelihood(q, masks) @ self.weights

    def joint_law(self, q, full_recent_probability):
        """P[selected mask, full-pool recent bit], shaped (N,16,2).

        For each group q contains maximum selected probabilities, including zero
        in an empty group. full_recent_probability is its complete-pool maximum.
        All groups, including unselected groups, must be supplied to the oracle.
        """
        q = _probabilities(q)
        full = np.asarray(full_recent_probability, dtype=float)
        assert full.shape == (len(q),) and np.isfinite(full).all()
        assert ((full >= q[:, 3]) & (full <= 1)).all()
        bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)[None, :, None, :]
        z = np.arange(2)[None, None, :]
        result = np.zeros((len(q), 16, 2))
        for weight, (partition, signs) in zip(self.weights, COMPONENTS):
            if weight == 0:
                continue
            mass = np.ones_like(result)
            for block in partition:
                lower, upper = np.zeros_like(result), np.ones_like(result)
                for r in block:
                    threshold = (q[:, r] if signs[r] == 1 else 1 - q[:, r])[:, None, None]
                    below = bits[..., r] == (1 if signs[r] == 1 else 0)
                    upper = np.minimum(upper, np.where(below, threshold, 1))
                    lower = np.maximum(lower, np.where(below, 0, threshold))
                if 3 in block:
                    threshold = (full if signs[3] == 1 else 1 - full)[:, None, None]
                    below = z == (1 if signs[3] == 1 else 0)
                    upper = np.minimum(upper, np.where(below, threshold, 1))
                    lower = np.maximum(lower, np.where(below, 0, threshold))
                mass *= np.maximum(upper - lower, 0)
            result += weight * mass
        assert np.max(np.abs(result.sum(axis=(1, 2)) - 1), initial=0) < 1e-12
        return result


def independence():
    w = np.zeros(49)
    w[INDEPENDENT] = 1
    return SignedPartitionMixture(w)
