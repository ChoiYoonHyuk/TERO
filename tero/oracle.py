"""Exact temporal recovery and candidate-law contractions."""
from functools import lru_cache
from numbers import Integral
import copy
import math
import numpy as np
from .ownership import _tables

LAW_TOLERANCE = 2e-13
ARITHMETIC_TOLERANCE = 5e-12

def reset_recent(state, recent_role):
    """Erase this role while preserving every other role's owner relations."""
    bit = 1 << recent_role
    uncovered, single, double = state
    remaining = {part: part & ~bit for part in single}
    singles = tuple(sorted(part for part in remaining.values() if part))
    doubles = []
    for part, endpoints in double:
        residual = part & ~bit
        if residual:
            owners = tuple(sorted(remaining[owner] for owner in endpoints if remaining[owner]))
            # Different anonymous owner pairs remain different double blocks.
            doubles.append((residual, owners))
    return (uncovered | bit, singles, tuple(sorted(doubles)))

@lru_cache(maxsize=16)
def _chronological_tables(m, recent_role):
    states, add, rewards = _tables(m)
    state_index = {state: i for i, state in enumerate(states)}
    reset = np.asarray([state_index[reset_recent(s, recent_role)] for s in states], dtype=np.intp)
    size, masks = len(states), 1 << m
    transition = np.empty((2 * size, masks * 2), dtype=np.intp)
    for seen in range(2):
        for mask in range(masks):
            for trigger in range(2):
                before = reset if trigger and not seen else np.arange(size)
                transition[seen * size:(seen + 1) * size, 2 * mask + trigger] = (
                    add[before, mask] + (seen | trigger) * size)
    # A boundary clears only the current-session flag, not existing witnesses.
    boundary = np.concatenate((transition[:size], transition[:size]), axis=0)
    terminal = np.concatenate((rewards, rewards), axis=0)
    for array in (reset, transition, boundary, terminal):
        array.setflags(write=False)
    return states, reset, transition, boundary, terminal

class ChronologicalRecoveryOracle:
    """Fixed-universe exact expectations and single-slot replacement kernels.

    Input laws and returned kernels use original group-index order. Sessions
    are processed in ascending order, preserving original order within each.
    Replacing two groups requires a second contraction after one replacement.
    """

    def __init__(self, group_sessions, full_recent_probabilities, m=4, recent_role=3):
        if isinstance(m, bool) or not isinstance(m, Integral) or not 1 <= m <= 4:
            raise ValueError('m must be an integer from one to four')
        if isinstance(recent_role, bool) or not isinstance(recent_role, Integral) or not 0 <= recent_role < m:
            raise ValueError('recent_role must identify one of the m roles')
        sessions = tuple(group_sessions)
        if len(sessions) < 2:
            raise ValueError('The complete original group universe must contain at least two groups')
        if any(isinstance(s, bool) or not isinstance(s, Integral) for s in sessions):
            raise ValueError('Session identifiers must be chronological integers')
        full = np.asarray(full_recent_probabilities, dtype=np.float64)
        if full.shape != (len(sessions),) or not np.isfinite(full).all() or np.any((full < 0) | (full > 1)):
            raise ValueError('Full recent probabilities must have one [0,1] value per original group')
        self.m, self.recent_role = int(m), int(recent_role)
        self.group_sessions = sessions
        self.full_recent_probabilities = full.copy()
        self.full_recent_probabilities.setflags(write=False)
        self.group_count, self.mask_count = len(sessions), 1 << self.m
        self.order = np.asarray(sorted(range(len(sessions)), key=lambda h: sessions[h]), dtype=np.intp)
        ordered_sessions = [sessions[h] for h in self.order]
        self.boundaries = tuple(i == 0 or ordered_sessions[i] != ordered_sessions[i - 1]
                                for i in range(len(sessions)))
        (self.states, self.reset_indices, self.transitions,
         self.boundary_transitions, self.rewards) = _chronological_tables(self.m, self.recent_role)
        self.ownership_state_count = len(self.states)
        self.state_count = len(self.rewards)
        self.last_audit = None

    def _laws(self, P, indices=None):
        P = np.asarray(P, dtype=np.float64)
        ids = np.arange(self.group_count) if indices is None else np.asarray(indices)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= self.group_count)):
            raise ValueError('Group indices must be a one-dimensional integer vector within the original universe')
        if P.shape != (len(ids), self.mask_count, 2):
            raise ValueError('P must have shape (number of specified groups, 2**m, 2)')
        if not np.isfinite(P).all() or np.any((P < 0) | (P > 1 + LAW_TOLERANCE)):
            raise ValueError('Joint laws must be finite and nonnegative')
        if len(ids) and np.max(np.abs(P.sum(axis=(1, 2)) - 1)) > LAW_TOLERANCE:
            raise ValueError('Each joint law must sum to one')
        recent_masks = (np.arange(self.mask_count) & (1 << self.recent_role)) != 0
        if np.any(P[:, recent_masks, 0] != 0):
            raise ValueError('Selected recent support must imply full-history recent support')
        if len(ids) and np.max(np.abs(P[:, :, 1].sum(axis=1) - self.full_recent_probabilities[ids])) > LAW_TOLERANCE:
            raise ValueError('A candidate must preserve the original full-history recent marginal')
        return P

    def _matrix(self, position):
        return self.boundary_transitions if self.boundaries[position] else self.transitions

    def _push(self, mu, law, position):
        flat = law.reshape(-1)
        positive = np.flatnonzero(flat)
        destinations = self._matrix(position)[:, positive]
        weights = mu[:, None] * flat[positive]
        return np.bincount(destinations.ravel(), weights=weights.ravel(), minlength=self.state_count)

    def _pull(self, reward, law, position):
        flat = law.reshape(-1)
        positive = np.flatnonzero(flat)
        return np.einsum('a,sar->sr', flat[positive], reward[self._matrix(position)[:, positive]], optimize=False)

    def _audit(self, mu, P):
        mass = math.fsum(mu.tolist())
        row_sums = P.sum(axis=(1, 2))
        input_mass = math.prod(row_sums.tolist())
        if not np.isfinite(mu).all() or np.any(mu < 0):
            raise ArithmeticError('Invalid propagated state probabilities')
        if (abs(mass - input_mass) > ARITHMETIC_TOLERANCE or
                abs(mass - 1) > ARITHMETIC_TOLERANCE + len(P) * LAW_TOLERANCE):
            raise ArithmeticError('Propagated probability mass failed its audit')
        result = {'groups': self.group_count, 'sessions': len(set(self.group_sessions)),
                  'ownership_states': self.ownership_state_count, 'augmented_states': self.state_count,
                  'probability_mass': mass, 'input_product_mass': input_mass,
                  'mass_error': abs(mass - 1), 'positive_states': int(np.count_nonzero(mu)),
                  'input_maximum_mass_error': float(np.max(np.abs(row_sums - 1))),
                  'positive_mass_pruned': False, 'probabilities_renormalized': False,
                  'unselected_groups_retained': True, 'latest_session_recomputed_after_deletion': False,
                  'value_order': ['expected_worst_two', 'expected_worst_one', 'expected_nominal']}
        self.last_audit = result
        return result

    def distribution(self, P):
        P = self._laws(P)
        mu = np.zeros(self.state_count)
        mu[0] = 1.
        for position, group in enumerate(self.order):
            mu = self._push(mu, P[group], position)
        self._audit(mu, P)
        return mu

    def evaluate(self, P):
        """Return exact-model [E worst-two, E worst-one, E nominal]."""
        return self.distribution(P) @ self.rewards

    def analyze(self, P):
        """Return kernels for replacing any one original group law.

        A new selected group replaces its unselected law at its existing slot.
        The full recent marginal stays fixed. Two distinct group changes are
        not a single replacement and require an additional exact analysis.
        """
        P = self._laws(P)
        H = self.group_count
        forward = np.zeros((H + 1, self.state_count))
        forward[0, 0] = 1.
        for position, group in enumerate(self.order):
            forward[position + 1] = self._push(forward[position], P[group], position)
        audit = self._audit(forward[-1], P)
        suffix = np.empty((H + 1, self.state_count, 3))
        suffix[H] = self.rewards
        for position in range(H - 1, -1, -1):
            suffix[position] = self._pull(suffix[position + 1], P[self.order[position]], position)
        kernel = np.empty((H, self.mask_count, 2, 3))
        for position, group in enumerate(self.order):
            kernel[group] = np.einsum('s,sar->ar', forward[position],
                                      suffix[position + 1][self._matrix(position)],
                                      optimize=False).reshape(self.mask_count, 2, 3)
        value = forward[-1] @ self.rewards
        if not np.allclose(value, suffix[0, 0], rtol=0, atol=ARITHMETIC_TOLERANCE):
            raise ArithmeticError('Forward/backward expectation mismatch')
        return {'value': value, 'replacement_kernel': kernel, 'audit': audit}

    def replacement_values(self, analysis, group_indices, candidate_laws):
        """Contract a batch of alternative single-slot laws, allowing repeated IDs."""
        ids = np.asarray(group_indices)
        laws = self._laws(candidate_laws, ids)
        kernels = np.asarray(analysis['replacement_kernel'])
        if kernels.shape != (self.group_count, self.mask_count, 2, 3):
            raise ValueError('Replacement kernels have an incompatible shape')
        return np.einsum('baz,bazr->br', laws, kernels[ids], optimize=False)

class DependentTemporalRecovery:
    def __init__(self, full_group_q, group_sessions, mixture, *, oracle_class=ChronologicalRecoveryOracle):
        full = np.array(full_group_q, dtype=np.float64, copy=True)
        if full.ndim != 2 or full.shape[1] != 4 or len(full) < 2:
            raise ValueError('The complete universe requires H>=2 groups and four roles')
        if not np.isfinite(full).all() or np.any((full < 0) | (full > 1)):
            raise ValueError('Full-group probabilities must be finite and in [0,1]')
        self.full_group_q = full
        self.full_group_q.setflags(write=False)
        self.group_count = len(full)
        # Own the model weights so caller mutation cannot invalidate exact-law
        # memoization. The model class must implement the fixed joint-law API.
        self.mixture = copy.copy(mixture)
        self.mixture.weights = np.array(mixture.weights, dtype=np.float64, copy=True)
        self.mixture.weights.setflags(write=False)
        self.oracle = oracle_class(group_sessions, full[:, 3])
        if self.oracle.group_count != self.group_count:
            raise ValueError('Every full group must have exactly one session')
        self._law_cache = {}
        zero = np.zeros((self.group_count, 4))
        self._unselected_laws = self._joint_laws(tuple(range(self.group_count)), zero)
        self._unselected_laws.setflags(write=False)
        self.last_audit = None

    def _ids_q(self, ids, probabilities, unique=True, scalar_allowed=False):
        scalar = isinstance(ids, Integral) and not isinstance(ids, bool)
        if scalar and not scalar_allowed:
            raise ValueError('Selected group IDs must be a sequence')
        values = [ids] if scalar else list(ids)
        if any(isinstance(h, bool) or not isinstance(h, Integral) or not 0 <= h < self.group_count for h in values):
            raise ValueError('Group IDs must index the fixed original universe')
        values = tuple(map(int, values))
        if unique and len(set(values)) != len(values):
            raise ValueError('Selected group IDs must be unique')
        q = np.array(probabilities, dtype=np.float64, copy=True)
        if scalar and q.shape == (4,):
            q = q.reshape(1, 4)
        elif not values and q.ndim == 1 and q.size == 0:
            q = q.reshape(0, 4)
        if q.shape != (len(values), 4) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError('Selected probabilities must have shape (groups,4) and range [0,1]')
        if np.any(q > self.full_group_q[list(values)]):
            raise ValueError('Selected probabilities exceed the full-group nested bound')
        return values, q, scalar

    def _joint_laws(self, ids, q):
        keys = [(h, q[j].tobytes()) for j, h in enumerate(ids)]
        missing = {}
        for j, key in enumerate(keys):
            if key not in self._law_cache:
                missing.setdefault(key, j)
        if missing:
            positions = list(missing.values())
            group_ids = [ids[j] for j in positions]
            laws = self.mixture.joint_law(q[positions], self.full_group_q[group_ids, 3])
            self.oracle._laws(laws, np.asarray(group_ids, dtype=np.intp))
            for key, law in zip(missing, laws):
                law = law.copy()
                law.setflags(write=False)
                self._law_cache[key] = law
        return np.asarray([self._law_cache[key] for key in keys], dtype=np.float64).reshape(len(ids), 16, 2)

    def _full_laws(self, ids, q):
        laws = self._unselected_laws.copy()
        if ids:
            laws[list(ids)] = self._joint_laws(ids, q)
        return laws

    def evaluate(self, selected_group_ids, selected_q):
        ids, q, _ = self._ids_q(selected_group_ids, selected_q)
        value = self.oracle.evaluate(self._full_laws(ids, q))
        self.last_audit = dict(self.oracle.last_audit, selected_groups=len(ids),
                               cross_role_dependence_allowed=True,
                               law_memoization='exact group ID and float64 probability bytes')
        return value

    def analyze(self, selected_group_ids, selected_q):
        ids, q, _ = self._ids_q(selected_group_ids, selected_q)
        selected = set(ids)
        full_laws = self._full_laws(ids, q)
        analysis = self.oracle.analyze(full_laws)
        removal_analyses = {}

        def add(candidate_ids, candidate_q):
            new_ids, new_q, scalar = self._ids_q(candidate_ids, candidate_q, unique=False, scalar_allowed=True)
            if any(h in selected for h in new_ids):
                raise ValueError('Addition kernels require previously unselected groups')
            values = self.oracle.replacement_values(analysis, np.asarray(new_ids, dtype=np.intp),
                                                    self._joint_laws(new_ids, new_q))
            return values[0] if scalar else values

        def replace(position, candidate_ids, candidate_q):
            if isinstance(position, bool) or not isinstance(position, Integral) or not 0 <= position < len(ids):
                raise ValueError('Replacement position must index the selected-group array')
            old_id = ids[position]
            new_ids, new_q, scalar = self._ids_q(candidate_ids, candidate_q, unique=False, scalar_allowed=True)
            if any(h in selected and h != old_id for h in new_ids):
                raise ValueError('Cross-group replacement requires an absent destination group')
            laws = self._joint_laws(new_ids, new_q)
            values = np.empty((len(new_ids), 3))
            same = [j for j, h in enumerate(new_ids) if h == old_id]
            different = [j for j, h in enumerate(new_ids) if h != old_id]
            if same:
                values[same] = self.oracle.replacement_values(analysis,
                    np.asarray([new_ids[j] for j in same], dtype=np.intp), laws[same])
            if different:
                if old_id not in removal_analyses:
                    removed = full_laws.copy()
                    removed[old_id] = self._unselected_laws[old_id]
                    removal_analyses[old_id] = self.oracle.analyze(removed)
                values[different] = self.oracle.replacement_values(removal_analyses[old_id],
                    np.asarray([new_ids[j] for j in different], dtype=np.intp), laws[different])
            return values[0] if scalar else values

        audit = dict(analysis['audit'], selected_groups=len(ids),
                     cross_role_dependence_allowed=True,
                     cross_group_exchange_requires_one_analysis_per_removed_group=True)
        self.last_audit = audit
        return {'value': analysis['value'], 'add': add, 'replace': replace, 'audit': audit}
