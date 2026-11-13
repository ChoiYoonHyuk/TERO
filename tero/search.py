"""Greedy and exchange search driven by exact recovery expectations."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
import numpy as np
from .copula import SignedPartitionMixture
from .oracle import DependentTemporalRecovery

TRANSFORMATIONS = ("two_first", "mean_recovery")

@dataclass
class Objective:
    evaluate: Callable[[np.ndarray], np.ndarray]
    analyze: Callable[[np.ndarray], dict] | None = None
    name: str = "expected_recovery"

class Search:
    def __init__(self, probabilities, costs, budget, objective: Objective,
                 groups=None, relevance=None, decimals=11):
        self.q = np.asarray(probabilities, dtype=float)
        if self.q.ndim != 2 or not 1 <= self.q.shape[1] <= 4:
            raise ValueError("Expected a source-by-role matrix with one to four roles")
        self.n, self.m = self.q.shape
        if not np.isfinite(self.q).all() or np.any(self.q < 0) or np.any(self.q > 1):
            raise ValueError("Invalid support probabilities")
        values = tuple(costs)
        if len(values) != self.n or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in values):
            raise ValueError("One positive integer token cost per source is required")
        if isinstance(budget, bool) or int(budget) != budget or budget < 0:
            raise ValueError("Invalid token budget")
        self.costs, self.budget = np.asarray(values, dtype=np.int64), int(budget)
        labels = list(range(self.n)) if groups is None else list(groups)
        if len(labels) != self.n:
            raise ValueError("One group label per source is required")
        mapping = {}
        self.groups = np.asarray([mapping.setdefault(g, len(mapping)) for g in labels], dtype=int)
        self.objective, self.decimals = objective, int(decimals)
        self.tolerance = 10. ** (-self.decimals)
        self.relevance = np.zeros(self.n) if relevance is None else np.asarray(relevance, dtype=float)
        if self.relevance.shape != (self.n,) or not np.isfinite(self.relevance).all():
            raise ValueError("Invalid relevance vector")
        self.evaluations = self.analyses = self.neighbors = 0
        self.canonical_values = {}

    def normalize(self, selected):
        indices = tuple(sorted(selected))
        if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < self.n for i in indices):
            raise ValueError("Invalid source selection")
        if self.cost(indices) > self.budget:
            raise ValueError("Selection exceeds budget")
        return indices

    def cost(self, selected):
        return int(sum(int(self.costs[i]) for i in selected))

    def group_state(self, selected):
        members = {}
        for i in selected:
            members.setdefault(int(self.groups[i]), []).append(i)
        labels = sorted(members)
        q = np.asarray([self.q[members[h]].max(axis=0) for h in labels], dtype=float).reshape(-1, self.m)
        return labels, members, q


    def upper_value(self, value):
        # Nonnegative contractions can differ in their final floating-point
        # rounding. The interval screens candidates, while every accepted
        # comparison uses one canonical, directly evaluated subset value.
        value = np.asarray(value, dtype=float)
        return value + 1e-10 * np.maximum(np.abs(value), 1e-12)

    def canonical_candidate(self, selected, estimate):
        value = self.evaluate(selected)
        guard = 1e-10 * np.maximum(np.maximum(np.abs(estimate), np.abs(value)), 1e-12)
        if np.any(np.abs(value - estimate) > guard):
            raise AssertionError("Kernel discrepancy exceeds the comparison guard")
        return value

    def key(self, value, selected):
        return tuple(np.round(value, self.decimals)) + (-self.cost(selected), -len(selected), tuple(-i for i in selected))



    def greedy(self, selected=()):
        selected = self.normalize(selected)
        while True:
            remaining = self.budget - self.cost(selected)
            candidates = [i for i in range(self.n) if i not in selected and self.costs[i] <= remaining]
            if not candidates:
                return selected
            baseline = self.evaluate(selected)
            values = self.additions(selected, candidates)
            gains = values - baseline
            if np.any(gains < -1e-8):
                raise AssertionError("Adding a source reduced a monotone recovery objective")
            def gain_key(value, candidate):
                gain = value - baseline
                return (tuple(np.round(np.maximum(gain, 0.) / self.costs[candidate], self.decimals)),
                        tuple(np.round(gain, self.decimals)), -int(self.costs[candidate]), -candidate)
            best, best_key, best_value = None, None, None
            for j, candidate in enumerate(candidates):
                if best_key is not None and gain_key(self.upper_value(values[j]), candidate) <= best_key:
                    continue
                indices = tuple(sorted((*selected, candidate)))
                exact = self.canonical_candidate(indices, values[j])
                key = gain_key(exact, candidate)
                if best_key is None or key > best_key:
                    best, best_key, best_value = candidate, key, exact
            if np.max(best_value - baseline) <= self.tolerance:
                return selected
            selected = tuple(sorted((*selected, best)))

    def complete_relevance(self, selected):
        result = set(selected)
        used = self.cost(selected)
        for i in sorted(range(self.n), key=lambda i: (-self.relevance[i], int(self.costs[i]), i)):
            if i not in result and used + self.costs[i] <= self.budget:
                result.add(i)
                used += int(self.costs[i])
        return tuple(sorted(result))


def transform_values(values, transformation):
    """Transform the last axis of a scalar or batch of three-value vectors."""
    if transformation not in TRANSFORMATIONS:
        raise ValueError("Unknown recovery transformation")
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite recovery vectors with final dimension three")
    if transformation == "two_first":
        return array.copy()
    return np.stack((array.mean(axis=-1), array[..., 0], array[..., 2]), axis=-1)

@dataclass
class ContextObjective:
    evaluate: Callable
    analyze: Callable | None = None
    name: str = 'context_recovery'

class ContextSearch(Search):
    def __init__(self, probabilities, costs, budget, objective, *, groups=None, relevance=None, decimals=11):
        def unbound(_):
            raise AssertionError('A context-dependent oracle requires original group identities')
        super().__init__(probabilities, costs, budget, Objective(unbound), groups, relevance, decimals)
        self.context_objective = objective

    def evaluate(self, selected):
        selected = tuple(sorted(selected))
        if selected in self.canonical_values:
            return self.canonical_values[selected].copy()
        self.evaluations += 1
        labels, _, q = self.group_state(selected)
        value = np.asarray(self.context_objective.evaluate(labels, q), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise AssertionError('Context objective returned invalid values')
        self.canonical_values[selected] = value.copy()
        return value.copy()

    def analysis(self, selected):
        labels, members, q = self.group_state(selected)
        if self.context_objective.analyze is None:
            return labels, members, None
        self.analyses += 1
        return labels, members, self.context_objective.analyze(labels, q)

    def additions(self, selected, candidates, analysis=None):
        if not candidates:
            return np.empty((0, 3))
        labels, members, result = analysis if analysis is not None else self.analysis(selected)
        if result is None:
            return np.asarray([self.evaluate(tuple(sorted((*selected, i)))) for i in candidates])
        position = {h: j for j, h in enumerate(labels)}
        output = np.empty((len(candidates), 3))
        absent = [j for j, i in enumerate(candidates) if int(self.groups[i]) not in position]
        if absent:
            sources = [candidates[j] for j in absent]
            output[absent] = result['add'](self.groups[sources].tolist(), self.q[sources])
        for h, k in position.items():
            positions = [j for j, i in enumerate(candidates) if int(self.groups[i]) == h]
            if positions:
                sources = [candidates[j] for j in positions]
                updated = np.maximum(self.q[members[h]].max(axis=0), self.q[sources])
                output[positions] = result['replace'](k, [h] * len(sources), updated)
        return output

    def exchange(self, selected, max_exchanges):
        selected = self.normalize(selected)
        visited = {selected}
        steps = 0
        while True:
            value = self.evaluate(selected)
            best, best_key = selected, self.key(value, selected)
            labels, members, analysis = self.analysis(selected)
            position = {h: j for j, h in enumerate(labels)}
            outside = [i for i in range(self.n) if i not in selected]
            remaining = self.budget - self.cost(selected)
            additions = [i for i in outside if self.costs[i] <= remaining]
            for i, after in zip(additions, self.additions(selected, additions, (labels, members, analysis))):
                candidate = tuple(sorted((*selected, i)))
                self.neighbors += 1
                if self.key(self.upper_value(after), candidate) <= best_key:
                    continue
                after = self.canonical_candidate(candidate, after)
                key = self.key(after, candidate)
                if key > best_key:
                    best, best_key = candidate, key
            for removed in selected:
                h = int(self.groups[removed])
                candidates = [i for i in outside if self.costs[i] <= remaining + self.costs[removed]]
                remainder = tuple(i for i in selected if i != removed)
                if analysis is not None and len(members[h]) == 1:
                    available = [i for i in candidates if int(self.groups[i]) not in position or int(self.groups[i]) == h]
                    values = analysis['replace'](position[h], self.groups[available].tolist(), self.q[available]) if available else []
                    evaluated = dict(zip(available, values))
                else:
                    evaluated = {}
                for i in candidates:
                    candidate = tuple(sorted((*remainder, i)))
                    after = evaluated.get(i)
                    if after is None:
                        after = self.evaluate(candidate)
                    self.neighbors += 1
                    if self.key(self.upper_value(after), candidate) <= best_key:
                        continue
                    after = self.canonical_candidate(candidate, after)
                    key = self.key(after, candidate)
                    if key > best_key:
                        best, best_key = candidate, key
            if best == selected:
                return selected, steps, True
            if steps >= max_exchanges:
                return selected, steps, False
            selected = best
            if selected in visited:
                raise AssertionError('A strict canonical comparison revisited a selection')
            visited.add(selected)
            steps += 1

def select_context(probabilities, costs, budget, objective, *, groups=None, initial_sets=((),),
                   relevance=None, max_exchanges=32, decimals=11, completion=True):
    began = perf_counter()
    engine = ContextSearch(probabilities, costs, budget, objective, groups=groups, relevance=relevance, decimals=decimals)
    starts = sorted(set(engine.normalize(s) for s in initial_sets))
    if not starts:
        raise ValueError('At least one initial selection is required')
    candidates = [engine.greedy(s) for s in starts]
    chosen = max(candidates, key=lambda s: engine.key(engine.evaluate(s), s))
    initial_value = engine.evaluate(chosen)
    core, steps, local_optimal = engine.exchange(chosen, max_exchanges)
    selected = engine.complete_relevance(core) if completion else core
    core_value, value = engine.evaluate(core), engine.evaluate(selected)
    if np.any(value < core_value - 1e-8):
        raise AssertionError('Completion reduced expected recovery')
    return dict(selected=list(selected), core=list(core), cost=engine.cost(selected), core_cost=engine.cost(core),
        value=value.tolist(), core_value=core_value.tolist(), initial_value=initial_value.tolist(),
        initial_sets=[list(s) for s in starts], greedy_sets=[list(s) for s in candidates], exchanges=steps,
        neighborhood_optimal_before_completion=local_optimal, global_selection_optimality_claimed=False,
        objective=objective.name, objective_evaluations=engine.evaluations, batched_analyses=engine.analyses,
        neighbors_evaluated=engine.neighbors, elapsed_seconds=perf_counter()-began,
        original_group_id_order='first occurrence in the complete candidate source list')

def full_group_context(probabilities, group_labels, source_sessions):
    q = np.asarray(probabilities, dtype=float)
    if len(group_labels) != len(q) or len(source_sessions) != len(q):
        raise ValueError('Group and session arrays must cover the complete source pool')
    mapping, members = {}, {}
    for i, label in enumerate(group_labels):
        h = mapping.setdefault(label, len(mapping))
        members.setdefault(h, []).append(i)
    sessions = []
    for h in range(len(members)):
        session = {source_sessions[i] for i in members[h]}
        if len(session) != 1:
            raise ValueError('A failure group must be contained in a single session')
        sessions.append(session.pop())
    return np.asarray([q[members[h]].max(axis=0) for h in range(len(members))]), sessions

def transformed_context(raw, family, transformation):
    def transform(value):
        value = np.asarray(value, dtype=float)
        if family == 'nominal':
            return np.stack((value[..., 2], np.zeros(value.shape[:-1]), np.zeros(value.shape[:-1])), axis=-1)
        return transform_values(value, transformation)
    if family == 'incidence':
        evaluate = raw.evaluate
        analyze = raw.analyze
    else:
        evaluate = lambda ids, q: raw.evaluate(family, ids, q)
        analyze = lambda ids, q: raw.analyze(family, ids, q)
    def wrapped_analysis(ids, q):
        result = analyze(ids, q)
        return dict(value=transform(result['value']),
                    add=lambda new_ids, new_q: transform(result['add'](new_ids, new_q)),
                    replace=lambda k, new_ids, new_q: transform(result['replace'](k, new_ids, new_q)))
    return ContextObjective(lambda ids, q: transform(evaluate(ids, q)), wrapped_analysis,
                            'temporal_' + family + ('' if family == 'nominal' else ':' + transformation))

def retrieve(probabilities, costs, sessions, budget, weights, *, relevance,
             groups=None, initial_sets=((),), utility="two_first",
             max_exchanges=32, completion=True):
    """Select block indices from calibrated probabilities and supplied starts.

    Role order is baseline, goal, attempt, recent. Group labels may denote
    individual blocks or sessions; each group must lie in one session.
    Relevance is the separate raw relevance score, not mean role support.
    """
    q = np.array(probabilities, dtype=float, copy=True)
    sessions = np.asarray(sessions)
    if q.ndim != 2 or q.shape[1] != 4 or sessions.shape != (len(q),):
        raise ValueError("Use an N-by-4 probability matrix and N session indices")
    if not np.issubdtype(sessions.dtype, np.integer) or np.any(sessions < 1):
        raise ValueError("Sessions must be positive integers starting at one")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Support probabilities must be finite and in [0,1]")
    q[sessions != 1, 0] = 0
    labels = list(range(len(q))) if groups is None else list(groups)
    full, group_sessions = full_group_context(q, labels, sessions)
    raw = DependentTemporalRecovery(full, group_sessions, SignedPartitionMixture(weights))
    objective = transformed_context(raw, "incidence", utility)
    return select_context(q, costs, budget, objective, groups=labels,
        initial_sets=initial_sets, relevance=relevance, max_exchanges=max_exchanges,
        decimals=11, completion=completion)
