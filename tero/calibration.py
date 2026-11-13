"""Client-weighted marginal calibration and signed-partition mixture fitting.

These routines contain the numerical fitting steps used by the production
pipeline. They perform no file access, model inference, temporal masking, or
fold assignment. Callers supply unique source observations and keep every
observation from a client in the same training or validation partition.

Marginal observations are source-role pairs. Dependence observations are
sources with four-bit support labels, before temporal eligibility is applied.
Each client receives equal total weight within the supplied training sample.
"""
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .copula import INDEPENDENT, component_observed_likelihood


ROLES = ('baseline', 'goal', 'attempt', 'recent')
EPS = 1e-6
PENALTY = 1e-4
DEPENDENCE_PENALTY = .01
TOLERANCE = 2e-6


def logits(probabilities):
    """Clip input probabilities to [1e-6, 1-1e-6] and return their logits."""
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1-EPS)
    assert np.isfinite(p).all()
    return np.log(p)-np.log1p(-p)


def client_weights(client_ids):
    """Return 1 / (client count * observations for this client) per row."""
    counts = Counter(client_ids)
    n = len(counts)
    assert n > 0
    return np.array([1/(n*counts[cid]) for cid in client_ids])


def loss_gradient(theta, x, y, roles, weights):
    """Stable, unclipped logistic loss with the shared-slope penalty."""
    eta = theta[roles]+theta[4]*x
    unclipped = expit(eta)
    objective = float(np.dot(weights, np.logaddexp(0., eta)-y*eta)+PENALTY*theta[4]**2/2)
    residual = weights*(unclipped-y)
    gradient = np.r_[np.bincount(roles, weights=residual, minlength=4),
                     np.dot(residual, x)+PENALTY*theta[4]]
    return objective, gradient


def predict_probability(probabilities, role_indices, parameters):
    """Apply four role intercepts and a shared nonnegative slope."""
    theta = np.array(parameters['intercepts']+[parameters['shared_slope']], dtype=float)
    assert theta.shape == (5,) and theta[4] >= 0 and np.isfinite(theta).all()
    roles = np.asarray(role_indices, dtype=int)
    assert ((roles >= 0) & (roles < 4)).all()
    return np.clip(expit(theta[roles]+theta[4]*logits(probabilities)), EPS, 1-EPS)


def calibrate(raw_probabilities, role_indices, parameters):
    """Calibrate equally shaped probability and integer-role arrays.

    ``parameters`` is the first return value of ``fit_marginals``. The output
    has the input shape and lies in [1e-6, 1-1e-6]. This function does not zero
    baseline or recent-role probabilities; eligibility belongs to the caller.
    """
    p = np.asarray(raw_probabilities, dtype=float)
    roles = np.asarray(role_indices)
    if (p.shape != roles.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError('Probabilities and role indices must have equal shapes and valid probabilities')
    if roles.dtype.kind not in 'iu' or np.any((roles < 0) | (roles > 3)):
        raise ValueError('Role indices must be integers in 0..3')
    return predict_probability(p, roles, parameters)


def fit_marginals(raw_probabilities, labels, role_indices, client_ids):
    """Fit the fixed five-parameter marginal model and return (parameters, audit).

    Inputs are one-dimensional arrays of the same length: raw probabilities
    in [0,1], binary labels, role indices 0..3, and hashable client identifiers.
    Include each unique source-role observation once, with all four roles
    represented. ``parameters`` contains four ``intercepts`` and one
    ``shared_slope``. A failed numerical convergence check raises AssertionError.
    """
    p = np.asarray(raw_probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    roles = np.asarray(role_indices, dtype=int)
    assert p.ndim == 1 and y.shape == roles.shape == p.shape and len(client_ids) == len(p)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and set(np.unique(y)) <= {0., 1.}
    assert set(np.unique(roles)) == set(range(4))
    weights = client_weights(client_ids)
    x = logits(p)
    prevalence = np.array([np.dot(weights[roles == d], y[roles == d])/weights[roles == d].sum() for d in range(4)])
    start = np.r_[logits(prevalence), 0.]
    result = minimize(loss_gradient, start, args=(x, y, roles, weights), method='L-BFGS-B', jac=True,
                      bounds=[(None, None)]*4+[(0., None)],
                      options={'maxiter':10000, 'ftol':1e-13, 'gtol':1e-9, 'maxls':50})
    objective, gradient = loss_gradient(result.x, x, y, roles, weights)
    projected = gradient.copy()
    if result.x[4] <= 1e-10:
        projected[4] = min(gradient[4], 0.)
    parameters = {'role_order':list(ROLES), 'intercepts':result.x[:4].tolist(),
                  'shared_slope':float(result.x[4]), 'probability_clip':[EPS, 1-EPS],
                  'role_intercepts_penalized':False, 'slope_penalty':PENALTY}
    raw = expit(result.x[roles]+result.x[4]*x)
    audit = {'optimizer':'L-BFGS-B', 'success':bool(result.success), 'message':str(result.message),
             'iterations':int(result.nit), 'observations':len(p), 'clients':len(set(client_ids)),
             'objective':objective, 'weighted_cross_entropy':objective-PENALTY*result.x[4]**2/2,
             'initial_objective':loss_gradient(start, x, y, roles, weights)[0],
             'gradient_infinity_norm':float(np.max(np.abs(gradient))),
             'projected_gradient_infinity_norm':float(np.max(np.abs(projected))),
             'clipped_training_predictions':int(np.sum((raw < EPS) | (raw > 1-EPS))),
             'finite_parameters':bool(np.isfinite(result.x).all()), 'nonnegative_slope':bool(result.x[4] >= 0),
             'weighted_role_prevalence':prevalence.tolist(), 'training_client_weight_max_error':
             max(abs(weights[np.array(client_ids) == cid].sum()-1/len(set(client_ids))) for cid in set(client_ids))}
    audit['converged'] = (audit['success'] and audit['finite_parameters'] and audit['nonnegative_slope']
                          and audit['projected_gradient_infinity_norm'] <= 2e-6)
    assert audit['converged'], audit
    return parameters, audit


def _fit_likelihood(likelihood, client_ids):
    weights = client_weights(client_ids)
    assert np.all(likelihood[:, INDEPENDENT] > 0)
    x = np.full(49, 1/49)

    def objective(x):
        p = likelihood @ x
        if x[INDEPENDENT] <= 0 or np.any(p <= 0):
            return np.inf, np.zeros(49)
        value = -float(weights @ np.log(p))-DEPENDENCE_PENALTY*np.log(x[INDEPENDENT])
        gradient = -(likelihood.T @ (weights/p))
        gradient[INDEPENDENT] -= DEPENDENCE_PENALTY/x[INDEPENDENT]
        return value, gradient

    initial = objective(x)[0]
    for _ in range(200):
        mass = x*(likelihood.T @ (weights/(likelihood @ x)))
        mass[INDEPENDENT] += DEPENDENCE_PENALTY
        x = mass/(1+DEPENDENCE_PENALTY)
    result = minimize(objective, x, jac=True, method='SLSQP',
                      bounds=[(1e-12 if c == INDEPENDENT else 0, 1) for c in range(49)],
                      constraints={'type':'eq', 'fun':lambda w: w.sum()-1, 'jac':lambda w: np.ones(49)},
                      options={'maxiter':5000, 'ftol':1e-13})
    x = np.maximum(result.x, 0)
    x /= x.sum()
    value, gradient = objective(x)
    gap = max(0., float(np.max(-gradient)-(1+DEPENDENCE_PENALTY)))
    audit = {'success':bool(result.success), 'message':str(result.message), 'iterations':int(result.nit),
             'objective':value, 'initial_objective':initial, 'duality_gap_upper_bound':gap,
             'independence_weight':float(x[INDEPENDENT]), 'positive_components':int(np.sum(x > 1e-10)),
             'clients':len(set(client_ids)), 'sources':len(client_ids),
             'training_joint_nll':float(-weights @ np.log(likelihood @ x)),
             'training_independent_joint_nll':float(-weights @ np.log(likelihood[:, INDEPENDENT]))}
    audit['converged'] = bool(np.isfinite(value) and gap <= TOLERANCE
                             and x[INDEPENDENT] >= DEPENDENCE_PENALTY/(1+DEPENDENCE_PENALTY)-TOLERANCE)
    assert audit['converged'], audit
    return x, audit


def fit_dependence(q, masks, client_ids):
    """Return (49 mixture weights, audit) from calibrated source observations.

    ``q`` has shape (n,4). ``masks`` has n integers in 0..15, with bit d equal
    to the raw binary label for role d. ``client_ids`` has n entries. Supply
    each unique source once, before any baseline or temporal masking. The
    independent model must assign positive probability to every observed mask.
    Weights use the exact component order from ``tero.copula.COMPONENTS``.
    """
    probabilities = np.asarray(q, dtype=float)
    labels = np.asarray(masks)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4 or len(probabilities) == 0
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('q must be a nonempty n-by-4 probability array')
    if labels.shape != (len(probabilities),) or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels > 15)):
        raise ValueError('masks must contain one integer in 0..15 per source')
    if len(client_ids) != len(probabilities):
        raise ValueError('client_ids must contain one identifier per source')
    return _fit_likelihood(component_observed_likelihood(probabilities, labels), client_ids)
