"""Constructed example: dependence reverses the preferred three-group set."""
import json
import numpy as np
from tero import COMPONENTS, SignedPartitionMixture, independence, ChronologicalRecoveryOracle, retrieve

full = np.array([[1., .75, .75, 1.]] * 3 + [[1., .95, .20, 1.]] * 3)
sessions = [1] * 6
oracle = ChronologicalRecoveryOracle(sessions, full[:, 3])
positive = np.zeros(49)
positive[next(i for i, (parts, signs) in enumerate(COMPONENTS)
              if len(parts) == 1 and signs == (1, 1, 1, 1))] = 1.
models = {'independent': independence(), 'shared_uniform': SignedPartitionMixture(positive)}
values = {}
for name, model in models.items():
    values[name] = {}
    for label, selected in (('AAA', [0, 1, 2]), ('BBB', [3, 4, 5])):
        q = np.zeros_like(full)
        q[selected] = full[selected]
        values[name][label] = oracle.evaluate(model.joint_law(q, full[:, 3])).tolist()

q = np.zeros_like(full)
q[:3] = full[:3]
model = models['shared_uniform']
laws = model.joint_law(q, full[:, 3])
analysis = oracle.analyze(laws)
modified = q[0].copy()
modified[1] = .5
new_law = model.joint_law(modified[None], full[:1, 3])
fast = oracle.replacement_values(analysis, np.array([0]), new_law)[0]
laws[0] = new_law[0]
np.testing.assert_allclose(fast, oracle.evaluate(laws), atol=1e-12, rtol=0)

result = retrieve(full, [1]*6, sessions, 3, positive, relevance=[1.]*6,
                  initial_sets=[(), (0, 1, 2), (3, 4, 5)])
print(json.dumps({'ownership_states':oracle.ownership_state_count,
    'augmented_states':oracle.state_count, 'value_order':['R2','R1','R0'],
    'constructed_example':values, 'selected_indices':result['selected'],
    'selected_value':result['value'], 'kernel_matches_direct':True}, indent=2))
