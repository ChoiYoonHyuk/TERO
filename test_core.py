"""Small exact-enumeration and search-equivalence checks; synthetic inputs only."""
from itertools import combinations, product
import unittest
import numpy as np
from tero import COMPONENTS, ChronologicalRecoveryOracle, SignedPartitionMixture, independence, retrieve
from tero.oracle import DependentTemporalRecovery
from tero.ownership import _tables
from tero.search import ContextObjective, full_group_context, select_context, transformed_context


def literal_expectation(laws, sessions):
    choices = [[(a,z,p[a,z]) for a,z in product(range(16),range(2)) if p[a,z]>0] for p in laws]
    answer = np.zeros(3)
    for world in product(*choices):
        mass = np.prod([x[2] for x in world])
        latest = max((sessions[h] for h,x in enumerate(world) if x[1]), default=0)
        masks = [x[0] if sessions[h]==latest else x[0]&~8 for h,x in enumerate(world)]
        for coordinate,j in enumerate((2,1,0)):
            worst = 4
            for size in range(j+1):
                for deleted in combinations(range(len(laws)),size):
                    union = 0
                    for h,mask in enumerate(masks):
                        if h not in deleted: union |= mask
                    worst = min(worst,union.bit_count())
            answer[coordinate] += mass*worst
    return answer


class CoreChecks(unittest.TestCase):
    def test_state_counts(self):
        self.assertEqual([len(_tables(m)[0]) for m in range(1,5)],[4,20,126,992])
        self.assertEqual(ChronologicalRecoveryOracle([1,2],[0.,0.]).state_count,1984)

    def test_mixture_and_literal_recovery(self):
        q=np.array([[.6,.8,0.,.2],[0.,.3,.7,0.],[0.,0.,1.,.6]])
        full=np.array([.4,.5,.8]); sessions=[1,1,2]
        model=SignedPartitionMixture(np.full(49,1/49))
        laws=model.joint_law(q,full)
        bits=(np.arange(16)[:,None]>>np.arange(4))&1
        np.testing.assert_allclose(laws.sum(axis=(1,2)),1.,atol=1e-13)
        np.testing.assert_allclose(laws.sum(axis=2)@bits,q,atol=1e-13)
        np.testing.assert_allclose(laws[:,:,1].sum(axis=1),full,atol=1e-13)
        oracle=ChronologicalRecoveryOracle(sessions,full)
        np.testing.assert_allclose(oracle.evaluate(laws),literal_expectation(laws,sessions),atol=1e-12,rtol=0)
        candidate=model.joint_law(q[[0,2]]*.5,full[[0,2]])
        fast=oracle.replacement_values(oracle.analyze(laws),np.array([0,2]),candidate)
        for k,h in enumerate((0,2)):
            altered=laws.copy(); altered[h]=candidate[k]
            np.testing.assert_allclose(fast[k],oracle.evaluate(altered),atol=1e-12,rtol=0)

    def test_unselected_recent_trigger(self):
        q=np.array([[0.,0.,0.,1.],[0.,0.,0.,0.]])
        model=independence()
        for full,expected in (([1.,0.],1.),([1.,1.],0.)):
            oracle=ChronologicalRecoveryOracle([1,2],full)
            self.assertAlmostEqual(oracle.evaluate(model.joint_law(q,full))[2],expected)

    def test_search_with_and_without_kernels(self):
        q=np.array([[1.,.7,.2,.2],[.5,.4,.8,.3],[0.,.8,.3,.5],
                    [0.,.4,.9,.7],[0.,.5,.8,.8],[0.,.9,.7,.6]])
        costs=[2,1,2,1,2,1]; sessions=[1,1,2,2,3,3]
        model=SignedPartitionMixture(np.full(49,1/49))
        for groups in ([0,1,2,3,4,5],sessions):
            full,gs=full_group_context(q,groups,sessions)
            raw=DependentTemporalRecovery(full,gs,model)
            accelerated=transformed_context(raw,'incidence','two_first')
            direct=ContextObjective(accelerated.evaluate,None,accelerated.name)
            kwargs=dict(groups=groups,initial_sets=[(),(0,3)],relevance=[.8,.9,.5,.7,.2,.6])
            left=select_context(q,costs,5,accelerated,**kwargs)
            right=select_context(q,costs,5,direct,**kwargs)
            for field in ('selected','core','cost','core_cost','exchanges','greedy_sets','neighborhood_optimal_before_completion'):
                self.assertEqual(left[field],right[field])
            np.testing.assert_allclose(left['value'],right['value'],atol=1e-12,rtol=0)

    def test_ranking_reversal(self):
        full=np.array([[1.,.75,.75,1.]]*3+[[1.,.95,.20,1.]]*3)
        w=np.zeros(49)
        w[next(i for i,(p,s) in enumerate(COMPONENTS) if len(p)==1 and s==(1,1,1,1))]=1.
        oracle=ChronologicalRecoveryOracle([1]*6,full[:,3])
        for model,preferred in ((independence(),'A'),(SignedPartitionMixture(w),'B')):
            scores=[]
            for selected in ([0,1,2],[3,4,5]):
                q=np.zeros_like(full); q[selected]=full[selected]
                scores.append(oracle.evaluate(model.joint_law(q,full[:,3]))[0])
            self.assertEqual('A' if scores[0]>scores[1] else 'B',preferred)


if __name__=='__main__':
    unittest.main(verbosity=2)
