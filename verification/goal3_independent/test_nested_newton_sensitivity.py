#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import nested_newton_sensitivity as exact
import nested_validation as nv

class ExactNewtonNestedSensitivityTests(unittest.TestCase):
    def setUp(self):
        exact.FIT_DIAGNOSTICS=[]
        exact.DESIGN_CACHE.clear()
        self.prior=[
            {"date":f"2026-01-{day:02d}","home":home,"away":away,"hg":hg,"ag":ag,"year":2026}
            for day,home,away,hg,ag in [
                (1,"A","B",2,0),(2,"C","D",1,1),(3,"B","C",0,1),(4,"D","A",1,2),
                (5,"A","C",3,1),(6,"B","D",1,0),(7,"C","A",0,0),(8,"D","B",2,2),
                (9,"A","D",1,0),(10,"C","B",2,1),(11,"B","A",1,3),(12,"D","C",0,1),
            ]
        ]

    def test_exact_newton_converges_below_strict_tolerance(self):
        params,idx,info=exact.fit_shrinkage_exact_newton(self.prior,6.0)
        self.assertTrue(info["success"])
        self.assertEqual(info["status"],0)
        self.assertLessEqual(info["maxAbsGradient"],exact.SOLVER_TOLERANCE)
        self.assertTrue(all(map(lambda x:x==x,params)))
        ph,pa=nv.predict_shrinkage(params,idx,"A","B")
        self.assertAlmostEqual(sum(ph),1.0,places=12)
        self.assertAlmostEqual(sum(pa),1.0,places=12)

    def test_exact_newton_reaches_stationary_solution_of_original_objective(self):
        exact_params,exact_idx,exact_info=exact.fit_shrinkage_exact_newton(self.prior,10.0)
        lbfgs_params,lbfgs_idx,_=nv.fit_shrinkage(self.prior,10.0)
        self.assertEqual(exact_idx,lbfgs_idx)
        design=exact._design(self.prior)
        exact_obj,exact_grad,_=exact._objective_gradient_hessian(exact_params,design,10.0,False)
        lbfgs_obj,_,_=exact._objective_gradient_hessian(lbfgs_params,design,10.0,False)
        self.assertLessEqual(exact_obj,lbfgs_obj+1e-10)
        self.assertLessEqual(float(abs(exact_grad).max()),exact.SOLVER_TOLERANCE)

    def test_analytic_hessian_matches_finite_difference_gradient(self):
        design=exact._design(self.prior)
        n=len(design[0])
        x=[0.1,-0.05]+[0.02*(i-n/2) for i in range(2*n)]
        import numpy as np
        x=np.asarray(x,dtype=float)
        _,_,hessian=exact._objective_gradient_hessian(x,design,6.0,True)
        eps=1e-5
        numeric=np.zeros_like(hessian)
        for j in range(len(x)):
            xp=x.copy();xm=x.copy()
            xp[j]+=eps;xm[j]-=eps
            gp=exact._objective_gradient_hessian(xp,design,6.0,False)[1]
            gm=exact._objective_gradient_hessian(xm,design,6.0,False)[1]
            numeric[:,j]=(gp-gm)/(2*eps)
        self.assertLess(float(np.max(np.abs(hessian-numeric))),2e-6)

    def test_empty_prior_preserves_documented_fallback(self):
        params,idx,info=exact.fit_shrinkage_exact_newton([],6.0)
        self.assertIsNone(params)
        self.assertEqual(idx,{})
        self.assertFalse(info["optimizerFit"])
        self.assertEqual(info["status"],0)

if __name__=="__main__":
    unittest.main(verbosity=2)
