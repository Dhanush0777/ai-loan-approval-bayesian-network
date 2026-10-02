"""
Unit tests for the Bayesian Network Loan Approval Model
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from bayesian_model import LoanBayesianModel


class TestLoanBayesianModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LoanBayesianModel()

    def test_model_validity(self):
        """Verify graph is valid DAG and CPDs are consistent."""
        self.assertTrue(self.model.model.check_model())
        self.assertEqual(len(self.model.model.nodes()), 12)
        self.assertEqual(len(self.model.model.edges()), 11)

    def test_applicant_1_prime(self):
        """Prime applicant should have high approval probability and low risk."""
        app1 = {
            'monthly_income': 70000,
            'credit_score': 780,
            'employment_status': 'Salaried',
            'existing_loans': 'No',
            'loan_amount': 500000,
            'repayment_history': 'Excellent',
            'employment_duration': 5,
            'existing_monthly_debt': 10000
        }
        res = self.model.predict(app1)
        print("\nApplicant 1 Result:", res['approval_probability'], "% Approval, Risk:", res['risk_level'])
        
        # Must have high approval probability
        self.assertGreater(res['approval_probability'], 80.0)
        self.assertLess(res['rejection_probability'], 20.0)
        self.assertEqual(res['risk_level'], 'LOW')
        self.assertAlmostEqual(res['approval_probability'] + res['rejection_probability'], 100.0, places=1)
        self.assertTrue(any(f['type'] == 'positive' for f in res['explanations']))

    def test_applicant_2_high_risk(self):
        """High risk applicant should have high rejection probability and high risk tier."""
        app2 = {
            'monthly_income': 30000,
            'credit_score': 580,
            'employment_status': 'Self-employed',
            'existing_loans': 'Yes',
            'loan_amount': 1000000,
            'repayment_history': 'Poor',
            'employment_duration': 1,
            'existing_monthly_debt': 16000
        }
        res = self.model.predict(app2)
        print("Applicant 2 Result:", res['approval_probability'], "% Approval, Risk:", res['risk_level'])
        
        # Must have high rejection probability
        self.assertLess(res['approval_probability'], 35.0)
        self.assertGreater(res['rejection_probability'], 65.0)
        self.assertEqual(res['risk_level'], 'HIGH')
        self.assertAlmostEqual(res['approval_probability'] + res['rejection_probability'], 100.0, places=1)
        self.assertTrue(any(f['type'] == 'warning' for f in res['explanations']))

    def test_applicant_3_moderate(self):
        """Moderate applicant should yield balanced/intermediate probability."""
        app3 = {
            'monthly_income': 50000,
            'credit_score': 700,
            'employment_status': 'Salaried',
            'existing_loans': 'Yes',
            'loan_amount': 400000,
            'repayment_history': 'Good',
            'employment_duration': 3,
            'existing_monthly_debt': 14000
        }
        res = self.model.predict(app3)
        print("Applicant 3 Result:", res['approval_probability'], "% Approval, Risk:", res['risk_level'])
        
        self.assertGreater(res['approval_probability'], 45.0)
        self.assertLess(res['approval_probability'], 85.0)
        self.assertAlmostEqual(res['approval_probability'] + res['rejection_probability'], 100.0, places=1)

    def test_what_if_sensitivity(self):
        """Changing credit score from 580 to 780 should substantially boost approval probability."""
        base_app = {
            'monthly_income': 50000,
            'credit_score': 580,
            'employment_status': 'Salaried',
            'existing_loans': 'No',
            'loan_amount': 400000,
            'repayment_history': 'Average',
            'employment_duration': 3,
            'existing_monthly_debt': 10000
        }
        res_before = self.model.predict(base_app)
        
        improved_app = dict(base_app)
        improved_app['credit_score'] = 780
        improved_app['repayment_history'] = 'Excellent'
        res_after = self.model.predict(improved_app)
        
        print(f"What-If Comparison: {res_before['approval_probability']}% -> {res_after['approval_probability']}%")
        self.assertGreater(res_after['approval_probability'], res_before['approval_probability'])

    def test_cpt_retrieval(self):
        """Verify CPT retrieval for visual inspector."""
        cpt = self.model.get_cpt('FinancialStability')
        self.assertIsNotNone(cpt)
        self.assertEqual(cpt['variable'], 'FinancialStability')
        self.assertEqual(len(cpt['parents']), 3)


if __name__ == '__main__':
    unittest.main()
