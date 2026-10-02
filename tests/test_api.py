"""
API Integration tests for the Flask backend.
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from app import app, database


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_samples_endpoint(self):
        response = self.app.get('/api/samples')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('sample_1', data)
        self.assertIn('sample_2', data)
        self.assertIn('sample_3', data)

    def test_model_info_endpoint(self):
        response = self.app.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        node_ids = [n['id'] for n in data['nodes']]
        self.assertIn('LoanApproval', node_ids)
        self.assertIn('CreditScore', node_ids)

    def test_cpt_endpoint(self):
        response = self.app.get('/api/cpt/FinancialStability')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['variable'], 'FinancialStability')
        self.assertIn('parents', data)

    def test_predict_endpoint_prime(self):
        payload = {
            'applicant_name': 'Test Prime User',
            'age': 35,
            'monthly_income': 70000,
            'credit_score': 780,
            'employment_status': 'Salaried',
            'existing_loans': 'No',
            'existing_monthly_debt': 7000,
            'loan_amount': 500000,
            'loan_tenure': 48,
            'repayment_history': 'Excellent',
            'employment_duration': 5
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertIn('approval_probability', res)
        self.assertIn('rejection_probability', res)
        self.assertIn('risk_level', res)
        self.assertEqual(res['risk_level'], 'LOW')
        self.assertGreater(res['approval_probability'], 80)
        self.assertIn('record_id', res)

    def test_predict_endpoint_high_risk(self):
        payload = {
            'applicant_name': 'Test Subprime User',
            'age': 28,
            'monthly_income': 30000,
            'credit_score': 580,
            'employment_status': 'Self-employed',
            'existing_loans': 'Yes',
            'existing_monthly_debt': 16000,
            'loan_amount': 1000000,
            'loan_tenure': 60,
            'repayment_history': 'Poor',
            'employment_duration': 1
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertEqual(res['risk_level'], 'HIGH')
        self.assertLess(res['approval_probability'], 35)

    def test_what_if_endpoint(self):
        payload = {
            'original': {
                'monthly_income': 30000,
                'credit_score': 580,
                'employment_status': 'Self-employed',
                'existing_loans': 'Yes',
                'existing_monthly_debt': 16000,
                'loan_amount': 1000000,
                'loan_tenure': 60,
                'repayment_history': 'Poor',
                'employment_duration': 1
            },
            'modified': {
                'monthly_income': 70000,
                'credit_score': 780,
                'employment_status': 'Salaried',
                'existing_loans': 'No',
                'existing_monthly_debt': 5000,
                'loan_amount': 500000,
                'loan_tenure': 48,
                'repayment_history': 'Excellent',
                'employment_duration': 5
            }
        }
        response = self.app.post('/api/what-if', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertIn('delta_approval', res)
        self.assertGreater(res['delta_approval'], 0)

    def test_dashboard_stats_and_history(self):
        stats_resp = self.app.get('/api/dashboard-stats')
        self.assertEqual(stats_resp.status_code, 200)
        stats = json.loads(stats_resp.data)
        self.assertGreaterEqual(stats['total_applications'], 1)

        hist_resp = self.app.get('/api/history')
        self.assertEqual(hist_resp.status_code, 200)
        hist = json.loads(hist_resp.data)
        self.assertGreaterEqual(len(hist['predictions']), 1)


if __name__ == '__main__':
    unittest.main()
