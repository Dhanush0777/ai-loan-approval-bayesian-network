import sys
sys.path.append('backend')
from bayesian_model import LoanBayesianModel
from test_soft_evidence import compute_credit_score_soft_evidence

model = LoanBayesianModel()

raw_base = {
    'monthly_income': 50000,
    'loan_amount': 400000,
    'repayment_history': 'Good',
    'existing_loans': 'No',
    'employment_status': 'Salaried',
    'employment_duration': 3,
    'existing_monthly_debt': 10000
}

# Pre-query for each discrete credit score state
evidence_base, _ = model.discretize_inputs(raw_base)
state_posteriors = {}
appr_idx = model.inference_engine.query(['LoanApproval'], evidence={}).state_names['LoanApproval'].index('Approved')

for cs_state in ['Poor', 'Fair', 'Good', 'Excellent']:
    ev = dict(evidence_base, CreditScore=cs_state)
    q = model.inference_engine.query(['LoanApproval'], evidence=ev)
    state_posteriors[cs_state] = float(q.values[appr_idx])

print("Discrete State Posteriors:")
for k, v in state_posteriors.items():
    print(f"  {k:9s}: {v*100:.2f}%")

print("\nContinuous Credit Score vs Bayesian Approval Probability:")
for cs in range(500, 850, 25):
    soft_p = compute_credit_score_soft_evidence(cs)
    prob = sum(soft_p[s] * state_posteriors[s] for s in ['Poor', 'Fair', 'Good', 'Excellent'])
    print(f"CS: {cs:3d} -> Approval: {prob*100:5.2f}%")
