import sys
sys.path.append('backend')
from bayesian_model import LoanBayesianModel

model = LoanBayesianModel()

scores = [500, 570, 580, 620, 660, 670, 710, 739, 740, 780, 820]
print("=== CREDIT SCORE STEP FUNCTION IN CURRENT MODEL ===")
for cs in scores:
    res = model.predict({
        'credit_score': cs,
        'monthly_income': 50000,
        'loan_amount': 400000,
        'repayment_history': 'Good',
        'existing_loans': 'No'
    })
    print(f"CS: {cs:3d} -> State: {res['evidence']['CreditScore']:9s} | P(Approval): {res['approval_probability']:5.2f}%")
