"""
Bayesian Network Model for Loan Approval Prediction
Using pgmpy for exact probabilistic inference (Variable Elimination).
"""

import numpy as np
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class LoanBayesianModel:
    def __init__(self):
        self.model = None
        self.inference_engine = None
        self._build_network()

    def _build_network(self):
        # 1. Define DAG structure
        self.model = BayesianNetwork([
            ('Income', 'FinancialStability'),
            ('EmploymentStatus', 'FinancialStability'),
            ('EmploymentDuration', 'FinancialStability'),
            ('CreditScore', 'Creditworthiness'),
            ('RepaymentHistory', 'Creditworthiness'),
            ('ExistingLoans', 'Creditworthiness'),
            ('LoanAmount', 'LoanAffordability'),
            ('DebtToIncomeRatio', 'LoanAffordability'),
            ('FinancialStability', 'LoanApproval'),
            ('Creditworthiness', 'LoanApproval'),
            ('LoanAffordability', 'LoanApproval')
        ])

        # 2. Define Prior CPDs for Evidence Nodes
        cpd_income = TabularCPD(
            variable='Income',
            variable_card=3,
            values=[[0.30], [0.50], [0.20]],
            state_names={'Income': ['Low', 'Medium', 'High']}
        )

        cpd_emp_status = TabularCPD(
            variable='EmploymentStatus',
            variable_card=3,
            values=[[0.10], [0.30], [0.60]],
            state_names={'EmploymentStatus': ['Unemployed', 'SelfEmployed', 'Salaried']}
        )

        cpd_emp_duration = TabularCPD(
            variable='EmploymentDuration',
            variable_card=3,
            values=[[0.25], [0.45], [0.30]],
            state_names={'EmploymentDuration': ['Short', 'Medium', 'Long']}
        )

        cpd_credit_score = TabularCPD(
            variable='CreditScore',
            variable_card=4,
            values=[[0.15], [0.25], [0.40], [0.20]],
            state_names={'CreditScore': ['Poor', 'Fair', 'Good', 'Excellent']}
        )

        cpd_repay_hist = TabularCPD(
            variable='RepaymentHistory',
            variable_card=4,
            values=[[0.15], [0.25], [0.40], [0.20]],
            state_names={'RepaymentHistory': ['Poor', 'Average', 'Good', 'Excellent']}
        )

        cpd_existing_loans = TabularCPD(
            variable='ExistingLoans',
            variable_card=2,
            values=[[0.45], [0.55]],
            state_names={'ExistingLoans': ['No', 'Yes']}
        )

        cpd_loan_amount = TabularCPD(
            variable='LoanAmount',
            variable_card=3,
            values=[[0.35], [0.45], [0.20]],
            state_names={'LoanAmount': ['Low', 'Medium', 'High']}
        )

        cpd_dti = TabularCPD(
            variable='DebtToIncomeRatio',
            variable_card=3,
            values=[[0.35], [0.45], [0.20]],
            state_names={'DebtToIncomeRatio': ['Low', 'Moderate', 'High']}
        )

        # 3. Intermediate CPTs
        # FinancialStability given Income (3), EmploymentStatus (3), EmploymentDuration (3)
        # 3 x 27 = 81 values
        fs_table = []
        for inc_idx in range(3):        # Low=0, Medium=1, High=2
            for emp_idx in range(3):    # Unemployed=0, SelfEmployed=1, Salaried=2
                for dur_idx in range(3):# Short=0, Medium=1, Long=2
                    score = (inc_idx * 0.48) + (emp_idx * 0.36) + (dur_idx * 0.16)  # 0..2.0
                    p_low = 1.0 / (1.0 + np.exp(3.0 * (score - 0.75)))
                    p_high = 1.0 / (1.0 + np.exp(-3.0 * (score - 1.35)))
                    remainder = max(0.04, 1.0 - p_low - p_high)
                    tot = p_low + remainder + p_high
                    fs_table.append((p_low / tot, remainder / tot, p_high / tot))

        fs_values = np.array(fs_table).T.tolist()

        cpd_financial_stability = TabularCPD(
            variable='FinancialStability',
            variable_card=3,
            values=fs_values,
            evidence=['Income', 'EmploymentStatus', 'EmploymentDuration'],
            evidence_card=[3, 3, 3],
            state_names={
                'FinancialStability': ['Low', 'Medium', 'High'],
                'Income': ['Low', 'Medium', 'High'],
                'EmploymentStatus': ['Unemployed', 'SelfEmployed', 'Salaried'],
                'EmploymentDuration': ['Short', 'Medium', 'Long']
            }
        )

        # Creditworthiness given CreditScore (4), RepaymentHistory (4), ExistingLoans (2)
        # 3 x 32 = 96 values
        # Smooth strictly monotonic distribution ensuring Fair, Good, and Excellent are fully distinct
        cw_table = []
        for cs_idx in range(4):         # Poor=0, Fair=1, Good=2, Excellent=3
            for rh_idx in range(4):     # Poor=0, Average=1, Good=2, Excellent=3
                for el_idx in range(2): # No=0, Yes=1
                    has_debt_penalty = 0.35 if el_idx == 1 else 0.0
                    score = (cs_idx * 0.52) + (rh_idx * 0.44) - has_debt_penalty
                    p_low = 1.0 / (1.0 + np.exp(2.8 * (score - 0.9)))
                    p_high = 1.0 / (1.0 + np.exp(-2.8 * (score - 1.8)))
                    remainder = max(0.05, 1.0 - p_low - p_high)
                    tot = p_low + remainder + p_high
                    cw_table.append((p_low / tot, remainder / tot, p_high / tot))

        cw_values = np.array(cw_table).T.tolist()

        cpd_creditworthiness = TabularCPD(
            variable='Creditworthiness',
            variable_card=3,
            values=cw_values,
            evidence=['CreditScore', 'RepaymentHistory', 'ExistingLoans'],
            evidence_card=[4, 4, 2],
            state_names={
                'Creditworthiness': ['Low', 'Medium', 'High'],
                'CreditScore': ['Poor', 'Fair', 'Good', 'Excellent'],
                'RepaymentHistory': ['Poor', 'Average', 'Good', 'Excellent'],
                'ExistingLoans': ['No', 'Yes']
            }
        )

        # LoanAffordability given LoanAmount (3), DebtToIncomeRatio (3)
        # 3 x 9 = 27 values
        # LoanAmount: Low=0, Medium=1, High=2
        # DTI: Low=0, Moderate=1, High=2
        la_table = []
        for la_idx in range(3):
            for dti_idx in range(3):
                burden = (la_idx * 0.45) + (dti_idx * 0.55)  # 0..2
                p_high = 1.0 / (1.0 + np.exp(3.0 * (burden - 0.7)))
                p_low = 1.0 / (1.0 + np.exp(-3.0 * (burden - 1.3)))
                remainder = max(0.05, 1.0 - p_low - p_high)
                tot = p_low + remainder + p_high
                la_table.append((p_low / tot, remainder / tot, p_high / tot))

        la_values = np.array(la_table).T.tolist()

        cpd_affordability = TabularCPD(
            variable='LoanAffordability',
            variable_card=3,
            values=la_values,
            evidence=['LoanAmount', 'DebtToIncomeRatio'],
            evidence_card=[3, 3],
            state_names={
                'LoanAffordability': ['Low', 'Medium', 'High'],
                'LoanAmount': ['Low', 'Medium', 'High'],
                'DebtToIncomeRatio': ['Low', 'Moderate', 'High']
            }
        )

        # 4. Target Node CPD: LoanApproval given FinancialStability (3), Creditworthiness (3), LoanAffordability (3)
        # Values: ['Approved', 'Rejected'] -> 2 x 27 = 54 values
        # States: Low=0, Medium=1, High=2
        appr_table = []
        for fs_idx in range(3):
            for cw_idx in range(3):
                for la_idx in range(3):
                    total_score = (cw_idx * 0.50) + (fs_idx * 0.28) + (la_idx * 0.22)  # 0..2.0
                    if cw_idx == 0:
                        p_appr = max(0.03, 0.05 + total_score * 0.08)
                    else:
                        p_appr = 1.0 / (1.0 + np.exp(-3.2 * (total_score - 1.05)))
                        p_appr = max(0.05, min(0.96, p_appr))
                    p_rej = 1.0 - p_appr
                    appr_table.append((p_appr, p_rej))

        appr_values = np.array(appr_table).T.tolist()

        cpd_loan_approval = TabularCPD(
            variable='LoanApproval',
            variable_card=2,
            values=appr_values,
            evidence=['FinancialStability', 'Creditworthiness', 'LoanAffordability'],
            evidence_card=[3, 3, 3],
            state_names={
                'LoanApproval': ['Approved', 'Rejected'],
                'FinancialStability': ['Low', 'Medium', 'High'],
                'Creditworthiness': ['Low', 'Medium', 'High'],
                'LoanAffordability': ['Low', 'Medium', 'High']
            }
        )

        # 5. Add all CPDs to the model and validate
        self.model.add_cpds(
            cpd_income, cpd_emp_status, cpd_emp_duration,
            cpd_credit_score, cpd_repay_hist, cpd_existing_loans,
            cpd_loan_amount, cpd_dti,
            cpd_financial_stability, cpd_creditworthiness, cpd_affordability,
            cpd_loan_approval
        )

        assert self.model.check_model(), "Bayesian Network model check failed!"
        self.inference_engine = VariableElimination(self.model)

    def discretize_inputs(self, raw_input):
        """
        Maps continuous/raw form inputs to Bayesian discrete evidence states.
        """
        # Income
        inc = float(raw_input.get('monthly_income', 0))
        if inc < 35000:
            income_state = 'Low'
        elif inc <= 80000:
            income_state = 'Medium'
        else:
            income_state = 'High'

        # Employment Status
        emp_status_map = {
            'Unemployed': 'Unemployed',
            'Self-employed': 'SelfEmployed',
            'SelfEmployed': 'SelfEmployed',
            'Salaried': 'Salaried'
        }
        emp_status_state = emp_status_map.get(raw_input.get('employment_status', 'Salaried'), 'Salaried')

        # Employment Duration
        dur = float(raw_input.get('employment_duration', 0))
        if dur < 2.0:
            emp_dur_state = 'Short'
        elif dur <= 5.0:
            emp_dur_state = 'Medium'
        else:
            emp_dur_state = 'Long'

        # Credit Score
        cs = float(raw_input.get('credit_score', 650))
        if cs < 580:
            credit_score_state = 'Poor'
        elif cs < 670:
            credit_score_state = 'Fair'
        elif cs < 740:
            credit_score_state = 'Good'
        else:
            credit_score_state = 'Excellent'

        # Repayment History
        rh = raw_input.get('repayment_history', 'Good')
        repay_map = {
            'Poor': 'Poor',
            'Average': 'Average',
            'Good': 'Good',
            'Excellent': 'Excellent'
        }
        repay_state = repay_map.get(rh, 'Good')

        # Existing Loans
        el = raw_input.get('existing_loans', 'No')
        existing_loans_state = 'Yes' if str(el).lower() in ['yes', 'true', '1'] else 'No'

        # Loan Amount
        amount = float(raw_input.get('loan_amount', 0))
        if amount < 300000:
            loan_amount_state = 'Low'
        elif amount <= 750000:
            loan_amount_state = 'Medium'
        else:
            loan_amount_state = 'High'

        # Debt to Income Ratio
        # Can be supplied or computed from existing_monthly_debt / monthly_income * 100
        dti_raw = raw_input.get('debt_to_income_ratio')
        if dti_raw is not None and str(dti_raw).strip() != '':
            dti = float(dti_raw)
        else:
            debt = float(raw_input.get('existing_monthly_debt', 0))
            dti = (debt / inc * 100.0) if inc > 0 else 50.0

        if dti < 28.0:
            dti_state = 'Low'
        elif dti <= 43.0:
            dti_state = 'Moderate'
        else:
            dti_state = 'High'

        return {
            'Income': income_state,
            'EmploymentStatus': emp_status_state,
            'EmploymentDuration': emp_dur_state,
            'CreditScore': credit_score_state,
            'RepaymentHistory': repay_state,
            'ExistingLoans': existing_loans_state,
            'LoanAmount': loan_amount_state,
            'DebtToIncomeRatio': dti_state
        }, dti

    def predict(self, raw_input):
        """
        Executes exact Bayesian inference P(LoanApproval | Evidence).
        """
        evidence, computed_dti = self.discretize_inputs(raw_input)
        
        # Exact Bayesian Inference using Variable Elimination
        query_result = self.inference_engine.query(
            variables=['LoanApproval'],
            evidence=evidence
        )
        
        # In pgmpy query_result.values maps to state_names
        appr_idx = query_result.state_names['LoanApproval'].index('Approved')
        rej_idx = query_result.state_names['LoanApproval'].index('Rejected')
        
        p_approved = float(query_result.values[appr_idx])
        p_rejected = float(query_result.values[rej_idx])
        
        # Intermediate latent node posterior estimates for deep explainability
        query_intermediates = self.inference_engine.query(
            variables=['FinancialStability', 'Creditworthiness', 'LoanAffordability'],
            evidence=evidence
        )

        fs_idx = query_intermediates.state_names['FinancialStability'].index('High')
        cw_idx = query_intermediates.state_names['Creditworthiness'].index('High')
        la_idx = query_intermediates.state_names['LoanAffordability'].index('High')

        # Determine risk tier and decision
        if p_approved >= 0.70:
            prediction = "HIGH PROBABILITY OF APPROVAL"
            risk_level = "LOW"
            decision_code = "APPROVED"
        elif p_approved >= 0.45:
            prediction = "MODERATE / CONDITIONAL APPROVAL"
            risk_level = "MEDIUM"
            decision_code = "REVIEW"
        else:
            prediction = "HIGH PROBABILITY OF REJECTION"
            risk_level = "HIGH"
            decision_code = "REJECTED"

        # Generate Explainable AI breakdown
        explanations = self.generate_explanations(evidence, p_approved, raw_input, computed_dti)

        return {
            'approval_probability': round(p_approved * 100, 2),
            'rejection_probability': round(p_rejected * 100, 2),
            'prediction': prediction,
            'risk_level': risk_level,
            'decision_code': decision_code,
            'evidence': evidence,
            'computed_dti': round(computed_dti, 2),
            'explanations': explanations
        }

    def generate_explanations(self, evidence, full_p_appr, raw_input, computed_dti):
        """
        Calculates Bayesian sensitivity / log-odds influence of each evidence variable.
        Compares isolated evidence P(LoanApproval = Approved | e_i) with baseline prior P(Approved).
        """
        # Baseline prior probability of approval with no evidence
        prior_res = self.inference_engine.query(variables=['LoanApproval'], evidence={})
        appr_idx = prior_res.state_names['LoanApproval'].index('Approved')
        baseline_p_appr = float(prior_res.values[appr_idx])

        factors = []

        descriptions = {
            'CreditScore': {
                'Poor': ('Credit Score is in the subprime/poor range (< 580).', -0.25, 'warning'),
                'Fair': ('Credit Score is fair (580-669), posing moderate underwriting risk.', -0.08, 'warning'),
                'Good': ('Credit Score is good (670-739), demonstrating reliable credit management.', +0.08, 'positive'),
                'Excellent': ('Excellent Credit Score (740+) significantly bolsters borrower credibility.', +0.18, 'positive')
            },
            'RepaymentHistory': {
                'Poor': ('Past repayment history indicates repeated defaults or late payments.', -0.22, 'warning'),
                'Average': ('Average repayment history with minor past delays.', -0.05, 'neutral'),
                'Good': ('Consistent on-time payment track record.', +0.10, 'positive'),
                'Excellent': ('Flawless repayment history with zero historical delinquency.', +0.16, 'positive')
            },
            'EmploymentStatus': {
                'Unemployed': ('Applicant is currently unemployed, creating critical default risk.', -0.20, 'warning'),
                'SelfEmployed': ('Self-employed status carries variable income flow considerations.', -0.02, 'neutral'),
                'Salaried': ('Stable salaried employment provides predictable monthly cash flow.', +0.06, 'positive')
            },
            'Income': {
                'Low': ('Monthly income is low relative to standard cost of living (< Rs. 35,000 / $35,000).', -0.10, 'warning'),
                'Medium': ('Monthly income is moderate and adequate for standard loan obligations.', +0.04, 'positive'),
                'High': ('High monthly income provides substantial financial headroom.', +0.12, 'positive')
            },
            'DebtToIncomeRatio': {
                'High': (f'Debt-to-Income ratio ({computed_dti:.1f}%) exceeds healthy benchmark (> 43%), severely constraining disposable income.', -0.14, 'warning'),
                'Moderate': (f'Debt-to-Income ratio ({computed_dti:.1f}%) is in acceptable moderate range (28%-43%).', +0.01, 'neutral'),
                'Low': (f'Low Debt-to-Income ratio ({computed_dti:.1f}%) indicates ample buffer for loan servicing.', +0.10, 'positive')
            },
            'LoanAmount': {
                'High': ('Requested loan amount is relatively high (> Rs. 750,000 / $750,000), escalating exposure.', -0.09, 'warning'),
                'Medium': ('Requested loan amount is within normal underwriting limits.', +0.02, 'neutral'),
                'Low': ('Requested loan amount is low, minimizing principal exposure.', +0.08, 'positive')
            },
            'ExistingLoans': {
                'Yes': ('Existing active loan obligations increase ongoing leverage.', -0.06, 'warning'),
                'No': ('No existing loans; borrower has clean liability profile.', +0.06, 'positive')
            },
            'EmploymentDuration': {
                'Short': ('Short employment tenure (< 2 yrs) suggests job transition vulnerability.', -0.05, 'warning'),
                'Medium': ('Reasonable employment stability (2-5 years).', +0.02, 'neutral'),
                'Long': ('Long tenure (> 5 yrs) confirms robust occupational stability.', +0.06, 'positive')
            }
        }

        for var, state in evidence.items():
            # Query isolated marginal
            single_res = self.inference_engine.query(variables=['LoanApproval'], evidence={var: state})
            p_single = float(single_res.values[appr_idx])
            delta = p_single - baseline_p_appr
            
            desc_tuple = descriptions.get(var, {}).get(state, (f'{var} state: {state}', delta, 'neutral'))
            text, _, default_type = desc_tuple
            
            impact_type = 'positive' if delta >= 0.03 else ('warning' if delta <= -0.03 else 'neutral')
            
            factors.append({
                'variable': var,
                'state': state,
                'impact': round(delta * 100, 2),
                'type': impact_type,
                'text': text
            })

        # Sort factors: warnings and strong positives first
        factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        return factors

    def get_network_schema(self):
        """
        Returns JSON-serializable structure of the Bayesian Network for frontend visualization.
        """
        nodes = []
        for node in self.model.nodes():
            cpd = self.model.get_cpds(node)
            nodes.append({
                'id': node,
                'label': node,
                'states': list(cpd.state_names[node]),
                'cardinality': cpd.variable_card,
                'isEvidence': len(cpd.variables) == 1,
                'isTarget': node == 'LoanApproval',
                'isLatent': node in ['FinancialStability', 'Creditworthiness', 'LoanAffordability']
            })

        edges = [{'from': u, 'to': v} for u, v in self.model.edges()]

        return {
            'nodes': nodes,
            'edges': edges
        }

    def get_cpt(self, node_name):
        """
        Returns the Conditional Probability Table for a requested node.
        """
        if node_name not in self.model.nodes():
            return None
        cpd = self.model.get_cpds(node_name)
        
        parents = list(cpd.variables[1:])
        return {
            'variable': node_name,
            'states': list(cpd.state_names[node_name]),
            'parents': parents,
            'parent_states': {p: list(cpd.state_names[p]) for p in parents},
            'values': cpd.values.tolist()
        }

    def predict_continuous(self, raw_input):
        """
        Calculates loan approval probability using continuous soft evidence conditioning
        (Jeffrey's Rule of Bayesian Conditioning) across continuous variables:
        CreditScore, MonthlyIncome, LoanAmount, DebtToIncomeRatio.
        Ensures smooth, sensitive, and responsive counterfactual simulations.
        """
        cs = float(raw_input.get('credit_score', 650))
        inc = float(raw_input.get('monthly_income', 50000))
        amount = float(raw_input.get('loan_amount', 500000))
        
        dti_raw = raw_input.get('debt_to_income_ratio')
        if dti_raw is not None and str(dti_raw).strip() != '':
            dti = float(dti_raw)
        else:
            debt = float(raw_input.get('existing_monthly_debt', 10000))
            dti = (debt / inc * 100.0) if inc > 0 else 20.0

        def soft_weights(val, centers, stds):
            weights = {}
            states = list(centers.keys())
            for s in states:
                c = centers[s]
                sd = stds[s]
                if s == states[0] and val <= c:
                    w = 1.0
                elif s == states[-1] and val >= c:
                    w = 1.0
                else:
                    w = float(np.exp(-0.5 * ((val - c) / sd) ** 2))
                weights[s] = w
            tot = sum(weights.values())
            # Keep meaningful weights >= 2% and re-normalize
            filtered = {s: w / tot for s, w in weights.items() if (w / tot) >= 0.02}
            f_tot = sum(filtered.values())
            return {s: w / f_tot for s, w in filtered.items()}

        cs_w = soft_weights(cs, {'Poor': 480.0, 'Fair': 625.0, 'Good': 705.0, 'Excellent': 790.0},
                                {'Poor': 65.0, 'Fair': 45.0, 'Good': 45.0, 'Excellent': 60.0})
        inc_w = soft_weights(inc, {'Low': 25000.0, 'Medium': 55000.0, 'High': 100000.0},
                                 {'Low': 12000.0, 'Medium': 18000.0, 'High': 25000.0})
        la_w = soft_weights(amount, {'Low': 200000.0, 'Medium': 500000.0, 'High': 1000000.0},
                                   {'Low': 80000.0, 'Medium': 150000.0, 'High': 250000.0})
        dti_w = soft_weights(dti, {'Low': 20.0, 'Moderate': 35.0, 'High': 52.0},
                                  {'Low': 7.0, 'Moderate': 8.0, 'High': 10.0})

        # Non-continuous variables mapped to discrete states
        discrete_evidence, _ = self.discretize_inputs(raw_input)
        base_discrete_ev = {
            'EmploymentStatus': discrete_evidence['EmploymentStatus'],
            'EmploymentDuration': discrete_evidence['EmploymentDuration'],
            'RepaymentHistory': discrete_evidence['RepaymentHistory'],
            'ExistingLoans': discrete_evidence['ExistingLoans']
        }

        # Dominant evidence states for display / logging
        dominant_cs = max(cs_w.items(), key=lambda x: x[1])[0]
        dominant_inc = max(inc_w.items(), key=lambda x: x[1])[0]
        dominant_la = max(la_w.items(), key=lambda x: x[1])[0]
        dominant_dti = max(dti_w.items(), key=lambda x: x[1])[0]

        effective_evidence = dict(discrete_evidence,
            CreditScore=dominant_cs,
            Income=dominant_inc,
            LoanAmount=dominant_la,
            DebtToIncomeRatio=dominant_dti
        )

        appr_idx = self.inference_engine.query(['LoanApproval'], evidence={}).state_names['LoanApproval'].index('Approved')
        total_appr = 0.0
        total_weight = 0.0

        for cs_s, cs_p in cs_w.items():
            for inc_s, inc_p in inc_w.items():
                for la_s, la_p in la_w.items():
                    for dti_s, dti_p in dti_w.items():
                        comb_weight = cs_p * inc_p * la_p * dti_p
                        ev = dict(base_discrete_ev, CreditScore=cs_s, Income=inc_s, LoanAmount=la_s, DebtToIncomeRatio=dti_s)
                        q = self.inference_engine.query(['LoanApproval'], evidence=ev)
                        p = float(q.values[appr_idx])
                        total_appr += p * comb_weight
                        total_weight += comb_weight

        p_approved = total_appr / total_weight
        p_approved = max(0.01, min(0.99, p_approved))
        p_rejected = 1.0 - p_approved

        if p_approved >= 0.70:
            prediction = "HIGH PROBABILITY OF APPROVAL"
            risk_level = "LOW"
            decision_code = "APPROVED"
        elif p_approved >= 0.45:
            prediction = "MODERATE / CONDITIONAL APPROVAL"
            risk_level = "MEDIUM"
            decision_code = "REVIEW"
        else:
            prediction = "HIGH PROBABILITY OF REJECTION"
            risk_level = "HIGH"
            decision_code = "REJECTED"

        return {
            'approval_probability': round(p_approved * 100, 2),
            'rejection_probability': round(p_rejected * 100, 2),
            'prediction': prediction,
            'risk_level': risk_level,
            'decision_code': decision_code,
            'evidence': effective_evidence,
            'computed_dti': round(dti, 2),
            'soft_evidence': {
                'CreditScore': {k: round(v * 100, 1) for k, v in cs_w.items()},
                'Income': {k: round(v * 100, 1) for k, v in inc_w.items()},
                'LoanAmount': {k: round(v * 100, 1) for k, v in la_w.items()},
                'DebtToIncomeRatio': {k: round(v * 100, 1) for k, v in dti_w.items()}
            }
        }

    def predict_what_if(self, original_raw, modified_raw):
        """
        Evaluates dynamic counterfactual prediction with continuous Bayesian conditioning.
        """
        res_original = self.predict_continuous(original_raw)
        res_modified = self.predict_continuous(modified_raw)

        diff_approval = round(res_modified['approval_probability'] - res_original['approval_probability'], 2)
        diff_rejection = round(res_modified['rejection_probability'] - res_original['rejection_probability'], 2)

        changed_vars = {}
        for k in modified_raw.keys():
            if str(original_raw.get(k)) != str(modified_raw.get(k)):
                changed_vars[k] = {
                    'before': original_raw.get(k),
                    'after': modified_raw.get(k)
                }

        return {
            'original': res_original,
            'modified': res_modified,
            'delta_approval': diff_approval,
            'delta_rejection': diff_rejection,
            'changed_variables': changed_vars
        }
