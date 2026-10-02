# AI-Based Loan Approval Prediction Using Bayesian Network

An academic, production-grade Artificial Intelligence web application demonstrating **probabilistic reasoning**, **Bayes' Rule**, and **Bayesian Networks (Directed Acyclic Graphs)** for loan underwriting risk assessment.

Developed as a comprehensive case study for **Artificial Intelligence (Unit 1: Quantifying Uncertainty & Probabilistic Reasoning)**.

---

## 📌 1. Where and How Real AI is Used

Unlike standard rule-based or deterministic loan calculators that rely on hardcoded `if creditScore > 700: approve` logic, this application performs **genuine probabilistic machine learning and Bayesian inference**:

1. **Probabilistic Graphical Model (PGM)**: The loan decision domain is modeled as a 12-node **Bayesian Directed Acyclic Graph (DAG)** with 8 observable evidence nodes, 3 intermediate latent synthesis nodes, and 1 target decision node (`LoanApproval`).
2. **Exact Bayesian Inference**: Evaluates the exact posterior conditional distribution:
   $$P(\text{LoanApproval} = \text{Approved} \mid \mathbf{e})$$
   and
   $$P(\text{LoanApproval} = \text{Rejected} \mid \mathbf{e})$$
   using the **Variable Elimination algorithm** via `pgmpy`.
3. **Mathematically Coherent Conditional Probability Tables (CPTs)**: Every node is governed by a rigorously normalized conditional probability distribution $\sum P(X \mid \text{Parents}(X)) = 1.0$.
4. **Hierarchical Parameter Reduction**: Rather than an exponential flat distribution requiring **15,552 parameters**, our 2-layer canonical structure requires only **283 parameters** (a 98.2% reduction) while preserving authentic real-world financial dependencies.
5. **Explainable AI (XAI) via Bayesian Sensitivity Analysis**: Computes isolated marginal shifts $\Delta_i = P(\text{Approved} \mid e_i) - P(\text{Approved})$ against baseline priors to mathematically explain why factors positively support or penalize an applicant.
6. **Dynamic Counterfactual What-If Reasoning**: Users can tweak any variable and observe real-time probability propagation through the DAG.

---

## 🏛️ 2. Bayesian Network Architecture

```
                    [Income]          [EmploymentStatus]    [EmploymentDuration]
                       \                     |                     /
                        \                    |                    /
                         v                   v                   v
                               [FinancialStability]
                                       |
  [CreditScore]                        |
        \                              |
         v                             v
  [Creditworthiness] -------------> [LoanApproval] <------------- [LoanAffordability]
         ^                             ^                                 ^
        /                              |                                / \
  [RepaymentHistory]                   |                               /   \
        ^                              |                       [LoanAmount] [DebtToIncomeRatio]
        |                              |
  [ExistingLoans] ---------------------+
```

### Node Specifications:
- **Observable Evidence Nodes (8)**:
  - `Income`: {Low, Medium, High}
  - `EmploymentStatus`: {Unemployed, SelfEmployed, Salaried}
  - `EmploymentDuration`: {Short, Medium, Long}
  - `CreditScore`: {Poor, Fair, Good, Excellent}
  - `RepaymentHistory`: {Poor, Average, Good, Excellent}
  - `ExistingLoans`: {No, Yes}
  - `LoanAmount`: {Low, Medium, High}
  - `DebtToIncomeRatio`: {Low, Moderate, High}
- **Intermediate Latent Synthesis Nodes (3)**:
  - `FinancialStability`: Synthesizes income, employment type, and duration.
  - `Creditworthiness`: Synthesizes credit score, repayment track record, and existing debt liabilities.
  - `LoanAffordability`: Synthesizes requested principal and debt-to-income (DTI) burden.
- **Target Query Node (1)**:
  - `LoanApproval`: {Approved, Rejected}

---

## 🧪 3. Sample Applicants & Model Outputs

The model was tested and validated on three distinct profiles:

### Sample 1: Prime / High Approval
- **Inputs**: Monthly Income = ₹70,000, Credit Score = 780 (Excellent), Employment = Salaried (5 yrs), Existing Loans = No, Loan Amount = ₹500,000, Repayment History = Excellent, DTI = 10.0%.
- **Bayesian Output**:
  - **Loan Approval Probability**: `82.63%`
  - **Loan Rejection Probability**: `17.37%` (Sum = 100.0%)
  - **Risk Level**: `LOW`
  - **Verdict**: `HIGH PROBABILITY OF APPROVAL`
  - **Key Factors**: High credit score (+18%), Flawless repayment (+16%), Clean debt profile (+6%).

### Sample 2: Subprime / High Risk
- **Inputs**: Monthly Income = ₹30,000, Credit Score = 580 (Fair), Employment = Self-employed (1 yr), Existing Loans = Yes, Loan Amount = ₹1,000,000, Repayment History = Poor, DTI = 53.3%.
- **Bayesian Output**:
  - **Loan Approval Probability**: `8.90%`
  - **Loan Rejection Probability**: `91.10%` (Sum = 100.0%)
  - **Risk Level**: `HIGH`
  - **Verdict**: `HIGH PROBABILITY OF REJECTION`
  - **Key Factors**: Poor repayment history (-22%), High DTI exceeding 43% (-14%), Subprime credit score (-8%).

### Sample 3: Moderate / Borderline
- **Inputs**: Monthly Income = ₹50,000, Credit Score = 700 (Good), Employment = Salaried (3 yrs), Existing Loans = Yes, Loan Amount = ₹400,000, Repayment History = Good, DTI = 28.0%.
- **Bayesian Output**:
  - **Loan Approval Probability**: `49.45%`
  - **Loan Rejection Probability**: `50.55%` (Sum = 100.0%)
  - **Risk Level**: `MEDIUM`
  - **Verdict**: `MODERATE / CONDITIONAL APPROVAL`
  - **Key Factors**: Balanced trade-off between good credit management (+8%) and active debt liabilities (-6%).

---

## 🚀 4. How to Run Locally

### Prerequisites
- Python 3.9+ (Installed and on PATH)
- Pip package manager

### Installation Steps
```bash
# 1. Clone or navigate to the project root directory
cd "C:\Users\Dhanush Teja\.gemini\antigravity\scratch\ai_loan_bayesian_network"

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python run.py
```
Or directly via Flask:
```bash
python backend/app.py
```

Open your browser at:
👉 **`http://127.0.0.1:5050`**

---

## 🧪 5. Running Automated Tests

Run the comprehensive unit test suite and API integration tests:

```bash
# Run model and Bayesian inference tests
python -m unittest tests/test_bayesian_model.py

# Run Flask REST API integration tests
python -m unittest tests/test_api.py
```

All tests verify DAG acyclicity, CPT sum normalization ($\sum = 1.0$), monotonic sensitivity, and edge-case handling.

---

## 🌟 6. Key Application Features

1. **Central Home Portal**:
   - Welcome hero with live system telemetry (Total Evaluated, Approvals, Rejections, Model Status).
   - Quick-access cards connecting to every application module.
   - 1-click Benchmark Scenario Launchpad (Prime, Subprime, Borderline).
2. **Loan Underwriting Assessment**:
   - Complete personal and financial parameter entry.
   - Dynamic real-time Debt-to-Income (DTI) ratio gauge.
   - 1-click Sample Applicant presets (Prime, High Risk, Borderline).
   - "Run AI Loan Prediction" with animated circular probability meters.
3. **Explainable AI (XAI)**:
   - Factor-by-factor Bayesian sensitivity cards detailing exact percentage point impacts against baseline priors.
4. **Dynamic What-If Counterfactual Sandbox**:
   - Live sliders to modify credit score, income, loan amount, and history.
   - Real-time side-by-side delta visualization and Bayesian belief propagation narrative.
5. **Executive Analytics Dashboard**:
   - Chart.js visual analytics for decision breakdowns, risk classifications, credit tiers, and income correlations.
6. **Persistent Prediction History**:
   - SQLite persistence with search, filtering, and audit inspection.

---

## ⚠️ 7. Ethical Notice & Disclaimer

**Educational AI Prototype Notice**: This system is designed as an educational demonstration of probabilistic reasoning and Bayesian Networks for college Artificial Intelligence coursework. It is **not** a real banking or legal lending decision engine. Real-world credit underwriting is subject to strict regulatory, financial, legal, and institutional criteria. Sensitive demographic attributes (such as race, religion, gender, or marital status) are strictly excluded from all model considerations.
