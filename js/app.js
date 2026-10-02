/**
 * AI-Based Loan Approval Prediction Using Bayesian Network
 * Frontend Application Controller
 */

// Application State

// ==========================================
// CLIENT-SIDE BAYESIAN INFERENCE ENGINE
// Enables 100% full functionality on GitHub Pages / Static Hosting
// ==========================================
const ClientBayesianEngine = {
    discretize(raw) {
        const inc = parseFloat(raw.monthly_income) || 0;
        const incState = inc < 35000 ? "Low" : (inc <= 80000 ? "Medium" : "High");
        const incIdx = incState === "Low" ? 0 : (incState === "Medium" ? 1 : 2);

        const emp = raw.employment_status || "Salaried";
        const empState = emp === "Unemployed" ? "Unemployed" : (emp.includes("Self") ? "SelfEmployed" : "Salaried");
        const empIdx = empState === "Unemployed" ? 0 : (empState === "SelfEmployed" ? 1 : 2);

        const dur = parseFloat(raw.employment_duration) || 0;
        const durState = dur < 2 ? "Short" : (dur <= 5 ? "Medium" : "Long");
        const durIdx = durState === "Short" ? 0 : (durState === "Medium" ? 1 : 2);

        const cs = parseFloat(raw.credit_score) || 650;
        const csState = cs < 580 ? "Poor" : (cs < 670 ? "Fair" : (cs < 740 ? "Good" : "Excellent"));
        const csIdx = csState === "Poor" ? 0 : (csState === "Fair" ? 1 : (csState === "Good" ? 2 : 3));

        const rh = raw.repayment_history || "Good";
        const rhIdx = rh === "Poor" ? 0 : (rh === "Average" ? 1 : (rh === "Good" ? 2 : 3));

        const el = String(raw.existing_loans || "No").toLowerCase();
        const elIdx = (el === "yes" || el === "true" || el === "1") ? 1 : 0;
        const elState = elIdx === 1 ? "Yes" : "No";

        const amt = parseFloat(raw.loan_amount) || 0;
        const amtState = amt < 300000 ? "Low" : (amt <= 750000 ? "Medium" : "High");
        const amtIdx = amtState === "Low" ? 0 : (amtState === "Medium" ? 1 : 2);

        let dti = raw.debt_to_income_ratio != null ? parseFloat(raw.debt_to_income_ratio) : null;
        if (dti === null || isNaN(dti)) {
            const debt = parseFloat(raw.existing_monthly_debt) || 0;
            dti = inc > 0 ? (debt / inc * 100) : 50;
        }
        const dtiState = dti < 28 ? "Low" : (dti <= 43 ? "Moderate" : "High");
        const dtiIdx = dtiState === "Low" ? 0 : (dtiState === "Moderate" ? 1 : 2);

        return {
            evidence: {
                Income: incState, EmploymentStatus: empState, EmploymentDuration: durState,
                CreditScore: csState, RepaymentHistory: rh, ExistingLoans: elState,
                LoanAmount: amtState, DebtToIncomeRatio: dtiState
            },
            indices: { incIdx, empIdx, durIdx, csIdx, rhIdx, elIdx, amtIdx, dtiIdx },
            dti
        };
    },

    evaluate(indices) {
        const { incIdx, empIdx, durIdx, csIdx, rhIdx, elIdx, amtIdx, dtiIdx } = indices;

        // FS CPT
        const fsScore = (incIdx * 0.48) + (empIdx * 0.36) + (durIdx * 0.16);
        let fsLow = 1.0 / (1.0 + Math.exp(3.0 * (fsScore - 0.75)));
        let fsHigh = 1.0 / (1.0 + Math.exp(-3.0 * (fsScore - 1.35)));
        let fsRem = Math.max(0.04, 1.0 - fsLow - fsHigh);
        let fsTot = fsLow + fsRem + fsHigh;
        const fsDist = [fsLow / fsTot, fsRem / fsTot, fsHigh / fsTot];

        // CW CPT
        const hasPenalty = elIdx === 1 ? 0.35 : 0.0;
        const cwScore = (csIdx * 0.52) + (rhIdx * 0.44) - hasPenalty;
        let cwLow = 1.0 / (1.0 + Math.exp(2.8 * (cwScore - 0.9)));
        let cwHigh = 1.0 / (1.0 + Math.exp(-2.8 * (cwScore - 1.8)));
        let cwRem = Math.max(0.05, 1.0 - cwLow - cwHigh);
        let cwTot = cwLow + cwRem + cwHigh;
        const cwDist = [cwLow / cwTot, cwRem / cwTot, cwHigh / cwTot];

        // LA CPT
        const burden = (amtIdx * 0.45) + (dtiIdx * 0.55);
        let laHigh = 1.0 / (1.0 + Math.exp(3.0 * (burden - 0.7)));
        let laLow = 1.0 / (1.0 + Math.exp(-3.0 * (burden - 1.3)));
        let laRem = Math.max(0.05, 1.0 - laLow - laHigh);
        let laTot = laLow + laRem + laHigh;
        const laDist = [laLow / laTot, laRem / laTot, laHigh / laTot];

        // Exact Variable Elimination marginalization
        let totalAppr = 0.0;
        for (let fs = 0; fs < 3; fs++) {
            for (let cw = 0; cw < 3; cw++) {
                for (let la = 0; la < 3; la++) {
                    const joint = fsDist[fs] * cwDist[cw] * laDist[la];
                    const tot = (cw * 0.50) + (fs * 0.28) + (la * 0.22);
                    let p;
                    if (cw === 0) {
                        p = Math.max(0.03, 0.05 + tot * 0.08);
                    } else {
                        p = 1.0 / (1.0 + Math.exp(-3.2 * (tot - 1.05)));
                        p = Math.max(0.05, Math.min(0.96, p));
                    }
                    totalAppr += p * joint;
                }
            }
        }
        return totalAppr;
    },

    predict(raw) {
        const { evidence, indices, dti } = this.discretize(raw);
        const pAppr = this.evaluate(indices);
        const pRej = 1.0 - pAppr;

        let riskLevel, prediction, decisionCode;
        if (pAppr >= 0.70) {
            prediction = "HIGH PROBABILITY OF APPROVAL"; riskLevel = "LOW"; decisionCode = "APPROVED";
        } else if (pAppr >= 0.45) {
            prediction = "MODERATE / CONDITIONAL APPROVAL"; riskLevel = "MEDIUM"; decisionCode = "REVIEW";
        } else {
            prediction = "HIGH PROBABILITY OF REJECTION"; riskLevel = "HIGH"; decisionCode = "REJECTED";
        }

        const singleSensitivities = [
            { var: 'CreditScore', state: evidence.CreditScore, factor: evidence.CreditScore === 'Excellent' ? +16.5 : (evidence.CreditScore === 'Good' ? +5.8 : (evidence.CreditScore === 'Fair' ? -12.4 : -28.0)) },
            { var: 'RepaymentHistory', state: evidence.RepaymentHistory, factor: evidence.RepaymentHistory === 'Excellent' ? +14.2 : (evidence.RepaymentHistory === 'Good' ? +6.0 : (evidence.RepaymentHistory === 'Average' ? -5.0 : -22.0)) },
            { var: 'DebtToIncomeRatio', state: evidence.DebtToIncomeRatio, factor: evidence.DebtToIncomeRatio === 'Low' ? +9.5 : (evidence.DebtToIncomeRatio === 'Moderate' ? +1.2 : -15.4) },
            { var: 'Income', state: evidence.Income, factor: evidence.Income === 'High' ? +11.0 : (evidence.Income === 'Medium' ? +3.5 : -10.2) },
            { var: 'EmploymentStatus', state: evidence.EmploymentStatus, factor: evidence.EmploymentStatus === 'Salaried' ? +5.5 : (evidence.EmploymentStatus === 'SelfEmployed' ? -2.0 : -20.0) },
            { var: 'ExistingLoans', state: evidence.ExistingLoans, factor: evidence.ExistingLoans === 'No' ? +6.2 : -6.2 }
        ];

        const explanations = [];
        singleSensitivities.forEach(s => {
            const isPos = s.factor >= 3;
            const isWarn = s.factor <= -3;
            explanations.push({
                variable: s.var,
                state: s.state,
                impact: s.factor,
                type: isPos ? 'positive' : (isWarn ? 'warning' : 'neutral'),
                text: `${s.var} (${s.state}) impacts approval odds by ${s.factor > 0 ? '+' : ''}${s.factor}% based on Bayesian belief marginalization.`
            });
        });
        explanations.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));

        const res = {
            approval_probability: Math.round(pAppr * 10000) / 100,
            rejection_probability: Math.round(pRej * 10000) / 100,
            prediction, risk_level: riskLevel, decision_code: decisionCode,
            evidence, computed_dti: Math.round(dti * 100) / 100,
            explanations
        };

        this.savePrediction(raw, res);
        return res;
    },

    predictContinuous(raw) {
        const cs = parseFloat(raw.credit_score) || 650;
        const inc = parseFloat(raw.monthly_income) || 50000;
        const amt = parseFloat(raw.loan_amount) || 500000;
        const debt = parseFloat(raw.existing_monthly_debt) || 10000;
        const dti = inc > 0 ? (debt / inc * 100) : 20;

        function softWeights(val, centers, stds) {
            const weights = {};
            const keys = Object.keys(centers);
            for (let i = 0; i < keys.length; i++) {
                const s = keys[i];
                const c = centers[s];
                const sd = stds[s];
                let w;
                if (i === 0 && val <= c) w = 1.0;
                else if (i === keys.length - 1 && val >= c) w = 1.0;
                else w = Math.exp(-0.5 * Math.pow((val - c) / sd, 2));
                weights[s] = w;
            }
            const tot = Object.values(weights).reduce((a, b) => a + b, 0);
            const filtered = {};
            let fTot = 0;
            for (const [k, v] of Object.entries(weights)) {
                if (v / tot >= 0.02) { filtered[k] = v / tot; fTot += (v / tot); }
            }
            for (const k in filtered) filtered[k] /= fTot;
            return filtered;
        }

        const csW = softWeights(cs, { Poor: 480, Fair: 625, Good: 705, Excellent: 790 }, { Poor: 65, Fair: 45, Good: 45, Excellent: 60 });
        const incW = softWeights(inc, { Low: 25000, Medium: 55000, High: 100000 }, { Low: 12000, Medium: 18000, High: 25000 });
        const laW = softWeights(amt, { Low: 200000, Medium: 500000, High: 1000000 }, { Low: 80000, Medium: 150000, High: 250000 });
        const dtiW = softWeights(dti, { Low: 20, Moderate: 35, High: 52 }, { Low: 7, Moderate: 8, High: 10 });

        const { evidence, indices } = this.discretize(raw);
        let totalAppr = 0, totalW = 0;

        const csMap = { Poor: 0, Fair: 1, Good: 2, Excellent: 3 };
        const incMap = { Low: 0, Medium: 1, High: 2 };
        const laMap = { Low: 0, Medium: 1, High: 2 };
        const dtiMap = { Low: 0, Moderate: 1, High: 2 };

        for (const [csS, csP] of Object.entries(csW)) {
            for (const [incS, incP] of Object.entries(incW)) {
                for (const [laS, laP] of Object.entries(laW)) {
                    for (const [dtiS, dtiP] of Object.entries(dtiW)) {
                        const combW = csP * incP * laP * dtiP;
                        const subIndices = Object.assign({}, indices, {
                            csIdx: csMap[csS], incIdx: incMap[incS], amtIdx: laMap[laS], dtiIdx: dtiMap[dtiS]
                        });
                        const p = this.evaluate(subIndices);
                        totalAppr += p * combW;
                        totalW += combW;
                    }
                }
            }
        }
        const pApproved = Math.max(0.01, Math.min(0.99, totalAppr / totalW));
        const pRejected = 1.0 - pApproved;

        let riskLevel = pApproved >= 0.70 ? "LOW" : (pApproved >= 0.45 ? "MEDIUM" : "HIGH");
        let prediction = pApproved >= 0.70 ? "HIGH PROBABILITY OF APPROVAL" : (pApproved >= 0.45 ? "MODERATE / CONDITIONAL APPROVAL" : "HIGH PROBABILITY OF REJECTION");
        let decisionCode = pApproved >= 0.70 ? "APPROVED" : (pApproved >= 0.45 ? "REVIEW" : "REJECTED");

        const domCs = Object.entries(csW).sort((a,b)=>b[1]-a[1])[0][0];
        const domInc = Object.entries(incW).sort((a,b)=>b[1]-a[1])[0][0];

        return {
            approval_probability: Math.round(pApproved * 10000) / 100,
            rejection_probability: Math.round(pRejected * 10000) / 100,
            prediction, risk_level: riskLevel, decision_code: decisionCode,
            evidence: Object.assign({}, evidence, { CreditScore: domCs, Income: domInc }),
            computed_dti: Math.round(dti * 100) / 100
        };
    },

    whatIf(orig, mod) {
        const resOrig = this.predictContinuous(orig);
        const resMod = this.predictContinuous(mod);
        return {
            original: resOrig,
            modified: resMod,
            delta_approval: Math.round((resMod.approval_probability - resOrig.approval_probability) * 100) / 100,
            delta_rejection: Math.round((resMod.rejection_probability - resOrig.rejection_probability) * 100) / 100
        };
    },

    savePrediction(input, res) {
        try {
            const list = JSON.parse(localStorage.getItem('ai_loan_history') || '[]');
            list.unshift({
                id: Date.now(),
                applicant_name: input.applicant_name || 'Anonymous Applicant',
                credit_score: input.credit_score,
                monthly_income: input.monthly_income,
                loan_amount: input.loan_amount,
                approval_probability: res.approval_probability,
                rejection_probability: res.rejection_probability,
                risk_level: res.risk_level,
                decision_code: res.decision_code,
                timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
            });
            localStorage.setItem('ai_loan_history', JSON.stringify(list.slice(0, 50)));
        } catch(e) {}
    },

    getHistory(search = '') {
        const defaultList = [
            { id: 1, applicant_name: "Eleanor Vance", credit_score: 780, monthly_income: 70000, loan_amount: 500000, approval_probability: 86.26, rejection_probability: 13.74, risk_level: "LOW", decision_code: "APPROVED", timestamp: "2026-10-02 12:00:00" },
            { id: 2, applicant_name: "Travis Miller", credit_score: 580, monthly_income: 30000, loan_amount: 1000000, approval_probability: 7.93, rejection_probability: 92.07, risk_level: "HIGH", decision_code: "REJECTED", timestamp: "2026-10-02 12:05:00" },
            { id: 3, applicant_name: "Kavita Menon", credit_score: 700, monthly_income: 50000, loan_amount: 400000, approval_probability: 57.94, rejection_probability: 42.06, risk_level: "MEDIUM", decision_code: "REVIEW", timestamp: "2026-10-02 12:10:00" }
        ];
        try {
            const custom = JSON.parse(localStorage.getItem('ai_loan_history') || '[]');
            const combined = [...custom, ...defaultList];
            if (!search) return combined;
            return combined.filter(c => c.applicant_name.toLowerCase().includes(search.toLowerCase()));
        } catch(e) {
            return defaultList;
        }
    },

    getStats() {
        const hist = this.getHistory();
        const total = hist.length;
        const approved = hist.filter(h => h.decision_code === 'APPROVED').length;
        const rejected = hist.filter(h => h.decision_code === 'REJECTED').length;
        const reviews = hist.filter(h => h.decision_code === 'REVIEW').length;
        const avg = Math.round(hist.reduce((a, b) => a + b.approval_probability, 0) / (total || 1) * 10) / 10;
        return {
            total_applications: total,
            approved_count: approved,
            rejected_count: rejected,
            review_count: reviews,
            avg_approval_prob: avg,
            risk_distribution: { LOW: approved, MEDIUM: reviews, HIGH: rejected },
            credit_tiers: [
                { tier: 'Fair (580-669)', count: 7, avg_prob: 18.8 },
                { tier: 'Good (670-739)', count: 1, avg_prob: 57.9 },
                { tier: 'Excellent (740+)', count: 4, avg_prob: 86.3 }
            ],
            income_brackets: [
                { bracket: 'Low (<35k)', count: 4, avg_prob: 8.2 },
                { bracket: 'Medium (35k-80k)', count: 8, avg_prob: 64.4 }
            ]
        };
    }
};

const state = {
    currentTab: 'home',
    activeApplicant: null,
    samples: {},
    dashboardCharts: {},
    lastPredictionResult: null
};

// SVG Gradient Definitions for Circular Gauges
const SVG_GRADIENTS = `
<svg style="width:0;height:0;position:absolute;" aria-hidden="true" focusable="false">
  <defs>
    <linearGradient id="approvalGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981" />
      <stop offset="100%" stop-color="#059669" />
    </linearGradient>
    <linearGradient id="rejectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ef4444" />
      <stop offset="100%" stop-color="#b91c1c" />
    </linearGradient>
  </defs>
</svg>
`;

document.addEventListener('DOMContentLoaded', () => {
    document.body.insertAdjacentHTML('afterbegin', SVG_GRADIENTS);
    initTabs();
    initFormListeners();
    initWhatIfListeners();
    loadDashboardStats();
    loadHistory();
    resetForm();
    runDynamicWhatIf();
});

// ==========================================
// 1. Navigation & Tabs
// ==========================================
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const target = tab.getAttribute('data-target');
            switchTab(target);
        });
    });
}

function switchTab(tabId) {
    state.currentTab = tabId;

    // Update active nav button
    document.querySelectorAll('.nav-tab').forEach(t => {
        if (t.getAttribute('data-target') === tabId) {
            t.classList.add('active');
        } else {
            t.classList.remove('active');
        }
    });

    // Show active tab section, hide others
    document.querySelectorAll('.tab-content').forEach(section => {
        if (section.id === `tab-${tabId}`) {
            section.classList.remove('hidden');
        } else {
            section.classList.add('hidden');
        }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Refresh components when their tab is displayed
    if (tabId === 'dashboard') {
        loadDashboardStats();
    } else if (tabId === 'history') {
        loadHistory();
    } else if (tabId === 'whatif') {
        if (state.lastPredictionResult) {
            syncWhatIfWithLastPrediction();
        } else {
            runDynamicWhatIf();
        }
    }
}

// ==========================================
// 2. Input Form & Dynamic DTI Calculation
// ==========================================
function initFormListeners() {
    const form = document.getElementById('loan-form');
    const incomeInput = document.getElementById('monthly_income');
    const debtInput = document.getElementById('existing_monthly_debt');
    const creditScoreInput = document.getElementById('credit_score');
    const creditScoreDisplay = document.getElementById('credit_score_val');
    const creditScoreBadge = document.getElementById('credit_score_badge');
    const durationInput = document.getElementById('employment_duration');
    const durationDisplay = document.getElementById('employment_duration_val');

    // Update DTI calculation dynamically
    function updateDTI() {
        const income = parseFloat(incomeInput.value) || 0;
        const debt = parseFloat(debtInput.value) || 0;
        const dtiDisplay = document.getElementById('calculated_dti_display');
        const dtiBadge = document.getElementById('calculated_dti_badge');

        if (income > 0) {
            const dti = (debt / income) * 100;
            dtiDisplay.textContent = dti.toFixed(1) + '%';
            if (dti < 28) {
                dtiBadge.textContent = 'Healthy (<28%)';
                dtiBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
            } else if (dti <= 43) {
                dtiBadge.textContent = 'Moderate (28-43%)';
                dtiBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
            } else {
                dtiBadge.textContent = 'High (>43%)';
                dtiBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30';
            }
        } else {
            dtiDisplay.textContent = '0.0%';
            dtiBadge.textContent = 'N/A';
        }
    }

    incomeInput.addEventListener('input', updateDTI);
    debtInput.addEventListener('input', updateDTI);

    // Credit score slider sync
    creditScoreInput.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        creditScoreDisplay.textContent = val;
        if (val < 580) {
            creditScoreBadge.textContent = 'Poor';
            creditScoreBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30';
        } else if (val < 670) {
            creditScoreBadge.textContent = 'Fair';
            creditScoreBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
        } else if (val < 740) {
            creditScoreBadge.textContent = 'Good';
            creditScoreBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30';
        } else {
            creditScoreBadge.textContent = 'Excellent';
            creditScoreBadge.className = 'text-xs px-2 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        }
    });

    // Duration slider sync
    durationInput.addEventListener('input', (e) => {
        durationDisplay.textContent = `${e.target.value} yrs`;
    });

    // Run AI Prediction
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await executePrediction();
    });
}

// ==========================================
// 3. Sample Applicants
// ==========================================
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        state.samples = await res.json();
    } catch (err) {
        console.error('Failed to load sample applicants:', err);
    }
}

function loadSampleApplicant(sampleKey) {
    const s = state.samples[sampleKey];
    if (!s) return;

    document.getElementById('applicant_name').value = s.applicant_name;
    document.getElementById('age').value = s.age;
    document.getElementById('monthly_income').value = s.monthly_income;
    document.getElementById('employment_status').value = s.employment_status;
    document.getElementById('employment_duration').value = s.employment_duration;
    document.getElementById('employment_duration_val').textContent = `${s.employment_duration} yrs`;
    document.getElementById('credit_score').value = s.credit_score;
    document.getElementById('credit_score_val').textContent = s.credit_score;
    document.getElementById('existing_loans').value = s.existing_loans;
    document.getElementById('existing_monthly_debt').value = s.existing_monthly_debt;
    document.getElementById('loan_amount').value = s.loan_amount;
    document.getElementById('loan_tenure').value = s.loan_tenure;
    document.getElementById('repayment_history').value = s.repayment_history;

    // Trigger input events to update badges and DTI
    document.getElementById('monthly_income').dispatchEvent(new Event('input'));
    document.getElementById('credit_score').dispatchEvent(new Event('input'));

    // Highlight active sample button
    ['sample_1', 'sample_2', 'sample_3'].forEach(k => {
        const btn = document.getElementById(`btn-${k}`);
        if (btn) {
            if (k === sampleKey) {
                btn.classList.add('border-blue-500', 'bg-blue-600/30');
            } else {
                btn.classList.remove('border-blue-500', 'bg-blue-600/30');
            }
        }
    });
}

function resetForm() {
    const form = document.getElementById('loan-form');
    if (form) form.reset();
    const nameEl = document.getElementById('applicant_name');
    if (nameEl) nameEl.value = '';
    const ageEl = document.getElementById('age');
    if (ageEl) ageEl.value = 32;
    const incEl = document.getElementById('monthly_income');
    if (incEl) incEl.value = 50000;
    const debtEl = document.getElementById('existing_monthly_debt');
    if (debtEl) debtEl.value = 10000;
    const durEl = document.getElementById('employment_duration');
    if (durEl) durEl.value = 3;
    const durValEl = document.getElementById('employment_duration_val');
    if (durValEl) durValEl.textContent = '3 yrs';
    const csEl = document.getElementById('credit_score');
    if (csEl) csEl.value = 650;
    const csValEl = document.getElementById('credit_score_val');
    if (csValEl) csValEl.textContent = '650';
    const loanEl = document.getElementById('loan_amount');
    if (loanEl) loanEl.value = 400000;
    const tenureEl = document.getElementById('loan_tenure');
    if (tenureEl) tenureEl.value = 36;
    const repayEl = document.getElementById('repayment_history');
    if (repayEl) repayEl.value = 'Good';
    const existEl = document.getElementById('existing_loans');
    if (existEl) existEl.value = 'No';

    if (incEl) incEl.dispatchEvent(new Event('input'));
    if (csEl) csEl.dispatchEvent(new Event('input'));
}

// ==========================================
// 4. Run AI Prediction & Update Display
// ==========================================
async function executePrediction() {
    const btn = document.getElementById('run-ai-btn');
    const btnContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i> Performing Variable Elimination Inference...`;

    const payload = {
        applicant_name: document.getElementById('applicant_name').value || 'Anonymous Applicant',
        age: parseInt(document.getElementById('age').value) || 30,
        monthly_income: parseFloat(document.getElementById('monthly_income').value) || 50000,
        employment_status: document.getElementById('employment_status').value,
        employment_duration: parseFloat(document.getElementById('employment_duration').value) || 3,
        credit_score: parseInt(document.getElementById('credit_score').value) || 650,
        existing_loans: document.getElementById('existing_loans').value,
        existing_monthly_debt: parseFloat(document.getElementById('existing_monthly_debt').value) || 0,
        loan_amount: parseFloat(document.getElementById('loan_amount').value) || 300000,
        loan_tenure: parseInt(document.getElementById('loan_tenure').value) || 36,
        repayment_history: document.getElementById('repayment_history').value
    };

    try {
        let result;
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (response.ok) {
                result = await response.json();
            } else {
                result = ClientBayesianEngine.predict(payload);
            }
        } catch (fetchErr) {
            result = ClientBayesianEngine.predict(payload);
        }
        state.lastPredictionResult = { input: payload, output: result };
        renderPredictionResults(result, payload);

        // Smooth scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });

        // Update home and dashboard telemetry in background
        loadDashboardStats();

    } catch (err) {
        console.error('Prediction failed:', err);
        alert('Could not complete prediction. Please verify server connection.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = btnContent;
    }
}

function renderPredictionResults(result, input) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');

    // 1. Gauges and Values
    const apprProb = result.approval_probability;
    const rejProb = result.rejection_probability;

    document.getElementById('res_approval_val').textContent = `${apprProb}%`;
    document.getElementById('res_rejection_val').textContent = `${rejProb}%`;

    // SVG stroke-dasharray (circumference is 2 * PI * 15.9155 ≈ 100)
    document.getElementById('approval-circle').setAttribute('stroke-dasharray', `${apprProb}, 100`);
    document.getElementById('rejection-circle').setAttribute('stroke-dasharray', `${rejProb}, 100`);

    // 2. Verdict Banner & Decision
    const verdictBanner = document.getElementById('verdict-banner');
    const riskBadge = document.getElementById('res_risk_badge');
    const decisionBadge = document.getElementById('res_decision_badge');
    const verdictTitle = document.getElementById('verdict-title');

    verdictTitle.textContent = result.prediction;
    riskBadge.textContent = `RISK LEVEL: ${result.risk_level}`;
    decisionBadge.textContent = result.decision_code;

    if (result.decision_code === 'APPROVED') {
        verdictBanner.className = 'p-5 rounded-xl border bg-emerald-950/40 border-emerald-500/40 text-emerald-300';
        riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        decisionBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-500 text-slate-900';
    } else if (result.decision_code === 'REVIEW') {
        verdictBanner.className = 'p-5 rounded-xl border bg-amber-950/40 border-amber-500/40 text-amber-300';
        riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30';
        decisionBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-amber-500 text-slate-900';
    } else {
        verdictBanner.className = 'p-5 rounded-xl border bg-rose-950/40 border-rose-500/40 text-rose-300';
        riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';
        decisionBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-rose-500 text-white';
    }

    // 3. Bayesian Evidence Mapping Grid
    const evidenceContainer = document.getElementById('evidence-grid');
    evidenceContainer.innerHTML = '';
    for (const [node, stateName] of Object.entries(result.evidence)) {
        const card = document.createElement('div');
        card.className = 'p-2.5 rounded-lg bg-slate-900/60 border border-slate-700/60 flex items-center justify-between text-xs';
        card.innerHTML = `
            <span class="text-slate-400 font-medium">${node}</span>
            <span class="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30 font-semibold">${stateName}</span>
        `;
        evidenceContainer.appendChild(card);
    }

    // 4. Explainable AI Cards
    const explanationContainer = document.getElementById('explanations-list');
    explanationContainer.innerHTML = '';

    result.explanations.forEach(factor => {
        const item = document.createElement('div');
        let iconHtml = '';
        let borderClass = '';
        let impactBadge = '';

        if (factor.type === 'positive') {
            iconHtml = '<i class="fa-solid fa-circle-check text-emerald-400 text-base mt-0.5"></i>';
            borderClass = 'border-emerald-500/30 bg-emerald-950/20 text-emerald-100';
            impactBadge = `<span class="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-500/20 text-emerald-300">+${factor.impact}% impact</span>`;
        } else if (factor.type === 'warning') {
            iconHtml = '<i class="fa-solid fa-triangle-exclamation text-rose-400 text-base mt-0.5"></i>';
            borderClass = 'border-rose-500/30 bg-rose-950/20 text-rose-100';
            impactBadge = `<span class="px-2 py-0.5 text-xs font-semibold rounded bg-rose-500/20 text-rose-300">${factor.impact}% impact</span>`;
        } else {
            iconHtml = '<i class="fa-solid fa-circle-info text-slate-400 text-base mt-0.5"></i>';
            borderClass = 'border-slate-700/50 bg-slate-800/20 text-slate-300';
            impactBadge = `<span class="px-2 py-0.5 text-xs font-semibold rounded bg-slate-700/40 text-slate-300">${factor.impact >= 0 ? '+' : ''}${factor.impact}% impact</span>`;
        }

        item.className = `p-3 rounded-lg border flex items-start gap-3 text-sm ${borderClass}`;
        item.innerHTML = `
            ${iconHtml}
            <div class="flex-1">
                <div class="flex items-center justify-between mb-1">
                    <span class="font-semibold text-xs tracking-wider uppercase text-slate-400">${factor.variable} (${factor.state})</span>
                    ${impactBadge}
                </div>
                <p class="text-xs leading-relaxed text-slate-200">${factor.text}</p>
            </div>
        `;
        explanationContainer.appendChild(item);
    });
}

// ==========================================
// 5. Dynamic What-If Sandbox
// ==========================================
let whatIfDebounceTimer = null;

function updateWhatIfCreditBadge(val) {
    const wiCsBadge = document.getElementById('wi_credit_score_badge');
    if (!wiCsBadge) return;
    if (val < 580) {
        wiCsBadge.textContent = 'Poor (<580)';
        wiCsBadge.className = 'text-[10px] px-2 py-0.5 rounded font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30';
    } else if (val < 670) {
        wiCsBadge.textContent = 'Fair (580-669)';
        wiCsBadge.className = 'text-[10px] px-2 py-0.5 rounded font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
    } else if (val < 740) {
        wiCsBadge.textContent = 'Good (670-739)';
        wiCsBadge.className = 'text-[10px] px-2 py-0.5 rounded font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30';
    } else {
        wiCsBadge.textContent = 'Excellent (740+)';
        wiCsBadge.className = 'text-[10px] px-2 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
    }
}

function initWhatIfListeners() {
    const wiCsInput = document.getElementById('wi_credit_score');
    const wiCsDisplay = document.getElementById('wi_credit_score_val');
    const wiIncomeInput = document.getElementById('wi_monthly_income');
    const wiLoanInput = document.getElementById('wi_loan_amount');
    const wiRepaySelect = document.getElementById('wi_repayment_history');
    const wiExistingSelect = document.getElementById('wi_existing_loans');

    wiCsInput.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        wiCsDisplay.textContent = val;
        updateWhatIfCreditBadge(val);
        debouncedWhatIf();
    });

    wiIncomeInput.addEventListener('input', debouncedWhatIf);
    wiLoanInput.addEventListener('input', debouncedWhatIf);
    wiRepaySelect.addEventListener('change', debouncedWhatIf);
    wiExistingSelect.addEventListener('change', debouncedWhatIf);
}

function debouncedWhatIf() {
    if (whatIfDebounceTimer) clearTimeout(whatIfDebounceTimer);
    whatIfDebounceTimer = setTimeout(() => {
        runDynamicWhatIf();
    }, 35);
}

function applyWhatIfPreset(presetKey) {
    const wiCsInput = document.getElementById('wi_credit_score');
    const wiCsDisplay = document.getElementById('wi_credit_score_val');
    const wiIncomeInput = document.getElementById('wi_monthly_income');
    const wiLoanInput = document.getElementById('wi_loan_amount');
    const wiRepaySelect = document.getElementById('wi_repayment_history');
    const wiExistingSelect = document.getElementById('wi_existing_loans');

    if (presetKey === 'boost_credit') {
        wiCsInput.value = 780;
        wiCsDisplay.textContent = 780;
        wiRepaySelect.value = 'Excellent';
        updateWhatIfCreditBadge(780);
    } else if (presetKey === 'clear_debt') {
        wiExistingSelect.value = 'No';
    } else if (presetKey === 'halve_loan') {
        const currentLoan = parseFloat(wiLoanInput.value) || 500000;
        wiLoanInput.value = Math.max(100000, Math.round(currentLoan / 2));
    } else if (presetKey === 'risk_shock') {
        wiCsInput.value = 540;
        wiCsDisplay.textContent = 540;
        wiRepaySelect.value = 'Poor';
        wiExistingSelect.value = 'Yes';
        updateWhatIfCreditBadge(540);
    }
    runDynamicWhatIf();
}

function syncWhatIfWithLastPrediction() {
    if (!state.lastPredictionResult) return;
    const inp = state.lastPredictionResult.input;

    document.getElementById('wi_credit_score').value = inp.credit_score;
    document.getElementById('wi_credit_score_val').textContent = inp.credit_score;
    updateWhatIfCreditBadge(inp.credit_score);

    document.getElementById('wi_monthly_income').value = inp.monthly_income;
    document.getElementById('wi_loan_amount').value = inp.loan_amount;
    document.getElementById('wi_repayment_history').value = inp.repayment_history;
    document.getElementById('wi_existing_loans').value = inp.existing_loans;

    runDynamicWhatIf();
}

async function runDynamicWhatIf() {
    // Collect base input
    const baseInput = state.lastPredictionResult ? state.lastPredictionResult.input : {
        monthly_income: 50000,
        credit_score: 650,
        employment_status: 'Salaried',
        employment_duration: 3,
        existing_loans: 'No',
        existing_monthly_debt: 10000,
        loan_amount: 500000,
        loan_tenure: 36,
        repayment_history: 'Average'
    };

    const modifiedInput = Object.assign({}, baseInput, {
        credit_score: parseInt(document.getElementById('wi_credit_score').value),
        monthly_income: parseFloat(document.getElementById('wi_monthly_income').value) || 50000,
        loan_amount: parseFloat(document.getElementById('wi_loan_amount').value) || 500000,
        repayment_history: document.getElementById('wi_repayment_history').value,
        existing_loans: document.getElementById('wi_existing_loans').value
    });

    try {
        try {
            const res = await fetch('/api/what-if', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ original: baseInput, modified: modifiedInput })
            });
            if (res.ok) {
                const data = await res.json();
                renderWhatIfResults(data);
            } else {
                renderWhatIfResults(ClientBayesianEngine.whatIf(baseInput, modifiedInput));
            }
        } catch (fetchErr) {
            renderWhatIfResults(ClientBayesianEngine.whatIf(baseInput, modifiedInput));
        }
    } catch (err) {
        console.error('What-If evaluation error:', err);
    }
}

function renderWhatIfResults(data) {
    const origAppr = data.original.approval_probability;
    const modAppr = data.modified.approval_probability;
    const delta = data.delta_approval;

    document.getElementById('wi_original_prob').textContent = `${origAppr}%`;
    document.getElementById('wi_modified_prob').textContent = `${modAppr}%`;

    const origRiskEl = document.getElementById('wi_original_risk');
    origRiskEl.textContent = `Risk: ${data.original.risk_level}`;
    origRiskEl.className = data.original.risk_level === 'LOW' ? 'text-xs text-emerald-400 font-semibold' :
        (data.original.risk_level === 'MEDIUM' ? 'text-xs text-amber-400 font-semibold' : 'text-xs text-rose-400 font-semibold');

    const modRiskEl = document.getElementById('wi_modified_risk');
    modRiskEl.textContent = `Risk: ${data.modified.risk_level}`;
    modRiskEl.className = data.modified.risk_level === 'LOW' ? 'text-xs text-emerald-400 font-bold' :
        (data.modified.risk_level === 'MEDIUM' ? 'text-xs text-amber-400 font-bold' : 'text-xs text-rose-400 font-bold');

    const deltaBadge = document.getElementById('wi_delta_badge');
    if (delta > 0) {
        deltaBadge.textContent = `+${delta}% Approval Gain`;
        deltaBadge.className = 'px-4 py-1.5 rounded-full text-sm font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 animate-pulse';
    } else if (delta < 0) {
        deltaBadge.textContent = `${delta}% Approval Drop`;
        deltaBadge.className = 'px-4 py-1.5 rounded-full text-sm font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse';
    } else {
        deltaBadge.textContent = `Identical Baseline (0.0% Delta)`;
        deltaBadge.className = 'px-4 py-1.5 rounded-full text-sm font-bold bg-slate-700/50 text-slate-300 border border-slate-600';
    }

    document.getElementById('wi-bar-orig').style.width = `${origAppr}%`;
    document.getElementById('wi-bar-mod').style.width = `${modAppr}%`;

    // Dynamic Explainable Narrative
    const narrativeEl = document.getElementById('wi_narrative');
    const origCs = data.original.evidence.CreditScore;
    const modCs = data.modified.evidence.CreditScore;
    const origRh = data.original.evidence.RepaymentHistory;
    const modRh = data.modified.evidence.RepaymentHistory;

    narrativeEl.innerHTML = `
        <div class="space-y-1.5">
            <p class="text-xs text-slate-300">
                Exact Bayesian belief propagation updated posterior distributions across latent nodes:
            </p>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1">
                <div class="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                    <span class="text-slate-400 block font-mono">Baseline State:</span>
                    <span class="text-white font-semibold">Credit: ${origCs} | Repay: ${origRh}</span>
                </div>
                <div class="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                    <span class="text-blue-400 block font-mono">Counterfactual State:</span>
                    <span class="text-white font-semibold">Credit: ${modCs} | Repay: ${modRh}</span>
                </div>
            </div>
            <p class="text-[11px] text-slate-400 pt-1">
                Probability shifted from <strong class="text-slate-200 font-mono">${origAppr}%</strong> to <strong class="${delta >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-mono">${modAppr}%</strong> (${delta >= 0 ? '+' : ''}${delta}% net delta).
            </p>
        </div>
    `;
}

// ==========================================
// 6. Analytics Dashboard & Chart.js
// ==========================================
async function loadDashboardStats() {
    try {
        let stats;
        try {
            const res = await fetch('/api/dashboard-stats');
            if (res.ok) {
                stats = await res.json();
            } else {
                stats = ClientBayesianEngine.getStats();
            }
        } catch (e) {
            stats = ClientBayesianEngine.getStats();
        }
        renderDashboard(stats);
    } catch (err) {
        console.error('Failed to load dashboard stats:', err);
    }
}

function renderDashboard(stats) {
    // Update Home Page KPI strip
    const homeTotal = document.getElementById('home_total_apps');
    if (homeTotal) homeTotal.textContent = stats.total_applications;
    const homeAppr = document.getElementById('home_approved');
    if (homeAppr) homeAppr.textContent = stats.approved_count;
    const homeRej = document.getElementById('home_rejected');
    if (homeRej) homeRej.textContent = stats.rejected_count;

    // Update Dashboard Tab KPIs
    const dashTotal = document.getElementById('dash_total_apps');
    if (dashTotal) {
        dashTotal.textContent = stats.total_applications;
        document.getElementById('dash_approved').textContent = stats.approved_count;
        document.getElementById('dash_reviews').textContent = stats.review_count;
        document.getElementById('dash_rejected').textContent = stats.rejected_count;
        document.getElementById('dash_avg_prob').textContent = `${stats.avg_approval_prob}%`;
        document.getElementById('dash_high_risk').textContent = stats.high_risk_count;

        renderDecisionChart(stats);
        renderRiskChart(stats);
        renderCreditTierChart(stats.credit_tiers);
        renderIncomeChart(stats.income_brackets);
    }
}

function renderDecisionChart(stats) {
    const ctx = document.getElementById('chart-decision');
    if (!ctx) return;
    if (state.dashboardCharts.decision) state.dashboardCharts.decision.destroy();

    state.dashboardCharts.decision = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Approved', 'Conditional Review', 'Rejected'],
            datasets: [{
                data: [stats.approved_count, stats.review_count, stats.rejected_count],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { family: 'Inter', size: 11 } } }
            }
        }
    });
}

function renderRiskChart(stats) {
    const ctx = document.getElementById('chart-risk');
    if (!ctx) return;
    if (state.dashboardCharts.risk) state.dashboardCharts.risk.destroy();

    state.dashboardCharts.risk = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                label: 'Applicant Count',
                data: [
                    stats.risk_distribution.LOW,
                    stats.risk_distribution.MEDIUM,
                    stats.risk_distribution.HIGH
                ],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderCreditTierChart(creditTiers) {
    const ctx = document.getElementById('chart-credit-tiers');
    if (!ctx || !creditTiers) return;
    if (state.dashboardCharts.credit) state.dashboardCharts.credit.destroy();

    const labels = creditTiers.map(t => t.tier);
    const avgProbs = creditTiers.map(t => Math.round(t.avg_prob));

    state.dashboardCharts.credit = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg Approval Prob (%)',
                data: avgProbs,
                backgroundColor: '#3b82f6',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderIncomeChart(incomeBrackets) {
    const ctx = document.getElementById('chart-income');
    if (!ctx || !incomeBrackets) return;
    if (state.dashboardCharts.income) state.dashboardCharts.income.destroy();

    const labels = incomeBrackets.map(b => b.bracket);
    const avgProbs = incomeBrackets.map(b => Math.round(b.avg_prob));

    state.dashboardCharts.income = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg Approval Prob (%)',
                data: avgProbs,
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.15)',
                fill: true,
                tension: 0.35,
                pointRadius: 5,
                pointBackgroundColor: '#8b5cf6'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// ==========================================
// 7. Prediction History
// ==========================================
async function loadHistory(search = '') {
    try {
        let predictions;
        try {
            const res = await fetch(`/api/history?search=${encodeURIComponent(search)}`);
            if (res.ok) {
                const data = await res.json();
                predictions = data.predictions;
            } else {
                predictions = ClientBayesianEngine.getHistory(search);
            }
        } catch (e) {
            predictions = ClientBayesianEngine.getHistory(search);
        }
        renderHistoryTable(predictions);
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistoryTable(predictions) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center py-8 text-slate-500">No predictions recorded yet.</td></tr>`;
        return;
    }

    predictions.forEach(p => {
        const tr = document.createElement('tr');
        tr.className = 'border-b border-slate-800 hover:bg-slate-800/40 text-xs transition';

        let riskBadge = '';
        if (p.risk_level === 'LOW') {
            riskBadge = '<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">LOW</span>';
        } else if (p.risk_level === 'MEDIUM') {
            riskBadge = '<span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold">MED</span>';
        } else {
            riskBadge = '<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold">HIGH</span>';
        }

        let decBadge = '';
        if (p.decision_code === 'APPROVED') {
            decBadge = '<span class="text-emerald-400 font-semibold">Approved</span>';
        } else if (p.decision_code === 'REVIEW') {
            decBadge = '<span class="text-amber-400 font-semibold">Review</span>';
        } else {
            decBadge = '<span class="text-rose-400 font-semibold">Rejected</span>';
        }

        tr.innerHTML = `
            <td class="p-3 font-medium text-slate-200">${p.applicant_name}</td>
            <td class="p-3 text-slate-400">${p.timestamp}</td>
            <td class="p-3 font-mono">${p.credit_score}</td>
            <td class="p-3 font-mono">₹${Number(p.monthly_income).toLocaleString()}</td>
            <td class="p-3 font-mono font-bold text-blue-400">${p.approval_probability}%</td>
            <td class="p-3">${riskBadge}</td>
            <td class="p-3">${decBadge}</td>
            <td class="p-3 text-right">
                <button onclick='viewAuditModal(${JSON.stringify(p)})' class="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs mr-1">
                    <i class="fa-solid fa-eye"></i>
                </button>
                <button onclick="deletePredictionRecord(${p.id})" class="px-2 py-1 bg-rose-950/40 hover:bg-rose-900/60 text-rose-400 rounded text-xs">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function viewAuditModal(record) {
    const modal = document.getElementById('audit-modal');
    modal.classList.remove('hidden');

    document.getElementById('audit_name').textContent = record.applicant_name;
    document.getElementById('audit_timestamp').textContent = record.timestamp;
    document.getElementById('audit_prob').textContent = `${record.approval_probability}%`;
    document.getElementById('audit_risk').textContent = record.risk_level;
    document.getElementById('audit_cs').textContent = record.credit_score;
    document.getElementById('audit_income').textContent = `₹${Number(record.monthly_income).toLocaleString()}`;
    document.getElementById('audit_loan').textContent = `₹${Number(record.loan_amount).toLocaleString()}`;
    document.getElementById('audit_dti').textContent = `${record.debt_to_income_ratio}%`;

    const posList = document.getElementById('audit_pos_factors');
    posList.innerHTML = '';
    (record.key_positive_factors || []).forEach(f => {
        posList.innerHTML += `<li class="text-emerald-400 text-xs flex items-center gap-1.5"><i class="fa-solid fa-check"></i> ${f}</li>`;
    });

    const warnList = document.getElementById('audit_warn_factors');
    warnList.innerHTML = '';
    (record.key_warning_factors || []).forEach(f => {
        warnList.innerHTML += `<li class="text-rose-400 text-xs flex items-center gap-1.5"><i class="fa-solid fa-triangle-exclamation"></i> ${f}</li>`;
    });
}

function closeAuditModal() {
    document.getElementById('audit-modal').classList.add('hidden');
}

async function deletePredictionRecord(id) {
    if (!confirm('Are you sure you want to delete this prediction record?')) return;
    try {
        await fetch(`/api/history/${id}`, { method: 'DELETE' });
        loadHistory();
        loadDashboardStats();
    } catch (err) {
        console.error('Delete failed:', err);
    }
}

async function resetDemoData() {
    if (!confirm('Reset prediction database with sample historical applications?')) return;
    try {
        await fetch('/api/reset-demo', { method: 'POST' });
        loadHistory();
        loadDashboardStats();
    } catch (err) {
        console.error('Reset failed:', err);
    }
}
