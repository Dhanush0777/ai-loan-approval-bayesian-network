"""
Live HTTP Verification Script for AI-Based Loan Approval Bayesian Network.
"""

import urllib.request
import json

def post(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    samples_res = urllib.request.urlopen('http://127.0.0.1:5050/api/samples')
    samples = json.loads(samples_res.read().decode('utf-8'))

    print('=== TESTING LIVE HTTP PREDICTIONS ===')
    for key, s in samples.items():
        res = post('http://127.0.0.1:5050/api/predict', s)
        appr = res['approval_probability']
        rej = res['rejection_probability']
        risk = res['risk_level']
        pred = res['prediction']
        factors_count = len(res['explanations'])
        print(f"\n{s['name']}:")
        print(f"   P(Approval): {appr}%, P(Rejection): {rej}% (Sum: {round(appr + rej, 2)}%)")
        print(f"   Risk: {risk}, Verdict: {pred}")
        print(f"   Explanations: {factors_count} factors calculated")
        top_pos = [f['text'] for f in res['explanations'] if f['type'] == 'positive'][:2]
        top_warn = [f['text'] for f in res['explanations'] if f['type'] == 'warning'][:2]
        if top_pos:
            print(f"   Top Positive: {top_pos}")
        if top_warn:
            print(f"   Top Warning: {top_warn}")

    print('\n=== WHAT-IF COUNTERFACTUAL TEST ===')
    wi_payload = {
        'original': samples['sample_2'],
        'modified': dict(samples['sample_2'], credit_score=780, repayment_history='Excellent')
    }
    wi_res = post('http://127.0.0.1:5050/api/what-if', wi_payload)
    print(f"Original Subprime: {wi_res['original']['approval_probability']}% (Risk: {wi_res['original']['risk_level']})")
    print(f"Modified (Score 780, Excellent history): {wi_res['modified']['approval_probability']}% (Risk: {wi_res['modified']['risk_level']})")
    print(f"Approval Delta: +{wi_res['delta_approval']}%")

if __name__ == '__main__':
    main()
