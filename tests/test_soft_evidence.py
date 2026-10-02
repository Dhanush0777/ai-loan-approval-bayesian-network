import numpy as np

def compute_credit_score_soft_evidence(cs):
    """
    Computes soft evidence distribution P(CreditScore = s | cs)
    using smooth Gaussian/triangular kernel membership functions centered at:
    Poor: 450, Fair: 625, Good: 705, Excellent: 790.
    """
    cs = float(np.clip(cs, 300, 850))
    # Centers and scales
    centers = {
        'Poor': 480.0,
        'Fair': 625.0,
        'Good': 705.0,
        'Excellent': 790.0
    }
    stds = {
        'Poor': 65.0,
        'Fair': 45.0,
        'Good': 45.0,
        'Excellent': 60.0
    }
    
    # Compute Gaussian density
    weights = {}
    for state, center in centers.items():
        std = stds[state]
        # For endpoints, give appropriate half-normal support
        if state == 'Poor' and cs <= center:
            w = 1.0
        elif state == 'Excellent' and cs >= center:
            w = 1.0
        else:
            w = np.exp(-0.5 * ((cs - center) / std) ** 2)
        weights[state] = float(w)
        
    total_w = sum(weights.values())
    probs = {s: w / total_w for s, w in weights.items()}
    return probs

print("Testing soft evidence distributions:")
for score in [500, 580, 625, 670, 705, 740, 790, 840]:
    p = compute_credit_score_soft_evidence(score)
    print(f"CS {score}: " + ", ".join([f"{k}: {v*100:4.1f}%" for k, v in p.items()]))
