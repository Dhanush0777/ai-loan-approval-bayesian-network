import numpy as np

def compute_cw_cpt():
    cw_table = []
    for cs_idx in range(4):         # Poor=0, Fair=1, Good=2, Excellent=3
        for rh_idx in range(4):     # Poor=0, Average=1, Good=2, Excellent=3
            for el_idx in range(2): # No=0, Yes=1
                has_debt_penalty = 0.35 if el_idx == 1 else 0.0
                # Base score 0 .. 3.0
                score = (cs_idx * 0.52) + (rh_idx * 0.44) - has_debt_penalty
                
                # Smooth continuous logistic-based mapping
                # Low creditworthiness
                p_low = 1.0 / (1.0 + np.exp(2.8 * (score - 0.9)))
                # High creditworthiness
                p_high = 1.0 / (1.0 + np.exp(-2.8 * (score - 1.8)))
                # Medium absorbs the remainder
                remainder = max(0.05, 1.0 - p_low - p_high)
                
                # Normalize
                tot = p_low + remainder + p_high
                p_low, p_med, p_high = p_low / tot, remainder / tot, p_high / tot
                
                cw_table.append((round(p_low, 3), round(p_med, 3), round(p_high, 3)))
    return cw_table

cw = compute_cw_cpt()
print("Sample entries from smooth CPT:")
# Check el_idx=0 (No loans), rh_idx=2 (Good repayment) for all 4 credit scores:
# stride for cs_idx is 4 * 2 = 8
for cs_idx, name in enumerate(['Poor', 'Fair', 'Good', 'Excellent']):
    idx = (cs_idx * 8) + (2 * 2) + 0
    print(f"CS: {name:9s} -> P(Low, Med, High): {cw[idx]}")
