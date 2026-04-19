"""
Replication code for the simulation in Section 5.2.3 (Intermediate Substitutability)
of "Promotion, Continuation, and Reputational Volatility under Task-Ease Uncertainty" by Yamagata and Tajika.

This script verifies the claim in the footnote to Proposition 5.
The threshold condition in Proposition 5 (equation 2) depends on p_E:

  Delta_L / Delta_H > [Q(p_E) + (1 - 2Q(p_E)) Q_L(p_E)]
                       / [Q(p_E) + (1 - 2Q(p_E)) Q_H(p_E)]

We check two conditions for 10,000 random parameter configurations:

  (A) Global: the sign of U'(p_E) is constant for ALL p_E in [0, 1].
      This means Delta_L/Delta_H lies outside the range of the threshold
      function, so the "strong complementarity" or "strong substitutability"
      case in Proposition 1 applies without qualification.

  (B) Local: the sign of U'(p_E) is constant on [P_E^f(p_E^0), P_E^s(p_E^0)].
      This is the weaker condition actually needed for Proposition 1, because
      the optimal promotion rule depends only on comparing U at the posterior
      beliefs after success and failure, not on global monotonicity.

Model primitives:
  q_{ad} : success probability given ability a in {H,L} and task ease d in {E,D}
  p_H    : prior probability of high ability
  p_E    : belief that the project is easy
  Delta_H = q_HE - q_HD,  Delta_L = q_LE - q_LD

Ordering restrictions (maintained throughout):
  q_HE >= q_HD,  q_LE >= q_LD  (easy project weakly raises success prob.)
  q_HE >= q_LE,  q_HD >= q_LD  (high ability weakly raises success prob.)

Parameters are drawn uniformly via rejection sampling.
"""

import numpy as np
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_DRAWS    = 10000
PE_GRID_SIZE = 100
SEED         = 2024

# ---------------------------------------------------------------------------
# Model functions (depend on globals set in the main loop)
# ---------------------------------------------------------------------------

def Q(pE):
    """Expected success probability, averaging over ability."""
    return pE * (pH * qHE + (1 - pH) * qLE) + (1 - pE) * (pH * qHD + (1 - pH) * qLD)

def QH(pE):
    """Success probability conditional on high ability."""
    return pE * qHE + (1 - pE) * qHD

def QL(pE):
    """Success probability conditional on low ability."""
    return pE * qLE + (1 - pE) * qLD

def Thres(pE):
    """
    Right-hand side of the threshold condition in Proposition 4.
    U'(p_E) < 0  iff  Delta_L/Delta_H > Thres(p_E).
    """
    return (Q(pE) + (1 - 2 * Q(pE)) * QL(pE)) / (Q(pE) + (1 - 2 * Q(pE)) * QH(pE))

def PEs(pE0):
    """Posterior belief about task ease after success (independent of e)."""
    EqE = pH * qHE + (1 - pH) * qLE
    EqD = pH * qHD + (1 - pH) * qLD
    return pE0 * EqE / (pE0 * EqE + (1 - pE0) * EqD)

def PEf(pE0):
    """Posterior belief about task ease after failure (with e = 1)."""
    EqE = pH * qHE + (1 - pH) * qLE
    EqD = pH * qHD + (1 - pH) * qLD
    return pE0 * (1 - EqE) / (pE0 * (1 - EqE) + (1 - pE0) * (1 - EqD))

def ratio_in_range(ratio, pE_lo, pE_hi):
    """
    Check whether Delta_L/Delta_H lies inside the range of Thres(p_E)
    on the interval [pE_lo, pE_hi].  If it does, the sign of U'
    changes within that interval (the "intermediate" case).
    """
    grid = np.linspace(pE_lo, pE_hi, PE_GRID_SIZE)
    thres_values = [Thres(p) for p in grid]
    return min(thres_values) <= ratio <= max(thres_values)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

random.seed(SEED)

q_value_sets = []

for _ in range(NUM_DRAWS):
    while True:
        qHE_ = random.uniform(0, 1)
        qHD_ = random.uniform(0, 1)
        qLE_ = random.uniform(0, 1)
        qLD_ = random.uniform(0, 1)
        pH_  = random.uniform(0, 1)
        pE0_ = random.uniform(0, 1)

        if qHD_ > qLD_ and qHE_ > qLE_ and qHE_ > qHD_ and qLE_ > qLD_:
            q_value_sets.append({
                'qHE': qHE_, 'qHD': qHD_,
                'qLE': qLE_, 'qLD': qLD_,
                'pH': pH_, 'pE0': pE0_
            })
            break

# ---------------------------------------------------------------------------
# Check both conditions
# ---------------------------------------------------------------------------

count_global = 0  # (A): constant sign on [0, 1]
count_local  = 0  # (B): constant sign on [P_E^f, P_E^s]

for q_set in q_value_sets:
    qHE = q_set['qHE']
    qHD = q_set['qHD']
    qLE = q_set['qLE']
    qLD = q_set['qLD']
    pH  = q_set['pH']
    pE0 = q_set['pE0']

    ratio = (qLE - qLD) / (qHE - qHD)

    # (A) Global: full [0, 1]
    if not ratio_in_range(ratio, 0, 1):
        count_global += 1

    # (B) Local: restricted to [P_E^f(p_E^0), P_E^s(p_E^0)]
    pEf = PEf(pE0)
    pEs = PEs(pE0)
    if not ratio_in_range(ratio, min(pEf, pEs), max(pEf, pEs)):
        count_local += 1

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("=" * 65)
print("Simulation results for Proposition 5 (Intermediate Substitutability)")
print("=" * 65)
print(f"  Number of configurations:  {NUM_DRAWS}")
print(f"  Random seed:               {SEED}")
print(f"  p_E grid size:             {PE_GRID_SIZE}")
print()
print(f"  (A) Constant sign on [0, 1]:          "
      f"{count_global/NUM_DRAWS:.1%}  ({count_global}/{NUM_DRAWS})")
print(f"  (B) Constant sign on [P_E^f, P_E^s]:  "
      f"{count_local/NUM_DRAWS:.1%}  ({count_local}/{NUM_DRAWS})")
print()
print(f"  Condition (B) is the one required for Proposition 1.")
print("=" * 65)
