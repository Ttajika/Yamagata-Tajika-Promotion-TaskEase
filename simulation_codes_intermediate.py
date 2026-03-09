"""
Replication code for the simulation in Section 5.4 (Intermediate Substitutability)
of "When to Promote, When to Continue" by Yamagata and Tajika.

This script verifies the claim in the footnote to Proposition 4:
for a random sample of 10,000 parameter configurations (q_HE, q_HD, q_LE, q_LD, p_H),
the threshold condition (equation XX) is either satisfied or violated for ALL p_E in [0,1]
in approximately 81.8% of cases. That is, the sign of U'(p_E) is uniform across p_E
for most parameter configurations, so the "strong complementarity" and "strong
substitutability" cases in Proposition 1 cover the bulk of the parameter space.

Model primitives:
  - q_{ad}: success probability given ability a in {H,L} and task ease d in {E,D}
  - p_H:   prior probability of high ability
  - p_E:   belief that the project is easy
  - Delta_H = q_HE - q_HD, Delta_L = q_LE - q_LD

The threshold from Proposition 4 is:
  Delta_L / Delta_H > [Q(p_E) + (1 - 2Q(p_E)) Q_L(p_E)] / [Q(p_E) + (1 - 2Q(p_E)) Q_H(p_E)]

If Delta_L/Delta_H lies outside the range of the right-hand side over p_E in [0,1],
then U'(p_E) has a constant sign for all p_E, and our main results apply without
the "strong" qualification.
"""

import numpy as np
import random

# ---------------------------------------------------------------------------
# Model functions
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
    Right-hand side of the threshold condition in Proposition 4 (equation XX).
    U'(p_E) < 0 iff Delta_L/Delta_H > Thres(p_E).
    """
    return (Q(pE) + (1 - 2 * Q(pE)) * QL(pE)) / (Q(pE) + (1 - 2 * Q(pE)) * QH(pE))

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

NUM_DRAWS = 10000          # number of parameter configurations
PE_GRID   = np.linspace(0, 1, 100)  # grid for p_E

random.seed(2024)  # for reproducibility

q_value_sets = []

# Draw parameter configurations uniformly, subject to the ordering restrictions:
#   q_HE >= q_HD,  q_LE >= q_LD   (easy project weakly raises success prob.)
#   q_HE >= q_LE,  q_HD >= q_LD   (high ability weakly raises success prob.)
# We use rejection sampling: draw four values and keep only if all inequalities hold.
for _ in range(NUM_DRAWS):
    while True:
        qHE = random.uniform(0, 1)
        qHD = random.uniform(0, 1)
        qLE = random.uniform(0, 1)
        qLD = random.uniform(0, 1)
        pH  = random.uniform(0, 1)

        if qHD > qLD and qHE > qLE and qHE > qHD and qLE > qLD:
            q_value_sets.append({
                'qHE': qHE, 'qHD': qHD,
                'qLE': qLE, 'qLD': qLD,
                'pH': pH
            })
            break

# For each configuration, check whether Delta_L/Delta_H lies inside
# the range of Thres(p_E) over p_E in [0,1].
# If it does, the sign of U'(p_E) changes within [0,1] (the "intermediate" case).
# If it does not, U'(p_E) has a constant sign (our main results apply directly).

intersection_count = 0  # counts configurations where the sign changes

for q_set in q_value_sets:
    qHE = q_set['qHE']
    qHD = q_set['qHD']
    qLE = q_set['qLE']
    qLD = q_set['qLD']
    pH  = q_set['pH']

    # Evaluate the threshold function on the p_E grid
    thres_values = [Thres(p) for p in PE_GRID]

    # Compute Delta_L / Delta_H
    ratio = (qLE - qLD) / (qHE - qHD)

    # Check whether the ratio falls inside [min Thres, max Thres]
    if min(thres_values) <= ratio <= max(thres_values):
        intersection_count += 1

# Report the fraction where the sign does NOT change (main results apply)
fraction_uniform_sign = 1 - intersection_count / NUM_DRAWS
print(f"Fraction of configurations where U'(p_E) has constant sign: {fraction_uniform_sign:.3f}")
print(f"({NUM_DRAWS} configurations, p_E grid size = {len(PE_GRID)})")
