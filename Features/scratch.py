# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:23:18 2025

@author: eoporter
"""

from scipy.optimize import minimize_scalar

# Experimental bounds (from your FOV1 data)
H1 = 15.0987847447  # Top bound in mm (Cy max)
H2 = 24.2896458507  # Bottom bound in mm (abs(Cy min))

# True inlet heights (in mm)
T1 = 50.8   # Top inlet
T2 = 76.2   # Bottom inlet

# Objective function: difference between alphas with given buffer b1
def alpha_difference(b1):
    h1 = H1 - b1
    h2 = H2 - ((T2 / T1) * h1)
    alpha1 = h1 / T1
    alpha2 = h2 / T2
    return abs(alpha1 - alpha2)

# Minimize the difference in alphas
result = minimize_scalar(alpha_difference, bounds=(0, H1), method='bounded')

# Get optimal buffer b1 and calculate b2 and alpha
b1_opt = result.x
h1_opt = H1 - b1_opt
alpha_opt = h1_opt / T1
b2_opt = H2 - alpha_opt * T2

print(f"Optimal buffer for top inlet (b1): {b1_opt:.4f} mm")
print(f"Optimal buffer for bottom inlet (b2): {b2_opt:.4f} mm")
print(f"Consistent scaling factor (alpha): {alpha_opt:.6f}")
