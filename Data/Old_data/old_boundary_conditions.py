# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:00:16 2025

@author: eoporter
"""
import numpy as np

case1_bc = {
    "name": "case_1",

    # --- Fundamental gas constants (air) ---
    "R": 287.058,                   # Specific gas constant [J/(kg·K)]
    "R_u": 8.31445,                 # Universal gas constant [J/(mol·K)]
    "molar_mass": 0.0289647,       # [kg/mol]
    "gamma": 1.4,                   # Ratio of specific heats
    "Pr": 0.70,                     # Prandtl number
    "Sutherland": 110.4,            # Sutherland constant [K]

    # --- Reference conditions (inlet total) ---
    "T_ref": 295.6,                 # K (also total inlet temp)
    "p_ref": 74480.76,              # Pa (total pressure at inlet)
    "p_in_static": 69650.0,         # Pa (static pressure at inlet)
    "p_out_static": 51722.75,       # Pa (static pressure at outlet)
    "T_total": 295.6,               # K (same as T_ref)
    "mu_ref": 18.7e-6,              # Pa·s (dynamic viscosity at T_ref)
    "M_in": 0.311,
    "M_out": 0.741,

    # --- Geometry ---
    "C_ax": 0.130023,               # Axial chord [m] (L_ref)

    # --- Derived thermo properties ---
    "Cp": 1 / (1.4 - 1),                          # Cp / R = 1 / (γ - 1)
    "Cv": 1 / (1.4 * (1.4 - 1)),                  # Cv / R = 1 / (γ(γ - 1))

    # --- Reference scalars (computed) ---
    "rho_ref": 69650.0 / (287.058 * 295.6),       # Inlet static: p / (R T)
    "u_ref": np.sqrt(1.4 * 287.058 * 295.6),      # √(γ R T)
    "k_ref": (18.7e-6 * (1 / (1.4 - 1))) / 0.70,   # μ Cp / Pr
    "nu_ref": 18.7e-6 / (69650.0 / (287.058 * 295.6)),  # μ / ρ
    "alpha_ref": (18.7e-6 / (69650.0 / (287.058 * 295.6))) / 0.70,  # ν / Pr
    "h_ref": (1 / (1.4 - 1)) * 295.6,             # Cp * T_ref
    "e_ref": (1 / (1.4 * (1.4 - 1))) * 295.6,     # Cv * T_ref

    # --- Reynolds number based on total inlet pressure ---
    "Re": (74480.76 * 0.130023) / (18.7e-6 * np.sqrt(1.4 * 287.058 * 295.6)),

    # --- Non-dimensional outputs ---
    "p_out_norm": 51722.75 / 74480.76,            # p_out / p_ref
    "T_out_norm": (51722.75 / 287.058) / (74480.76 / 287.058),  # (p_out/ρ_out)/p_ref/ρ_ref
    "rho_out_norm": (51722.75 / (287.058 * 295.6)) / (69650.0 / (287.058 * 295.6)),  # ρ_out / ρ_ref
    "u_in_norm": 0.311,                           # M_in (optional velocity scaling)
    "u_out_norm": 0.741                           # M_out
}

case_bcs = {'case_1': case1_bc}