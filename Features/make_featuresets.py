# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:00:26 2025

@author: eoporter
"""
FEATURE_SET_DICT = {
    'simp': ['base'],
    'FS0': ['base', 'comp_vec', 'visc_raw', 'rho_UUi_tensor', 'adv_vec'],
    'FS1': ['comp_vec', 'pressure_grad', 'visc_raw', 'adv_vec_decomp'],
    'FS1_simple': ['comp_vec', 'pressure_grad', 'visc_raw', 'adv_vec'],
    'FS2': ['comp_tensor', 'pressure_grad', 'adv_tensor', 'grad_visc'],
    'FS3': ['base', 'pressure_grad', 'velocity_grad'],
    'FS3a': ['base', 'pressure_grad', 'velocity_grad', 'viscous_tensor_raw'],
    'FS3b': ['base', 'pressure_grad', 'velocity_grad', 'k'],
    'FS3c': ['base', 'pressure_grad', 'velocity_grad', 'k', 'viscous_tensor_raw'],
    'FS4': ['base', 'pressure_grad', 'velocity_grad', 'Sij'],
    'FS5': ['velocity_grad', 'pressure_grad', 'grad_visc', 'base'],
    'FS6': ['base', 'velocity_grad', 'pressure_grad', 'Sij', 'Rij', 'visc_raw'],
    'FS7': ['comp_tensor', 'velocity_grad', 'pressure_grad'],
    'FS8':['base', 'pressure_grad', 'Sij', 'Rij', 'k'],
    'FS9': ['base', 'pressure_grad', 'velocity_grad', 'comp_vec', 'adv_vec_decomp', 'viscous_tensor_raw']

}
print(FEATURE_SET_DICT.keys())

CANONICAL_FEATURE_MODULES = {
    #rans model specific
    "simple": ['Ux', 'Uy', 'p', 'rho', 'mu_suth'],
    "density": ['rho'],
    "k": ['k'],
    "mu_t": ['mu_t'],
    "omega": ['omega'],
    "komega": ['k', 'omega'],
    "sst": ['k', 'omega', 'nu_t'],
    #base features
    "base": ['Ux', 'Uy', 'Uz', 'rho', 'p', 'mu_suth'],
    "based": ['Ux', 'Uy', 'Uz', 'rho', 'p','T', 'mu_suth'],

    "velocity_grad": ['dUx_dx', 'dUx_dy', 'dUx_dz','dUy_dx', 'dUy_dy',
                      'dUy_dz','dUz_dx', 'dUz_dy', 'dUz_dz',],
    #pressure Grad
    "pressure_grad": ['dp_dx', 'dp_dy', 'dp_dz'],
    #strain rotation Rates
    "Sij": ['S_xx', 'S_xy', 'S_yy', 'S_xz', 'S_yz', 'S_zz'],
    "Rij": ['R_xy', 'R_xz', 'R_yz'],
    #advection options
    "advection_vec": ['Adv_x', 'Adv_y', 'Adv_z'],
    "advection_vec_decomp": ['strain_adv_x', 'strain_adv_y', 'strain_adv_z',
                             'rot_adv_x', 'rot_adv_y', 'rot_adv_z'],
    "advection_tensor": ['Adv_xx', 'Adv_xy', 'Adv_xz', 'Adv_yx', 'Adv_yy', 
                         'Adv_yz','Adv_zx', 'Adv_zy', 'Adv_zz'],
    
    #compressibility options
    "compressibility_vec": ['comp_x', 'comp_y', 'comp_z'],
    "compressibility_tensor": ['comp_xx', 'comp_xy', 'comp_xz',
                               'comp_yx', 'comp_yy','comp_yz',
                               'comp_zx', 'comp_zy', 'comp_zz'],
    #viscous diffusion options
    "viscous_tensor_raw": ['Tao_xx', 'Tao_yy', 'Tao_zz',
                           'Tao_xy', 'Tao_xz', 'Tao_yz'],
    "grad_viscous_tensor": ['dTao_xx_dx', 'dTao_xx_dy', 'dTao_xx_dz',
                        'dTao_yy_dx', 'dTao_yy_dy', 'dTao_yy_dz',
                        'dTao_zz_dx', 'dTao_zz_dy', 'dTao_zz_dz',
                        'dTao_xy_dx', 'dTao_xy_dy', 'dTao_xy_dz',
                        'dTao_xz_dx', 'dTao_xz_dy', 'dTao_xz_dz',
                        'dTao_yz_dx', 'dTao_yz_dy', 'dTao_yz_dz',],
    "div_viscous_tensor":['div_Tao_x', 'div_Tao_y', 'div_Tao_z'],
    "cell_geometry": ['cellID', 'Cx', 'Cy', 'Cz'],
    "reynolds_stresses": ['uv', 'uu', 'vv', 'ww'],
    "rho_UUi_tensor": ['rho_UxUx', 'rho_UxUy', 'rho_UxUz',
                       'rho_UyUy', 'rho_UyUz', 'rho_UzUz'],
    
    "momentum_residuals": ['resid_mom_x', 'resid_mom_y', 'resid_mom_z'],
    "flow_invariants": ['Q_crit', 'Q_norm_S', 'dUi_dxi', 'Mach'],
    "grad_rho_T": ['drho_dx', 'drho_dy', 'drho_dz', 'dT_dx', 'dT_dy', 'dT_dz'],
    "div_rho_uiuj": ['div_rho_uiuj_x', 'div_rho_uiuj_y', 'div_rho_uiuj_z'],
    "velocity_hessian": [
                        'ddUx_dx_dx', 'ddUx_dx_dy', 'ddUx_dx_dz',
                        'ddUx_dy_dx', 'ddUx_dy_dy', 'ddUx_dy_dz',
                        'ddUy_dx_dx', 'ddUy_dx_dy', 'ddUy_dx_dz',
                        'ddUy_dy_dx', 'ddUy_dy_dy', 'ddUy_dy_dz',
                        'ddUz_dx_dx', 'ddUz_dx_dy', 'ddUz_dx_dz',
                        'ddUz_dy_dx', 'ddUz_dy_dy', 'ddUz_dy_dz'
                        ],
    "grad_rho_UUi_tensor": [
                            'drho_UxUx_dx', 'drho_UxUx_dy', 'drho_UxUx_dz',
                            'drho_UxUy_dx', 'drho_UxUy_dy', 'drho_UxUy_dz',
                            'drho_UxUz_dx', 'drho_UxUz_dy', 'drho_UxUz_dz',
                            'drho_UyUy_dx', 'drho_UyUy_dy', 'drho_UyUy_dz',
                            'drho_UyUz_dx', 'drho_UyUz_dy', 'drho_UyUz_dz',
                            'drho_UzUz_dx', 'drho_UzUz_dy', 'drho_UzUz_dz'
                            ],

}

ALIAS_KEYS = {
    'adv_vec': 'advection_vec',
    'adv_vec_decomp': 'advection_vec_decomp',
    'adv_tensor': 'advection_tensor',
    'comp_vec': 'compressibility_vec',
    'comp_tensor': 'compressibility_tensor',
    'visc_raw': 'viscous_tensor_raw',
    'visc_deriv': 'grad_viscous_tensor',
    'grad_visc': 'grad_viscous_tensor',
    'div_visc': 'div_viscous_tensor',
    'grad_vel': 'velocity_grad',
    'grad_u': 'velocity_grad',
    'grad_U': 'velocity_grad',
    'gradU': 'velocity_grad',
    'basic': 'base',
    'base_state': 'base',
    'S_ij': 'Sij',
    'R_ij': 'Rij'
}

ZERO_FEATURES_2D = [
    # Velocity
    'Uz',
    # Gradients
    'dUx_dz', 'dUy_dz', 'dUz_dx', 'dUz_dy', 'dUz_dz',
    'dp_dz',
    # Strain tensor
    'S_xz', 'S_zx', 'S_yz', 'S_zy', 'S_zz',
    # Rotation tensor
    'R_xz', 'R_zx', 'R_yz', 'R_zy', 'R_zz',
    # Viscous tensor
    'Tao_xz', 'Tao_zx', 'Tao_yz', 'Tao_zy', 'Tao_zz',
    # Gradients of viscous tensor
    'dTao_xx_dz', 'dTao_yy_dz', 'dTao_zz_dz',
    'dTao_xy_dz', 'dTao_xz_dx', 'dTao_xz_dy', 'dTao_xz_dz',
    'dTao_yz_dx', 'dTao_yz_dy', 'dTao_yz_dz',
    # Advection vector
    'Adv_z',
    # Advection tensor
    'Adv_xz', 'Adv_yz', 'Adv_zx', 'Adv_zy', 'Adv_zz',
    # Compressibility vector & tensor
    'comp_z',
    'comp_xz', 'comp_yz', 'comp_zx', 'comp_zy', 'comp_zz',
    # Stress-like advection terms
    'rho_UxUz', 'rho_UyUz', 'rho_UzUz',
    # Viscous divergence
    'div_Tao_z',
    # Decomposed advection (if implemented)
    'strain_adv_z', 'rot_adv_z',
]




def trim_features(feature_list):
    return [f for f in feature_list if f not in ZERO_FEATURES_2D]


# === Universal Builder ===
def build_feature_set(feature_keys, trim_z=False):
    feature_set = []

    for key in feature_keys:
        canonical_key = ALIAS_KEYS.get(key, key)
        if canonical_key not in CANONICAL_FEATURE_MODULES:
            raise ValueError(f"[Error] Unsupported feature key: '{key}'")

        feature_set += CANONICAL_FEATURE_MODULES[canonical_key]

    # Deduplicate and trim if needed
    feature_set = list(set(feature_set))
    if trim_z:
        feature_set = trim_features(feature_set)

    return feature_set

def get_feature_set(config):
    feat_cfg = config['features']
    feature_spec = feat_cfg.get('input', [])
    trim_z = feat_cfg.get('trim_z', False)
    
    if isinstance(feature_spec, str):
        if feature_spec not in FEATURE_SET_DICT:
            raise ValueError(f"[ERROR] Feature set '{feature_spec}' not defined.")
        feature_keys = FEATURE_SET_DICT[feature_spec]
    elif isinstance(feature_spec, list):
        feature_keys = feature_spec
    else:
        raise TypeError("[ERROR] 'features' must be a string or a list.")
    
    return build_feature_set(feature_keys, trim_z=trim_z)

'''Visualization of featuresets, pre minmax/abs norming'''

# Assume you have pre-loaded unnormalized DataFrames per case


'''
VALIDATION TESTING
#test 1
config1 = {
    "features": {
        "input": "FS1",
        "trim_z": True
    }
}
print('Test1')
features1 = get_feature_set(config1)
print("Features FS1 (z-trimmed):")
print(sorted(features1))

#test2
config2 = {
    "features": {
        "input": ['base', 'grad_vel', 'pressure_grad'],
        "trim_z": True
    }
}
print('Test2')

features2 = get_feature_set(config2)
print("Custom feature list with trim:")
print(sorted(features2))
#test3
config3 = {
    "features": {
        "input": "FS2",
        "trim_z": False
    }
}
print('TEST3')
features3 = get_feature_set(config3)
print("Features FS2 (untrimmed):")
print(sorted(features3))
#test 4
#print('Test4')
config4 = {
    "features": {
        "input": ['base', 'dp_dx', 'dp_dy', 'dp_dz', 'k'],
        "trim_z": True
    }
}

#features4 = get_feature_set(config4)
#print("Mixed base + raw features (z-trimmed):")
#print(sorted(features4))

#test5
print('Test5')
config5 = {
    "features": {
        "input": ['base', 'nonexistent_module'],
        "trim_z": False
    }
}

try:
    features5 = get_feature_set(config5)
except ValueError as e:
    print("Caught expected error:", e)
'''