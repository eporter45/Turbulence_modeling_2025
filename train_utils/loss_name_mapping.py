# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:06:25 2025

@author: eoporter
"""

# loss_name_maps.py

PHYSICS_LOSS_MAP = {
    'TKE': [
        'k', 'tke', 'turb_kinetic_energy', 'turbulent kinetic energy',
        'turbulent_kinetic_energy', 'k'
    ],
    'B_ij': [
        'bij', 'b_ij', 'normalized_anisotropy', 'norm_aij',
        'norm_anisotropy', 'norm_a'
    ],
    'A_ij': [
        'aij', 'a_ij', 'anisotropy', 'a'
    ]
}

CONSTRAINT_LOSS_MAP = {
    'A_ij': [   # Same as physics
        'aij', 'a_ij', 'anisotropy', 'a'
    ],
    'B_ij': [   # Same as physics
        'bij', 'b_ij', 'normalized_anisotropy', 'norm_aij',
        'norm_anisotropy', 'norm_a'
    ]
}
def normalize_loss_keys(loss_keys, domain='physics'):
    """
    Normalize loss keys from config to canonical keys using alias lists.

    domain: 'physics' or 'constraint'
    """
    if domain == 'physics':
        loss_map = PHYSICS_LOSS_MAP
    elif domain == 'constraint':
        loss_map = CONSTRAINT_LOSS_MAP
    else:
        raise ValueError(f"Invalid domain '{domain}'")

    # Create reverse lookup dict for fast alias->canonical
    alias_to_key = {}
    for canonical_key, aliases in loss_map.items():
        for alias in aliases:
            alias_to_key[alias.lower()] = canonical_key

    normalized = []
    for key in loss_keys:
        k = key.lower()
        if k not in alias_to_key:
            raise ValueError(
                f"Unsupported loss key '{key}' in domain '{domain}'. "
                f"Supported aliases: {sorted(alias_to_key.keys())}"
            )
        normalized.append(alias_to_key[k])

    # Remove duplicates but preserve order
    seen = set()
    normalized_unique = []
    for item in normalized:
        if item not in seen:
            normalized_unique.append(item)
            seen.add(item)
    return normalized_unique
