# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:30:30 2025

@author: eoporter
"""


'''this is a script that applies weights to the outputs of batch predictions of a model'''


def apply_weights_to_data(data_l, case_w, comp_w, data_w):
    for name, w in comp_w.items():
        if name in data_l:
            data_l[name] *= w * data_w * case_w
    return data_l


def apply_weights_to_phys(phys_l, case_w, phys_w):
    names = phys_l.keys()
    for i, name in enumerate(names):
        phys_l[name] *=  (case_w * phys_w)
    return phys_l

def apply_weights_to_constraints(const_l, case_w, const_w):
    names = const_l.keys()
    for i, name in enumerate(names):
        const_l[name] *= (case_w * const_w)
    return const_l



def apply_weights_to_loss_dict(loss_dict, weights_dict, case_name):
    case_w = weights_dict['case'][case_name]
    comp_w = weights_dict['component']
    group_w = weights_dict['group']
    
    # Always apply data weights
    data_l = loss_dict.get('data')
    if data_l is not None:
        data_w = group_w['data']
        data_l = apply_weights_to_data(data_l, case_w, comp_w, data_w)
        loss_dict['data'] = data_l

    # Conditionally apply physics weights
    phys_l = loss_dict.get('phys')
    if phys_l is not None:
        phys_w = group_w['phys']
        phys_l = apply_weights_to_phys(phys_l, case_w, phys_w)
        loss_dict['phys'] = phys_l

    # Conditionally apply constraint weights
    const_l = loss_dict.get('constraint')
    if const_l is not None:
        const_w = group_w['constraint']
        const_l = apply_weights_to_constraints(const_l, case_w, const_w)
        loss_dict['constraint'] = const_l

    return loss_dict
