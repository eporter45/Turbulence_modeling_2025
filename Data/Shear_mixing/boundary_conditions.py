# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:09:09 2025
Boundary conditions for the experimental data
@author: eoporter
"""
x1_ref = 50.8 #mm
x2_ref = 76.2 #mm
# Mc = (U1 - U2)/ (a1 + a2)
case1 = {'1': {'M': 0.463,
               'M_tol': 0.012,
               'P0': 109.32,
               'P0_tol': 0.05, 
               'P': 93.94,
               'P_tol': 0.1,
               'T0': 296.1, 
               'T0_tol': 0.5,
               'T': 283.98,
               'T_tol': 0.84, 
               'U': 156.25,
               'U_tol': 4.3,
               'x_ref': 0.0508,
               'rho': 1.1524053074974712,
               'm_dot': 9.147217128261177,
                'mu_ref': 1.7691074562977978e-05, #kg/ms 
                },
         '2': {'M': 0.089, 
               'M_tol': 0.009,
               'P0': 94.56,
               'P0_tol': 0.05,
               'P': 93.92,
               'P_tol': 0.1,
               'T0': 292.48,
               'T0_tol': 0.5,
               'T': 292.02,
               'T_tol': 0.51,
               'U': 30.35,
               'U_tol': 3.0, 
               'x_ref': 0.0762,
               'rho': 1.120438274605702, 
               'm_dot': 2.591203984532369,
               'mu_ref': 1.8079099774173828e-05,
               },
         'Relative': {'U_ratio': 0.194, 
                      'U_ratio_tol': 0.02,
                      'rho_ratio': 0.972,
                      'rho_ratio_tol': 0.004,
                      'Mc': 0.185, 
                      'Mc_tol': 0.008,
                      'm_dot_net': 11.738421112793546},
         'Reference': {'rho_ref': 1.1453487278938863,
                       'T_ref': 285.7547940575189, 
                       'P_ref': 93.93558508940917,
                       'U_ref': 128.45813783064327, 
                       'm_dot_1': 9.147217128261177,
                       'm_dot_2': 2.591203984532369, 
                       'm_dot_net': 11.738421112793546,
                       'delta_U': 125.9,
                       'x_ref': 0.0508,
                       'rho_U_ref': 147.1293647519448,
                       'mu_ref':  1.7777178584365316e-05
                       }
         }
case2 = {'1': {'M': 1.003,
               'M_tol': 0.021,
               'P0': 151.84,
               'P0_tol': 0.05, 
               'P': 80.37,
               'P_tol': 0.11,
               'T0': 294.6, 
               'T0_tol': 0.5,
               'T': 245.25, 
               'T_tol': 2.04,
               'U': 314.89, 
               'U_tol': 6.32, 
               'x_ref': 0.0508,
               'rho': 1.1416353318876455,
               'm_dot': 18.262069122631516, 
               'mu_ref': 1.5744474663270607e-05,
               },
         '2': {'M': 0.189,
               'M_tol': 0.009,
               'P0': 82.47,
               'P0_tol': 0.05,
               'P': 80.59,
               'P_tol': 0.11,
               'T0': 293.58,
               'T0_tol': 0.5,
               'T': 291.49,
               'T_tol': 0.54,
               'U': 64.75,
               'U_tol': 2.96,
               'x_ref': 0.0762, 
               'rho': 0.9631633405810408,
               'm_dot': 4.752199764259826,
               'mu_ref': 1.8053680552092157e-05,
                   }, 
         'Relative': {'U_ratio': 0.206,
                      'U_ratio_tol': 0.01, 
                      'rho_ratio': 0.844, 
                      'rho_ratio_tol': 0.007,
                      'Mc': 0.381, 
                      'Mc_tol': 0.011, 
                      'm_dot_net': 23.014268886891344},
          'Reference': {'rho_ref': 1.1047827792741964, 
                       'T_ref': 254.79806421091814,
                       'P_ref': 80.4154276411419,
                       'U_ref': 263.23877202164647,
                       'm_dot_1': 18.262069122631516,
                       'm_dot_2': 4.752199764259826, 
                       'm_dot_net': 23.014268886891344,
                       'delta_U': 250.14, 
                       'x_ref': 0.0508,
                       'rho_U_ref': 290.82166216680116,
                       'mu_ref': 1.6236902498556162e-05

                       }
         }
case3 = {'1': {'M': 1.571,
                'M_tol': 0.025,
                'P0': 270.41, 
                'P0_tol': 0.05,
                'P': 62.02,
                'P_tol': 0.1,
                'T0': 284.59,
                'T0_tol': 0.5,
                'T': 190.5,
                'T_tol': 2.68, 
                'U': 434.76, 
                'U_tol': 6.08,
                'x_ref': 0.0508,
                'rho': 1.1341728077406106,
                'm_dot': 25.04912287058004,
                'mu_ref': 1.2739657302082796e-05, #kg/ms 
                },
          '2': {'M': 0.285,
                'M_tol': 0.014, 
                'P0': 71.47, 
                'P0_tol': 0.05,
                'P': 66.29,
                'P_tol': 0.1,
                'T0': 295.57,
                'T0_tol': 0.5,
                'T': 290.85,
                'T_tol': 0.69,
                'U': 97.28,
                'U_tol': 4.87,
                'x_ref': 0.0762,
                'rho': 0.7940016404057121,
                'm_dot': 5.885724543894477,
                'mu_ref': 1.8022955902869732e-05, #kg/ms 

                },
          'Relative': {'U_ratio': 0.224,
                       'U_ratio_tol': 0.012,
                       'rho_ratio': 0.7, 
                       'rho_ratio_tol': 0.01,
                       'Mc': 0.546,
                       'Mc_tol': 0.013,
                       'm_dot_net': 30.934847414474515},
         'Reference': {'rho_ref': 1.0694511764397419, 
                       'T_ref': 209.59278717513413,
                       'P_ref': 62.83241854746211,
                       'U_ref': 370.5503954572569,
                       'm_dot_1': 25.04912287058004,
                       'm_dot_2': 5.885724543894477,
                       'm_dot_net': 30.934847414474515,
                       'delta_U': 337.48,
                       'x_ref': 0.0508,
                       'rho_U_ref': 396.285556351975,
                       'mu_ref': 1.3824892837527229e-05
                       }
         }
case4 = {'1': {'M': 1.955,
               'M_tol': 0.021,
               'P0': 445.5, 
               'P0_tol': 0.06,
               'P': 57.58, 
               'P_tol': 0.1,
               'T0': 298.02,
               'T0_tol': 0.5,
               'T': 168.87,
               'T_tol': 2.26,
               'U': 509.24, 
               'U_tol': 4.35,
               'x_ref': 0.0508,
               'rho': 1.1878500107691454,
               'm_dot': 30.728957565791244,
               'mu_ref':1.1456235519494888e-05,
               },
         '2': {'M': 0.269,
               'M_tol': 0.008,
               'P0': 63.83, 
               'P0_tol': 0.05,
               'P': 61.51,
               'P_tol': 0.1,
               'T0': 298.22,
               'T0_tol': 0.5,
               'T': 293.96, 
               'T_tol': 0.56,
               'U': 92.57,
               'U_tol': 2.79,
               'x_ref': 0.0762,
               'rho': 0.7289536873004133,
               'm_dot': 5.141918303905023,
               'mu_ref': 1.8171954197722954e-05,
                   },
         'Relative': {'U_ratio': 0.182,
                      'U_ratio_tol': 0.006,
                      'rho_ratio': 0.614,
                      'rho_ratio_tol': 0.008,
                      'Mc': 0.69, 
                      'Mc_tol': 0.009,
                      'm_dot_net': 35.87087586969627},
         'Reference': {'rho_ref': 1.1220694200244405,
                       'T_ref': 186.80105256119094, 
                       'P_ref': 58.14334668291213, 
                       'U_ref': 449.51235054223815,
                       'm_dot_1': 30.728957565791244,
                       'm_dot_2': 5.141918303905023,
                       'm_dot_net': 35.87087586969627, 
                       'delta_U': 416.67,
                       'x_ref': 0.0508,
                       'rho_U_ref': 504.3840624667522,
                       'mu_ref': 1.2524376039008286e-05, #kg/ms
                       } 
         }

BCs = {'Case1': case1, 
       'Case2': case2,
       'Case3': case3,
       'Case4': case4}


def make_reference_dict(case_dict):
    R_air = 287.05  # J/(kg·K)
    depth = 1.0     # m, assumed constant for both inlets

    # Deep copy to avoid modifying original input
    data = case_dict

    # 1. Compute rho for each stream
    for stream in ['1', '2']:
        P = data[stream]['P'] * 1000  # convert kPa to Pa
        T = data[stream]['T']
        rho = P / (R_air * T)
        data[stream]['rho'] = rho

    # 2. Compute mass flux (m_dot) for each stream
    for stream in ['1', '2']:
        rho = data[stream]['rho']
        print(f'{stream}, rho {rho}')
        U = data[stream]['U']
        x_ref = data[stream]['x_ref']
        A = x_ref * depth
        m_dot = rho * U * A
        data[stream]['m_dot'] = m_dot

    # 3. Compute total mass flux
    m_dot_1 = data['1']['m_dot']
    m_dot_2 = data['2']['m_dot']
    m_dot_net = m_dot_1 + m_dot_2
    data['Relative']['m_dot_net'] = m_dot_net

    # 4. Helper to compute mass-flux-averaged scalar
    def mass_flux_avg(field):
        val1 = data['1'][field]
        val2 = data['2'][field]
        return (m_dot_1 * val1 + m_dot_2 * val2) / m_dot_net

    # 5. Build reference dict with useful averaged fields
    ref_dict = {
        'rho_ref': mass_flux_avg('rho'),
        'T_ref': mass_flux_avg('T'),
        'P_ref': mass_flux_avg('P'),
        'U_ref': mass_flux_avg('U'),
        'm_dot_1': m_dot_1,
        'm_dot_2': m_dot_2,
        'm_dot_net': m_dot_net,
        'delta_U': abs(data['1']['U'] - data['2']['U']),
        'x_ref': data['1']['x_ref'],  # Typically the smaller/higher speed stream
        'rho_U_ref': mass_flux_avg('U') * mass_flux_avg('rho')  # Optional composite
    }

    return ref_dict

def sutherland_viscosity(T_ref):
    """
    Compute reference dynamic viscosity of air using Sutherland's law.

    Parameters:
        T_ref (float): Reference static temperature in Kelvin.

    Returns:
        mu_ref (float): Dynamic viscosity in kg/(m·s)
    """
    # Constants for air
    mu_0 = 1.716e-5       # Viscosity at T_0 [kg/(m·s)]
    T_0 = 273.15          # Reference temperature [K]
    S = 110.4             # Sutherland's constant for air [K]

    mu_ref = mu_0 * (T_ref / T_0)**1.5 * (T_0 + S) / (T_ref + S)
    return mu_ref


