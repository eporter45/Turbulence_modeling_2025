# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:43:48 2025

@author: eoporter
"""

import pandas as pd

df = pd.read_pickle('Case1_FOV1_data.pkl')

print(df.columns)

#['x_mm', 'y_mm', 'U', 'V', 'W', 'V_mag', 'uu', 'vv', 'ww', 'uv', 'uuu',
 #      'uvv', 'uww', 'vuu', 'vvv', 'vww', 'uuuu', 'vvvv', 'wwww'],


#bar (rho ui uj) != bar(rho) bar(ui uj)