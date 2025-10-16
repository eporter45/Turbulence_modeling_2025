# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:23:40 2025

@author: eoporter
"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri

def tricontour_plot(df, field='rho', levels=20, cmap='viridis', filled=True):
    cx = df['Cx'].values
    cy = df['Cy'].values
    values = df[field].values
    triang = Triangulation(cx, cy)

    plt.figure(figsize=(8, 6))
    if filled:
        tcf = plt.tricontourf(triang, values, levels=levels, cmap=cmap)
        plt.colorbar(tcf, label=field)
    else:
        tc = plt.tricontourf(triang, values, levels=levels, colors='k', linewidths=0.8)
        plt.clabel(tc, fmt='%.2f', inline=True, fontsize=8)
    plt.xlabel('Cx')
    plt.ylabel('Cy')
    plt.title(f'{field} field (tricontour{"f" if filled else ""})')
    plt.aspect('equal')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


from matplotlib.tri import Triangulation

def tripcolor_plot(df, field='rho', cmap='viridis', show=True):
    """
    Plots a scalar field using tripcolor and returns the figure object.

    Parameters:
    - df: DataFrame with 'Cx', 'Cy', and the target field.
    - field: Name of the field to plot.
    - cmap: Colormap.
    - show: Whether to display the plot using plt.show().

    Returns:
    - fig: The matplotlib Figure object.
    """
    cx = df['Cx'].values
    cy = df['Cy'].values
    values = df[field].values

    triang = Triangulation(cx, cy)

    fig, ax = plt.subplots(figsize=(8, 6))
    tpc = ax.tripcolor(triang, values, shading='gouraud', cmap=cmap)
    fig.colorbar(tpc, ax=ax, label=field)
    ax.set_xlabel('Cx')
    ax.set_ylabel('Cy')
    ax.set_title(f'{field} field (tripcolor)')
    ax.set_aspect('equal')
    ax.set_xlim(cx.min(), cx.max())
    ax.set_ylim(cy.min(), cy.max())
    ax.grid(False)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def scatter_colormap_plot(df, field='rho', s=10, cmap='viridis'):
    cx = df['Cx']
    cy = df['Cy']
    values = df[field]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(cx, cy, c=values, cmap=cmap, s=s, edgecolor='none')
    plt.colorbar(sc, label=field)
    plt.xlabel('Cx')
    plt.ylabel('Cy')
    plt.title(f'{field} field (scatter)')
    plt.grid(False)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
   
    
#snippet for saving feature plots   
'''
import os
import matplotlib.pyplot as plt
import sys
from matplotlib.tri import Triangulation
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
filename= 'TE_engr_grads_0p9ax1p6.pkl'
load_path = os.path.join(PROJECT_ROOT, 'Data', 'RANS_Dicts', filename)
df = pd.read_pickle(load_path)
cols = list(df.columns)
save_path  = os.path.join(PROJECT_ROOT, 'Output_plots', 'Feat_ENG_v2')
for col in cols:
    fig = tripcolor_plot(df, col, cmap='jet', show=False)
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f'{col}.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Plots saved to:\n - {fig_path}\n - {fig_path}")

'''
    