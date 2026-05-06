"""
04_lut.py
Lookup tables: X,Y,θ,h → efficiency.
Run from: coil_viz/data_analysis/
"""
import pandas as pd
import numpy as np

DATA = '../data/wpt_results.csv'
OUT  = './'

df   = pd.read_csv(DATA)
full = df[df['ablation_name'] == 'full_model']

# Full LUT
lut = full[['conic_curve_mm','x_cord','y_cord','angle','alignment','group',
            'd_lateral_mm','efficiency','k','kQ','Q_rx','M_nH',
            'S21_param','coupling_regime']].copy()
lut.columns = ['sagitta_h_mm','x_mm','y_mm','angle_deg','alignment','group',
               'd_lateral_mm','efficiency_pct','k','kQ','Q_rx','M_nH',
               'S21_dB','coupling_regime']
lut = lut.sort_values(['sagitta_h_mm','angle_deg','x_mm','y_mm'])
lut.to_csv(f'{OUT}lut_full.csv', index=False)
print(f'Saved: lut_full.csv  ({len(lut)} rows)')

# Aligned θ=0
lut_a = lut[lut.angle_deg == 0]
lut_a.to_csv(f'{OUT}lut_aligned.csv', index=False)
print(f'Saved: lut_aligned.csv  ({len(lut_a)} rows)')

# Pivot: X offset vs height (θ=0, Y=0)
pivot_xh = full[(full.alignment=='x-axis') & (full.angle==0) &
                (full.y_cord==0)].pivot_table(
    values='efficiency', index='x_cord', columns='conic_curve_mm'
).round(5)
pivot_xh.index.name   = 'x_offset_mm'
pivot_xh.columns.name = 'sagitta_h_mm'
pivot_xh.to_csv(f'{OUT}lut_pivot_xh.csv')
print('Saved: lut_pivot_xh.csv')

# Pivot: angle vs height (x=y=0)
pivot_ah = full[(full.x_cord==0) & (full.y_cord==0)].pivot_table(
    values='efficiency', index='angle', columns='conic_curve_mm'
).round(5)
pivot_ah.index.name   = 'angle_deg'
pivot_ah.columns.name = 'sagitta_h_mm'
pivot_ah.to_csv(f'{OUT}lut_pivot_angleh.csv')
print('Saved: lut_pivot_angleh.csv')
print(pivot_ah.to_string())

# Summary per height
summary = full.groupby('conic_curve_mm')['efficiency'].agg([
    'mean','max','std','median',
    ('pct_nonzero', lambda x: (x>0).mean()*100)
]).round(5)
summary.index.name = 'sagitta_h_mm'
summary.columns    = ['mean_eta_pct','max_eta_pct','std_eta',
                      'median_eta_pct','pct_poses_nonzero']
summary.to_csv(f'{OUT}lut_summary.csv')
print('\nSaved: lut_summary.csv')
print(summary.to_string())
