"""
07_run_all.py — Run all analysis scripts.
Usage: python3 07_run_all.py
Place in: coil_viz/data_analysis/
"""
import subprocess, sys, time, os

scripts = [
    ('01_efficiency_vs_curvature.py', 'Efficiency vs curvature + ablations'),
    ('02_efficiency_heatmap.py',       'Efficiency heatmaps'),
    ('03_misalignment_analysis.py',    'Misalignment analysis'),
    ('04_lut.py',                      'Lookup tables'),
    ('05_statistical_tests.py',        'Statistical tests'),
    ('06_coupling_analysis.py',        'Coupling & frequency splitting'),
]

print("="*58)
print("  WPT Data Analysis Pipeline")
print("  Run from: coil_viz/data_analysis/")
print("="*58)

t_all = time.time()
for script, label in scripts:
    print(f'\n[{label}]')
    t0 = time.time()
    res = subprocess.run([sys.executable, script],
                         capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f'  ✓ {elapsed:.1f}s')
        for line in res.stdout.strip().split('\n'):
            if 'Saved' in line or line.startswith('  '):
                print(f'    {line.strip()}')
    else:
        print(f'  ✗ ERROR:\n{res.stderr[-400:]}')

print(f'\n{"="*58}')
print(f'  All done in {time.time()-t_all:.1f}s')
print(f'\n  Output files:')
for f in sorted(os.listdir('.')):
    if f.endswith(('.png','.txt')) or ('lut' in f and f.endswith('.csv')):
        print(f'    {f:<42} {os.path.getsize(f)//1024}KB')
print("="*58)
