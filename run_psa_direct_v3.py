"""
PSA script - Growth scenario only (50% -> 61.25% PD prevalence)
500 iterations, 1% population scaled to full.
Produces: PSA_Results_Growth.xlsx with manuscript-ready tables only.
"""

import copy
import sys
import io
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from IBM_PD_AD_v3 import (
    general_config,
    run_probabilistic_sensitivity_analysis,
    save_results_compressed,
)

# ── Configuration ──────────────────────────────────────────────────────────────

PSA_ITERATIONS  = 500
SCALE_FACTOR    = 0.01
SEED            = 42
OUTPUT_DIR      = Path('psa_results_growth')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_LABELS = {
    'total_costs_nhs':        'Total formal healthcare costs (£bn)',
    'total_costs_informal':   'Total informal costs (£bn)',
    'total_costs_all':        'Total societal costs (£bn)',
    'total_qalys_patient':    'Total QALYs – patient (millions)',
    'total_qalys_caregiver':  'Total QALYs – caregiver (millions)',
    'incident_onsets_total':  'Incident onsets (thousands)',
}

SCALE_UNITS = {
    'total_costs_nhs':        1e9,
    'total_costs_informal':   1e9,
    'total_costs_all':        1e9,
    'total_qalys_patient':    1e6,
    'total_qalys_caregiver':  1e6,
    'incident_onsets_total':  1e3,
}

# ── Build config ───────────────────────────────────────────────────────────────

original_pop   = general_config.get('population', 10_787_479)
scaled_pop     = int(original_pop * SCALE_FACTOR)
scale_up       = int(1 / SCALE_FACTOR)

psa_config = copy.deepcopy(general_config)
psa_config['population'] = scaled_pop

# Enable growth scenario
pd_cfg = psa_config['risk_factors']['periodontal_disease']
pd_cfg['prevalence_schedule']['use'] = True

# Scale baseline overrides if present
overrides = psa_config.get('initial_summary_overrides', {})
for key in ('incident_onsets', 'deaths', 'entrants'):
    if key in overrides:
        overrides[key] = int(round(overrides[key] * SCALE_FACTOR))

psa_cfg = copy.deepcopy(psa_config.get('psa', {}))
psa_cfg.update({
    'use':                 True,
    'iterations':          PSA_ITERATIONS,
    'seed':                SEED,
    'original_population': original_pop,
})
psa_config['psa'] = psa_cfg

# ── Run PSA ───────────────────────────────────────────────────────────────────

print(f"Running PSA: {PSA_ITERATIONS} iterations, {scaled_pop:,} agents per draw")
start = datetime.now()

psa_results = run_probabilistic_sensitivity_analysis(
    psa_config,
    psa_cfg,
    collect_draw_level=True,
    seed=SEED,
    n_jobs=1,
)

duration_h = (datetime.now() - start).total_seconds() / 3600
print(f"PSA complete in {duration_h:.2f} hours")

save_results_compressed(psa_results, OUTPUT_DIR / 'psa_results_growth.pkl.gz')

# ── Scale draws ───────────────────────────────────────────────────────────────

draws_df = psa_results['draws'].copy()

count_keywords = {'total', 'cumulative', 'count', 'incident', 'deaths',
                  'onsets', 'mild', 'moderate', 'severe', 'cost', 'qaly'}
rate_keywords  = {'_per_', 'rate', 'ratio', 'proportion', 'mean', 'average', 'median'}

for col in draws_df.columns:
    if col == 'iteration' or not pd.api.types.is_numeric_dtype(draws_df[col]):
        continue
    col_l = col.lower()
    if any(k in col_l for k in rate_keywords):
        continue
    if any(k in col_l for k in count_keywords):
        draws_df[col] *= scale_up

# ── Build manuscript PSA table ────────────────────────────────────────────────

rows = []
for metric, label in METRIC_LABELS.items():
    if metric not in draws_df.columns:
        continue
    unit = SCALE_UNITS.get(metric, 1)
    s    = draws_df[metric] / unit
    sd   = float(s.std())
    mean = float(s.mean())
    cv   = (sd / mean * 100) if mean else 0.0
    rows.append({
        'Outcome': label,
        'Mean':    round(mean, 1),
        '95% CI':  f"{s.quantile(0.025):.1f}–{s.quantile(0.975):.1f}",
        'SD':      round(sd, 1),
        'CV (%)':  round(cv, 2),
    })

psa_table = pd.DataFrame(rows)

# ── Export ────────────────────────────────────────────────────────────────────

excel_path = OUTPUT_DIR / 'PSA_Results_Growth.xlsx'

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    psa_table.to_excel(writer, sheet_name='PSA_Table', index=False)
    draws_df.to_excel(writer, sheet_name='PSA_Draws', index=False)

print(f"Saved: {excel_path}")
