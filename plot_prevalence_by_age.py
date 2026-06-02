"""
Figure: Dementia prevalence rate by age in 2040, England.
Two panels (female left, male right).
Each panel: solid lines = scenario (baseline grey, growth black),
            dashed lines = one-way SA bounds (HR low and HR high) for growth scenario.

Requires four pkl.gz result files:
  - results_pd_baseline.pkl.gz       (50% stable PD, HR=1.21)
  - results_pd_growth.pkl.gz         (growth scenario, HR=1.21)
  - results_pd_growth_hr_low.pkl.gz  (growth scenario, HR=1.07)
  - results_pd_growth_hr_high.pkl.gz (growth scenario, HR=1.38)

Output: figures/Figure_Prevalence_By_Age_2040.png (300 dpi, publication ready)
"""

import sys
import io
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from IBM_PD_AD_v3 import load_results_compressed

# ── File paths ─────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path('results')
FIGURES_DIR  = Path('figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = {
    'baseline':  RESULTS_DIR / 'results_pd_baseline.pkl.gz',
    'growth':    RESULTS_DIR / 'results_pd_growth.pkl.gz',
    'hr_low':    RESULTS_DIR / 'results_pd_growth_hr_low.pkl.gz',
    'hr_high':   RESULTS_DIR / 'results_pd_growth_hr_high.pkl.gz',
}

TARGET_YEAR = 2040

# ── Load and extract prevalence-by-age-by-sex for target year ─────────────────

def extract_prev_by_age(result_key: str, sex: str) -> tuple:
    """
    Returns (age_midpoints, prevalence_pct) arrays from incidence_by_year_sex_df
    for the target year and the given sex.
    """
    path = RESULT_FILES[result_key]
    results = load_results_compressed(path)
    df = results.get('incidence_by_year_sex_df')
    if df is None or df.empty:
        raise ValueError(f"No incidence_by_year_sex_df in {path}")

    sub = df[
        (df['calendar_year'] == TARGET_YEAR) &
        (df['sex'] == sex)
    ].copy()

    if sub.empty:
        raise ValueError(f"No data for year={TARGET_YEAR}, sex={sex} in {path}")

    sub = sub.sort_values('age_lower').reset_index(drop=True)

    # Midpoint for each band; open-ended 90+ band gets midpoint 93
    mids = []
    for _, row in sub.iterrows():
        lo = int(row['age_lower'])
        up = row['age_upper']
        if up is None or (isinstance(up, float) and np.isnan(up)):
            mids.append(93)
        else:
            mids.append((lo + int(up)) / 2)

    prev_pct = sub['dementia_prevalence_in_band'].values * 100

    return np.array(mids), prev_pct


# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

SEXES       = ['female', 'male']
PANEL_LABELS= ['a)', 'b)']

for ax, sex, label in zip(axes, SEXES, PANEL_LABELS):

    # Load all four series for this sex
    ages_b,  prev_b  = extract_prev_by_age('baseline', sex)
    ages_g,  prev_g  = extract_prev_by_age('growth',   sex)
    ages_lo, prev_lo = extract_prev_by_age('hr_low',   sex)
    ages_hi, prev_hi = extract_prev_by_age('hr_high',  sex)

    # Baseline: solid grey
    ax.plot(ages_b,  prev_b,  color='#888888', linewidth=1.8,
            linestyle='-',  label='Baseline (50% stable)')

    # Growth: solid black
    ax.plot(ages_g,  prev_g,  color='#000000', linewidth=1.8,
            linestyle='-',  label='Growth scenario (HR=1.21)')

    # SA bounds for growth: dashed grey and dashed black
    ax.plot(ages_lo, prev_lo, color='#888888', linewidth=1.2,
            linestyle='--', label='Growth – HR=1.07 (low)')
    ax.plot(ages_hi, prev_hi, color='#000000', linewidth=1.2,
            linestyle='--', label='Growth – HR=1.38 (high)')

    # Panel label
    ax.text(0.04, 0.96, label, transform=ax.transAxes,
            fontsize=11, fontweight='normal', va='top', ha='left')

    ax.set_xlabel('Age', fontsize=10)
    ax.set_xlim(65, 97)
    ax.set_xticks([65, 70, 75, 80, 85, 90, 95])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)

axes[0].set_ylabel('Dementia prevalence rate (%)', fontsize=10)

# Single legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.12),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

out_path = FIGURES_DIR / 'Figure_Prevalence_By_Age_2040.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
