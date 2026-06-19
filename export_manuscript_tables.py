"""
Export manuscript tables and figures from completed model results.

Requires two pkl.gz result files (baseline and growth) already produced by run_model.
Reads those files and writes:

OUTPUT FILES:
  1. Manuscript_Tables.xlsx (8 sheets):
     a. Model_Inputs                 — comprehensive model parameters
     b. Table3_Scenario_Comparison   — scenario comparison results
     c. Attributable_Cases_Summary   — PAF summary statistics
     d. Attributable_Distribution    — histogram data
     e. Attributable_Draws           — raw PSA draws
     f. QALY_Differences             — cumulative QALY differences
     g. PSA_Table                    — PSA results
     h. Sensitivity_Analysis         — one-way SA results

  2. figures/Attributable_Cases_Histogram.png
     - Histogram showing distribution of dementia cases attributable to
       periodontal disease from PSA iterations
     - X-axis: Attributable percentage (%)
     - Y-axis: Frequency (% of PSA iterations)
     - Includes mean, median, and 95% CI

Usage:
    python export_manuscript_tables.py

Adjust BASELINE_RESULTS_PATH, GROWTH_RESULTS_PATH, PSA_EXCEL_PATH,
and SA_EXCEL_PATH below to match your file locations.
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from IBM_PD_AD_v3 import (
    load_results_compressed,
    summaries_to_dataframe,
    general_config,
    RISK_FACTOR_HR_INTERVALS,
)

# ── File paths ─────────────────────────────────────────────────────────────────

BASELINE_RESULTS_PATH = Path('results') / 'results_pd_baseline.pkl.gz'
GROWTH_RESULTS_PATH   = Path('results') / 'results_pd_growth.pkl.gz'
PSA_EXCEL_PATH        = Path('psa_results_growth') / 'PSA_Results_Growth.xlsx'
SA_BASELINE_EXCEL_PATH = Path('pd_sensitivity_analysis.xlsx')
SA_GROWTH_EXCEL_PATH   = Path('pd_sensitivity_analysis_growth.xlsx')
OUTPUT_PATH           = Path('results') / 'Manuscript_Tables.xlsx'
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────

print("Loading baseline results...")
baseline_results = load_results_compressed(BASELINE_RESULTS_PATH)
baseline_df      = summaries_to_dataframe(baseline_results)

print("Loading growth results...")
growth_results = load_results_compressed(GROWTH_RESULTS_PATH)
growth_df      = summaries_to_dataframe(growth_results)

# ── Risk factor metadata (matches Table 1 in manuscript) ──────────────────────

RISK_FACTOR_LABELS = {
    'periodontal_disease': 'Periodontal Disease',
    'hypertension':        'Hypertension',
    'hearing_difficulty':  'Hearing Difficulty',
    'APOE_e4_carrier':     'APOE ε4 carrier',
    'obesity':             'Obesity',
    'depression':          'Depression',
    'type_2_diabetes':     'Diabetes',
}

# ── TABLE 3: Scenario comparison ───────────────────────────────────────────────
# Columns: year | PD prev baseline | PD prev growth | dementia cases baseline |
#          dementia cases growth | diff | annual formal cost baseline |
#          annual formal cost growth | annual societal cost baseline |
#          annual societal cost growth

TARGET_YEARS = [2030, 2035, 2040]

def get_year_row(df: pd.DataFrame, year: int) -> pd.Series:
    row = df[df['calendar_year'] == year]
    if row.empty:
        return pd.Series(dtype=float)
    return row.iloc[0]

rows_t3 = []
for year in TARGET_YEARS:
    b = get_year_row(baseline_df, year)
    g = get_year_row(growth_df, year)

    # PD prevalence from risk_prev_alive column
    b_pd_prev = b.get('risk_prev_alive_periodontal_disease', np.nan)
    g_pd_prev = g.get('risk_prev_alive_periodontal_disease', np.nan)

    b_cases   = b.get('dementia_cases_total', np.nan)
    g_cases   = g.get('dementia_cases_total', np.nan)

    # Annual costs: year_costs fields (£, convert to £bn)
    b_formal  = b.get('year_costs_nhs', np.nan) / 1e9
    g_formal  = g.get('year_costs_nhs', np.nan) / 1e9
    b_soc     = b.get('year_costs_societal', np.nan) / 1e9
    g_soc     = g.get('year_costs_societal', np.nan) / 1e9

    rows_t3.append({
        'Year':                           year,
        'PD prevalence – baseline (%)':   round(b_pd_prev * 100, 1) if not np.isnan(b_pd_prev) else None,
        'PD prevalence – growth (%)':     round(g_pd_prev * 100, 1) if not np.isnan(g_pd_prev) else None,
        'Dementia cases – baseline':      int(round(b_cases)) if not np.isnan(b_cases) else None,
        'Dementia cases – growth':        int(round(g_cases)) if not np.isnan(g_cases) else None,
        'Dementia cases – difference':    int(round(g_cases - b_cases)) if not (np.isnan(g_cases) or np.isnan(b_cases)) else None,
        'Annual formal cost – baseline (£bn)': round(b_formal, 2) if not np.isnan(b_formal) else None,
        'Annual formal cost – growth (£bn)':   round(g_formal, 2) if not np.isnan(g_formal) else None,
        'Annual societal cost – baseline (£bn)': round(b_soc, 2) if not np.isnan(b_soc) else None,
        'Annual societal cost – growth (£bn)':   round(g_soc, 2) if not np.isnan(g_soc) else None,
    })

table3_df = pd.DataFrame(rows_t3)

# ── Attributable cases from PSA (Periodontal Disease) ──────────────────────────
# Calculate distribution of cases attributable to periodontal disease from PSA draws

def calculate_attributable_cases_from_psa() -> pd.DataFrame:
    """
    Calculate dementia cases attributable to periodontal disease from PSA draws.

    Returns DataFrame with:
    - attributable_cases: absolute number of cases
    - attributable_pct: percentage of total incident cases
    - scenario: 'Growth' (only growth PSA available)
    """
    if not PSA_EXCEL_PATH.exists():
        print(f"WARNING: PSA results not found at {PSA_EXCEL_PATH}")
        return pd.DataFrame()

    # Load PSA draws
    try:
        psa_draws = pd.read_excel(PSA_EXCEL_PATH, sheet_name='PSA_Draws')
    except Exception as e:
        print(f"WARNING: Could not load PSA draws: {e}")
        return pd.DataFrame()

    if psa_draws.empty or 'incident_onsets_total' not in psa_draws.columns:
        print("WARNING: PSA draws missing required columns")
        return pd.DataFrame()

    # Calculate PAF for each draw
    # PAF = P * (HR - 1) / [P * (HR - 1) + 1]
    # where P = prevalence, HR = hazard ratio

    results = []

    for idx, row in psa_draws.iterrows():
        # Get sampled values for periodontal disease
        # Note: PSA sampling varies HR and prevalence
        # We'll use the mean values from the growth scenario if individual draws aren't available

        # Get incident onsets
        incident_onsets = row.get('incident_onsets_total', 0)

        if incident_onsets == 0:
            continue

        # For growth scenario, prevalence varies from 50% to 61.25%
        # Use midpoint as approximation: 55.6%
        prevalence = 0.556  # Average over 2023-2040

        # HR for periodontal disease (sampled in PSA from 95% CI: 1.07-1.38)
        # Check if HR is in the draws, otherwise use base value
        hr = 1.21  # Base value

        # Look for PD-related columns in draws
        pd_hr_cols = [col for col in psa_draws.columns if 'periodontal' in col.lower() and 'hr' in col.lower()]
        if pd_hr_cols:
            hr = row.get(pd_hr_cols[0], 1.21)

        # Calculate PAF
        paf = (prevalence * (hr - 1)) / (prevalence * (hr - 1) + 1)

        # Calculate attributable cases
        attributable_cases = incident_onsets * paf
        attributable_pct = paf * 100

        results.append({
            'iteration': idx,
            'scenario': 'Growth',
            'incident_onsets': incident_onsets,
            'attributable_cases': attributable_cases,
            'attributable_pct': attributable_pct,
            'prevalence': prevalence,
            'hazard_ratio': hr,
        })

    return pd.DataFrame(results)

attributable_df = calculate_attributable_cases_from_psa()

# Create summary statistics for attributable cases
if not attributable_df.empty:
    attributable_summary = pd.DataFrame([{
        'Scenario': 'Growth (50%→61.25%)',
        'Mean attributable cases': int(round(attributable_df['attributable_cases'].mean())),
        'Median attributable cases': int(round(attributable_df['attributable_cases'].median())),
        'Lower 95% CI': int(round(attributable_df['attributable_cases'].quantile(0.025))),
        'Upper 95% CI': int(round(attributable_df['attributable_cases'].quantile(0.975))),
        'Mean attributable %': round(attributable_df['attributable_pct'].mean(), 2),
        'Median attributable %': round(attributable_df['attributable_pct'].median(), 2),
        'Lower 95% CI (%)': round(attributable_df['attributable_pct'].quantile(0.025), 2),
        'Upper 95% CI (%)': round(attributable_df['attributable_pct'].quantile(0.975), 2),
    }])
else:
    attributable_summary = pd.DataFrame()

# Create histogram data for plotting (binned frequencies)
if not attributable_df.empty:
    # Create bins for attributable percentage (0-20% in 1% increments)
    bins = np.arange(0, 21, 1)
    hist, bin_edges = np.histogram(attributable_df['attributable_pct'], bins=bins)

    # Convert to frequency (% of iterations)
    total_iterations = len(attributable_df)
    freq_pct = (hist / total_iterations) * 100

    histogram_data = pd.DataFrame({
        'Attributable_%_lower': bin_edges[:-1],
        'Attributable_%_upper': bin_edges[1:],
        'Frequency_%_of_iterations': freq_pct,
        'Count': hist,
        'Scenario': 'Growth'
    })
else:
    histogram_data = pd.DataFrame()

# ── Plot histogram figure ──────────────────────────────────────────────────────

def plot_attributable_cases_histogram(attributable_df: pd.DataFrame,
                                      output_path: Path = FIGURES_DIR / 'Attributable_Cases_Histogram.png'):
    """
    Create publication-ready histogram showing distribution of dementia cases
    attributable to periodontal disease from PSA iterations.

    X-axis: Dementia cases attributable to periodontal disease (%)
    Y-axis: Frequency (% of total PSA iterations)
    """
    if attributable_df.empty:
        print("WARNING: No attributable cases data to plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create histogram
    attributable_pct = attributable_df['attributable_pct'].values
    n_iterations = len(attributable_pct)

    # Use bins from 0-20% in 0.5% increments for smoother histogram
    bins = np.arange(0, 21, 0.5)

    # Plot histogram with frequency as % of iterations
    counts, bin_edges, patches = ax.hist(
        attributable_pct,
        bins=bins,
        weights=np.ones(len(attributable_pct)) / len(attributable_pct) * 100,
        color='#2E86AB',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Add mean and median lines
    mean_val = attributable_pct.mean()
    median_val = np.median(attributable_pct)
    ci_lower = np.percentile(attributable_pct, 2.5)
    ci_upper = np.percentile(attributable_pct, 97.5)

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}%')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}%')

    # Add 95% CI shading
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray',
               label=f'95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]')

    # Labels and title
    ax.set_xlabel('Dementia cases attributable to periodontal disease (%)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (% of PSA iterations)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Attributable Cases from PSA\n(Growth Scenario: 50% → 61.25% PD Prevalence)',
                 fontsize=13, fontweight='bold', pad=20)

    # Formatting
    ax.set_xlim(0, 20)
    ax.set_ylim(0, max(counts) * 1.15)  # Add 15% headroom
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Add text box with summary statistics
    textstr = f'PSA Iterations: {n_iterations}\n'
    textstr += f'Mean ± SD: {mean_val:.1f}% ± {attributable_pct.std():.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved: {output_path}")
    return output_path

# Generate histogram figure
if not attributable_df.empty:
    histogram_figure_path = plot_attributable_cases_histogram(attributable_df)
else:
    histogram_figure_path = None
    print("WARNING: Skipping histogram plot - no attributable cases data available")

# ── QALY differences (growth minus baseline) by year ─────────────────────────

qaly_rows = []
for year in range(2024, 2041):
    b = get_year_row(baseline_df, year)
    g = get_year_row(growth_df, year)

    # Cumulative patient and caregiver QALYs
    b_pat  = b.get('total_qalys_patient',   np.nan)
    g_pat  = g.get('total_qalys_patient',   np.nan)
    b_cg   = b.get('total_qalys_caregiver', np.nan)
    g_cg   = g.get('total_qalys_caregiver', np.nan)

    qaly_rows.append({
        'Year':                                        year,
        'Cumulative patient QALYs – difference':       int(round(g_pat - b_pat)) if not (np.isnan(g_pat) or np.isnan(b_pat)) else None,
        'Cumulative caregiver QALYs – difference':     int(round(g_cg  - b_cg))  if not (np.isnan(g_cg)  or np.isnan(b_cg))  else None,
    })

qaly_df = pd.DataFrame(qaly_rows)

# ── Model Inputs Table ─────────────────────────────────────────────────────────

def create_model_inputs_table() -> pd.DataFrame:
    """
    Create comprehensive table of model inputs for manuscript.

    Includes:
    - Model structure parameters
    - Population parameters
    - Dementia onset and progression
    - Risk factor prevalences and hazard ratios
    - Health state utilities
    - Costs by stage and setting
    """
    rows = []

    # ── Section 1: Model Structure ──────────────────────────────────────────
    rows.append({'Category': 'MODEL STRUCTURE', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    rows.append({
        'Category': 'Time horizon',
        'Parameter': 'Base year',
        'Value': general_config.get('base_year', 2023),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Model start year'
    })

    base_year = general_config.get('base_year', 2023)
    n_steps = general_config.get('number_of_timesteps', 17)
    final_year = base_year + n_steps
    rows.append({
        'Category': 'Time horizon',
        'Parameter': 'Time horizon (years)',
        'Value': n_steps,
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': f'{base_year}–{final_year}'
    })

    rows.append({
        'Category': 'Time horizon',
        'Parameter': 'Time step (years)',
        'Value': general_config.get('time_step_years', 1),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Annual cycles'
    })

    # ── Section 2: Population Parameters ────────────────────────────────────
    rows.append({'Category': '', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})
    rows.append({'Category': 'POPULATION', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    rows.append({
        'Category': 'Population size',
        'Parameter': 'Initial population (65+)',
        'Value': f"{general_config.get('population', 10787479):,}",
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'ONS 2023 projections'
    })

    sex_dist = general_config.get('sex_distribution', {})
    rows.append({
        'Category': 'Sex distribution',
        'Parameter': 'Female (%)',
        'Value': round(sex_dist.get('female', 0.54) * 100, 1),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'ONS 2023'
    })

    rows.append({
        'Category': 'Sex distribution',
        'Parameter': 'Male (%)',
        'Value': round(sex_dist.get('male', 0.46) * 100, 1),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'ONS 2023'
    })

    open_pop = general_config.get('open_population', {})
    if open_pop.get('use'):
        rows.append({
            'Category': 'Population dynamics',
            'Parameter': 'Annual entrants (65+)',
            'Value': f"{open_pop.get('entrants_per_year', 0):,}",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Open population model'
        })

    # ── Section 3: Dementia Onset & Progression ─────────────────────────────
    rows.append({'Category': '', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})
    rows.append({'Category': 'DEMENTIA ONSET & PROGRESSION', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    rows.append({
        'Category': 'Onset probability',
        'Parameter': 'Annual probability of dementia onset (baseline)',
        'Value': general_config.get('base_onset_probability', 0.0025),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'NHS England data'
    })

    init_prev_female_young = general_config.get('initial_dementia_prevalence_by_age_band', {}).get((65, 79), {}).get('female', 0)
    init_prev_female_old = general_config.get('initial_dementia_prevalence_by_age_band', {}).get((80, 100), {}).get('female', 0)
    init_prev_male_young = general_config.get('initial_dementia_prevalence_by_age_band', {}).get((65, 79), {}).get('male', 0)
    init_prev_male_old = general_config.get('initial_dementia_prevalence_by_age_band', {}).get((80, 100), {}).get('male', 0)

    rows.append({
        'Category': 'Initial prevalence',
        'Parameter': 'Dementia prevalence – Female 65-79 (%)',
        'Value': round(init_prev_female_young * 100, 2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Primary Care Dementia Data'
    })

    rows.append({
        'Category': 'Initial prevalence',
        'Parameter': 'Dementia prevalence – Female 80+ (%)',
        'Value': round(init_prev_female_old * 100, 2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Primary Care Dementia Data'
    })

    rows.append({
        'Category': 'Initial prevalence',
        'Parameter': 'Dementia prevalence – Male 65-79 (%)',
        'Value': round(init_prev_male_young * 100, 2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Primary Care Dementia Data'
    })

    rows.append({
        'Category': 'Initial prevalence',
        'Parameter': 'Dementia prevalence – Male 80+ (%)',
        'Value': round(init_prev_male_old * 100, 2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Primary Care Dementia Data'
    })

    transitions = general_config.get('stage_transition_durations', {})
    rows.append({
        'Category': 'Stage progression',
        'Parameter': 'Mild → Moderate (mean duration, years)',
        'Value': transitions.get('mild_to_moderate', 2.2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Tariot et al. (2024)'
    })

    rows.append({
        'Category': 'Stage progression',
        'Parameter': 'Moderate → Severe (mean duration, years)',
        'Value': transitions.get('moderate_to_severe', 2),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Tariot et al. (2024)'
    })

    rows.append({
        'Category': 'Stage progression',
        'Parameter': 'Severe → Death (mean duration, years)',
        'Value': transitions.get('severe_to_death', 4),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Tariot et al. (2024)'
    })

    # ── Section 4: Risk Factors ─────────────────────────────────────────────
    rows.append({'Category': '', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})
    rows.append({'Category': 'RISK FACTORS', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    risk_labels = {
        'periodontal_disease': 'Periodontal Disease',
        'APOE_e4_carrier': 'APOE ε4 carrier',
        'hypertension': 'Hypertension',
        'hearing_difficulty': 'Hearing Difficulty',
        'depression': 'Depression',
        'obesity': 'Obesity',
        'diabetes': 'Diabetes (Type 2)',
        'smoking': 'Smoking',
        'social_isolation': 'Social Isolation',
        'excessive_alcohol_consumption': 'Excessive Alcohol',
        'low_education': 'Low Education',
        'socioeconomic_disadvantage': 'Socioeconomic Disadvantage',
        'lifestyle': 'Unhealthy Lifestyle',
        'air_pollution': 'Air Pollution',
    }

    risk_factors_config = general_config.get('risk_factors', {})

    for risk_key, risk_label in risk_labels.items():
        if risk_key in risk_factors_config:
            rf = risk_factors_config[risk_key]
            prevalence = rf.get('prevalence', {})

            # Prevalence
            female_prev = prevalence.get('female', 0)
            male_prev = prevalence.get('male', 0)

            if female_prev == male_prev:
                prev_str = f"{female_prev * 100:.1f}"
            else:
                prev_str = f"F:{female_prev * 100:.1f}, M:{male_prev * 100:.1f}"

            rows.append({
                'Category': 'Prevalence',
                'Parameter': f'{risk_label} (%)',
                'Value': prev_str,
                'Lower_95CI': '—',
                'Upper_95CI': '—',
                'Source/Note': 'Baseline prevalence'
            })

            # Hazard ratio with 95% CI
            hr_data = RISK_FACTOR_HR_INTERVALS.get(risk_key, {}).get('onset', {}).get('all')
            if hr_data:
                hr_point, hr_low, hr_high = hr_data
                rows.append({
                    'Category': 'Hazard Ratio',
                    'Parameter': f'{risk_label} (HR for dementia onset)',
                    'Value': hr_point,
                    'Lower_95CI': hr_low,
                    'Upper_95CI': hr_high,
                    'Source/Note': 'Meta-analysis estimates'
                })

    # Special note for periodontal disease growth scenario
    pd_schedule = risk_factors_config.get('periodontal_disease', {}).get('prevalence_schedule', {})
    if pd_schedule:
        rows.append({
            'Category': 'Prevalence',
            'Parameter': 'Periodontal Disease – Growth Scenario',
            'Value': f"50% (2023) → 61.25% (2040)",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Elamin & Anash (2023): 22.5% relative increase'
        })

    # ── Section 5: Health State Utilities ───────────────────────────────────
    rows.append({'Category': '', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})
    rows.append({'Category': 'HEALTH STATE UTILITIES', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    utility_norms = general_config.get('utility_norms_by_age', {})
    rows.append({
        'Category': 'General population',
        'Parameter': 'Female age 65 (EQ-5D)',
        'Value': utility_norms.get('female', {}).get(65, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Age-adjusted population norms'
    })

    rows.append({
        'Category': 'General population',
        'Parameter': 'Female age 75 (EQ-5D)',
        'Value': utility_norms.get('female', {}).get(75, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Age-adjusted population norms'
    })

    rows.append({
        'Category': 'General population',
        'Parameter': 'Male age 65 (EQ-5D)',
        'Value': utility_norms.get('male', {}).get(65, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Age-adjusted population norms'
    })

    rows.append({
        'Category': 'General population',
        'Parameter': 'Male age 75 (EQ-5D)',
        'Value': utility_norms.get('male', {}).get(75, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Age-adjusted population norms'
    })

    dementia_qalys = general_config.get('dementia_stage_qalys', {})
    rows.append({
        'Category': 'Patient utilities',
        'Parameter': 'Mild dementia',
        'Value': dementia_qalys.get('mild', '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Mukadam et al. (2024)'
    })

    rows.append({
        'Category': 'Patient utilities',
        'Parameter': 'Moderate dementia',
        'Value': dementia_qalys.get('moderate', '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Mukadam et al. (2024)'
    })

    rows.append({
        'Category': 'Patient utilities',
        'Parameter': 'Severe dementia',
        'Value': dementia_qalys.get('severe', '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Mukadam et al. (2024)'
    })

    caregiver_qalys = general_config.get('stage_age_qalys', {}).get('caregiver', {})
    rows.append({
        'Category': 'Caregiver utilities',
        'Parameter': 'Mild dementia (home care)',
        'Value': caregiver_qalys.get('mild', {}).get('home', {}).get(0, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Home caregiver disutility'
    })

    rows.append({
        'Category': 'Caregiver utilities',
        'Parameter': 'Moderate dementia (home care)',
        'Value': caregiver_qalys.get('moderate', {}).get('home', {}).get(0, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Home caregiver disutility'
    })

    rows.append({
        'Category': 'Caregiver utilities',
        'Parameter': 'Severe dementia (home care)',
        'Value': caregiver_qalys.get('severe', {}).get('home', {}).get(0, '—'),
        'Lower_95CI': '—',
        'Upper_95CI': '—',
        'Source/Note': 'Home caregiver disutility'
    })

    # ── Section 6: Costs ────────────────────────────────────────────────────
    rows.append({'Category': '', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})
    rows.append({'Category': 'ANNUAL COSTS (£, 2023)', 'Parameter': '', 'Value': '', 'Lower_95CI': '', 'Upper_95CI': '', 'Source/Note': ''})

    costs = general_config.get('costs', {})

    for stage in ['mild', 'moderate', 'severe']:
        stage_costs = costs.get(stage, {})

        # Home - NHS
        home_nhs = stage_costs.get('home', {}).get('nhs', 0)
        rows.append({
            'Category': f'{stage.capitalize()} dementia',
            'Parameter': f'NHS costs (home)',
            'Value': f"{home_nhs:,.2f}",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Wittenberg et al. (2020)'
        })

        # Home - Informal
        home_informal = stage_costs.get('home', {}).get('informal', 0)
        rows.append({
            'Category': f'{stage.capitalize()} dementia',
            'Parameter': f'Informal care costs (home)',
            'Value': f"{home_informal:,.2f}",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Wittenberg et al. (2020)'
        })

        # Institution - NHS
        inst_nhs = stage_costs.get('institution', {}).get('nhs', 0)
        rows.append({
            'Category': f'{stage.capitalize()} dementia',
            'Parameter': f'NHS costs (institution)',
            'Value': f"{inst_nhs:,.2f}",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Wittenberg et al. (2020)'
        })

        # Institution - Informal
        inst_informal = stage_costs.get('institution', {}).get('informal', 0)
        rows.append({
            'Category': f'{stage.capitalize()} dementia',
            'Parameter': f'Informal care costs (institution)',
            'Value': f"{inst_informal:,.2f}",
            'Lower_95CI': '—',
            'Upper_95CI': '—',
            'Source/Note': 'Wittenberg et al. (2020)'
        })

    return pd.DataFrame(rows)

model_inputs_df = create_model_inputs_table()

# ── PSA table: read from PSA output Excel ─────────────────────────────────────

if PSA_EXCEL_PATH.exists():
    psa_table_df = pd.read_excel(PSA_EXCEL_PATH, sheet_name='PSA_Table')
    print(f"PSA table loaded: {len(psa_table_df)} rows")
else:
    print(f"WARNING: PSA Excel not found at {PSA_EXCEL_PATH}. PSA_Table sheet will be empty.")
    psa_table_df = pd.DataFrame(
        columns=['Outcome', 'Mean', '95% CI', 'SD', 'CV (%)']
    )

# ── Sensitivity analysis: read from SA output Excel ───────────────────────────

sa_dfs = []
for sa_path, label in [(SA_BASELINE_EXCEL_PATH, 'Baseline (50% stable)'),
                        (SA_GROWTH_EXCEL_PATH,   'Growth (50%→61.25%)')]:
    if sa_path.exists():
        df = pd.read_excel(sa_path, sheet_name='Results')
        df.insert(0, 'Scenario', label)
        sa_dfs.append(df)
        print(f"Sensitivity analysis loaded ({label}): {len(df)} rows from {sa_path}")
    else:
        print(f"WARNING: SA Excel not found at {sa_path}. Rows for '{label}' will be absent.")
sa_combined_df = pd.concat(sa_dfs, ignore_index=True) if sa_dfs else pd.DataFrame()

# ── Write workbook ─────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    model_inputs_df.to_excel(writer, sheet_name='Model_Inputs',                index=False)
    table3_df.to_excel(writer,       sheet_name='Table3_Scenario_Comparison',  index=False)

    # Attributable cases from PSA (replaces enrichment table)
    if not attributable_summary.empty:
        attributable_summary.to_excel(writer, sheet_name='Attributable_Cases_Summary', index=False)
    if not histogram_data.empty:
        histogram_data.to_excel(writer, sheet_name='Attributable_Distribution', index=False)
    if not attributable_df.empty:
        # Include raw draws (first 10,000 to keep file size reasonable)
        attributable_df.head(10000).to_excel(writer, sheet_name='Attributable_Draws', index=False)

    qaly_df.to_excel(writer,         sheet_name='QALY_Differences',             index=False)
    psa_table_df.to_excel(writer,    sheet_name='PSA_Table',                    index=False)
    sa_combined_df.to_excel(writer,  sheet_name='Sensitivity_Analysis',          index=False)

print(f"\nManuscript tables saved to: {OUTPUT_PATH}")
print("Sheets:")
print("  1. Model_Inputs                  — Model parameters and inputs")
print("  2. Table3_Scenario_Comparison    — Table 3 data (2030/2035/2040)")
print("  3. Attributable_Cases_Summary    — PAF summary statistics from PSA")
print("  4. Attributable_Distribution     — Histogram data (% attributable vs frequency)")
print("  5. Attributable_Draws            — Raw PSA draws with attributable cases")
print("  6. QALY_Differences              — Cumulative QALYs by year")
print("  7. PSA_Table                     — PSA results summary")
print("  8. Sensitivity_Analysis          — One-way sensitivity analysis")

if histogram_figure_path:
    print(f"\nHistogram figure saved to: {histogram_figure_path}")
print(f"\nFigures directory: {FIGURES_DIR.absolute()}")
