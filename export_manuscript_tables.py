"""
Export manuscript tables from completed model results.

Requires two pkl.gz result files (baseline and growth) already produced by run_model.
Reads those files and writes Manuscript_Tables.xlsx with six sheets:

  1. Model_Inputs                 — comprehensive model parameters: population, risk factors,
                                    hazard ratios, utilities, costs, progression rates
  2. Table3_Scenario_Comparison   — dementia cases, annual formal and societal costs
                                    at 2030, 2035, 2040 for baseline vs growth
  3. Risk_Factor_Enrichment       — general population vs dementia population prevalence
                                    and relative enrichment for all risk factors at 2024
  4. QALY_Differences             — cumulative patient and caregiver QALY differences
                                    (growth minus baseline) by year 2024-2040
  5. PSA_Table                    — merged from PSA_Results_Growth.xlsx (PSA_Table sheet)
  6. Sensitivity_Analysis         — merged from Sensitivity_Analysis.xlsx (both scenarios)

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
SA_EXCEL_PATH         = Path('sensitivity_analysis_results') / 'Sensitivity_Analysis.xlsx'
OUTPUT_PATH           = Path('results') / 'Manuscript_Tables.xlsx'
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

# ── Risk factor enrichment (baseline year 2024) ────────────────────────────────
# For each risk factor: prevalence in general population, prevalence in dementia
# population, relative enrichment (%)

b_2024 = get_year_row(baseline_df, 2024)

enrichment_rows = []
for rf_key, rf_label in RISK_FACTOR_LABELS.items():
    alive_col    = f'risk_prev_alive_{rf_key}'
    dementia_col = f'risk_prev_dementia_{rf_key}'
    gen_prev     = b_2024.get(alive_col,    np.nan)
    dem_prev     = b_2024.get(dementia_col, np.nan)
    if np.isnan(gen_prev) or np.isnan(dem_prev) or gen_prev == 0:
        enrichment = np.nan
    else:
        enrichment = (dem_prev - gen_prev) / gen_prev * 100
    enrichment_rows.append({
        'Risk factor':                      rf_label,
        'General population prevalence (%)': round(gen_prev * 100, 1) if not np.isnan(gen_prev) else None,
        'Dementia population prevalence (%)': round(dem_prev * 100, 1) if not np.isnan(dem_prev) else None,
        'Relative enrichment (%)':           round(enrichment, 1) if not np.isnan(enrichment) else None,
    })

# Sort by absolute enrichment descending
enrichment_df = pd.DataFrame(enrichment_rows)
enrichment_df = enrichment_df.sort_values(
    'Relative enrichment (%)', ascending=False, na_position='last'
).reset_index(drop=True)

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

if SA_EXCEL_PATH.exists():
    sa_baseline_df = pd.read_excel(SA_EXCEL_PATH, sheet_name='Baseline_Scenario')
    sa_growth_df   = pd.read_excel(SA_EXCEL_PATH, sheet_name='Growth_Scenario')
    sa_baseline_df.insert(0, 'Scenario', 'Baseline (50% stable)')
    sa_growth_df.insert(0, 'Scenario', 'Growth (50%→61.25%)')
    sa_combined_df = pd.concat([sa_baseline_df, sa_growth_df], ignore_index=True)
    print(f"Sensitivity analysis loaded: {len(sa_combined_df)} rows")
else:
    print(f"WARNING: SA Excel not found at {SA_EXCEL_PATH}. Sensitivity_Analysis sheet will be empty.")
    sa_combined_df = pd.DataFrame()

# ── Write workbook ─────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    model_inputs_df.to_excel(writer, sheet_name='Model_Inputs',                index=False)
    table3_df.to_excel(writer,       sheet_name='Table3_Scenario_Comparison',  index=False)
    enrichment_df.to_excel(writer,   sheet_name='Risk_Factor_Enrichment',       index=False)
    qaly_df.to_excel(writer,         sheet_name='QALY_Differences',             index=False)
    psa_table_df.to_excel(writer,    sheet_name='PSA_Table',                    index=False)
    sa_combined_df.to_excel(writer,  sheet_name='Sensitivity_Analysis',          index=False)

print(f"\nManuscript tables saved to: {OUTPUT_PATH}")
print("Sheets:")
print("  1. Model_Inputs                — Model parameters and inputs")
print("  2. Table3_Scenario_Comparison  — Table 3 data (2030/2035/2040)")
print("  3. Risk_Factor_Enrichment      — Figure 2 data (enrichment)")
print("  4. QALY_Differences            — Figure 4 data (cumulative QALYs)")
print("  5. PSA_Table                   — Table 4 (PSA results)")
print("  6. Sensitivity_Analysis        — Table 5 (one-way SA)")
