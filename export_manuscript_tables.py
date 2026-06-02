"""
Export manuscript tables from completed model results.

Requires two pkl.gz result files (baseline and growth) already produced by run_model.
Reads those files and writes Manuscript_Tables.xlsx with five sheets:

  1. Table3_Scenario_Comparison   — dementia cases, annual formal and societal costs
                                    at 2030, 2035, 2040 for baseline vs growth
  2. Risk_Factor_Enrichment       — general population vs dementia population prevalence
                                    and relative enrichment for all risk factors at 2024
  3. QALY_Differences             — cumulative patient and caregiver QALY differences
                                    (growth minus baseline) by year 2024-2040
  4. PSA_Table                    — merged from PSA_Results_Growth.xlsx (PSA_Table sheet)
  5. Sensitivity_Analysis         — merged from Sensitivity_Analysis.xlsx (both scenarios)

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

from IBM_PD_AD_v3 import load_results_compressed, summaries_to_dataframe

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
    table3_df.to_excel(writer,       sheet_name='Table3_Scenario_Comparison',  index=False)
    enrichment_df.to_excel(writer,   sheet_name='Risk_Factor_Enrichment',       index=False)
    qaly_df.to_excel(writer,         sheet_name='QALY_Differences',             index=False)
    psa_table_df.to_excel(writer,    sheet_name='PSA_Table',                    index=False)
    sa_combined_df.to_excel(writer,  sheet_name='Sensitivity_Analysis',          index=False)

print(f"\nManuscript tables saved to: {OUTPUT_PATH}")
print("Sheets:")
print("  1. Table3_Scenario_Comparison  — Table 3 data (2030/2035/2040)")
print("  2. Risk_Factor_Enrichment      — Figure 2 data (enrichment)")
print("  3. QALY_Differences            — Figure 4 data (cumulative QALYs)")
print("  4. PSA_Table                   — Table 4 (PSA results)")
print("  5. Sensitivity_Analysis        — Table 5 (one-way SA)")
