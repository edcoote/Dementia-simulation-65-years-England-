"""
Counterfactual analysis: incidence/prevalence not attributable to preventable risk factors.

This script leaves `IBM_PD_AD_v3.py` untouched. It:
1) Runs the baseline model.
2) Runs a counterfactual with preventable risks removed (prevalence=0, HR=1).
3) Reports how much incidence and end-of-horizon prevalence remain without those risks.

Output:
  results/non_preventable_risk_analysis.xlsx
"""

import copy
from pathlib import Path
import pandas as pd

from IBM_PD_AD_v3 import (
    general_config,
    run_model,
    extract_psa_metrics,
)


# Treat all modifiable risks as preventable; leave genetic APOE e4 untouched.
PREVENTABLE_RISKS = [
    'periodontal_disease',
    'socioeconomic_disadvantage',
    'low_education',
    'hearing_difficulty',
    'hypertension',
    'obesity',
    'lifestyle',
    'excessive_alcohol_consumption',
    'smoking',
    'depression',
    'social_isolation',
    'diabetes',
    'air_pollution',
]

RESULTS_PATH = Path("results") / "non_preventable_risk_analysis.xlsx"
DEFAULT_SEED = 42


def _zero_prevalence(meta: dict) -> None:
    prev = meta.setdefault('prevalence', {})
    prev['female'] = 0.0
    prev['male'] = 0.0


def _neutralise_hazard_ratios(meta: dict) -> None:
    hr_map = meta.get('hazard_ratios') or meta.get('relative_risks')
    if not isinstance(hr_map, dict):
        return
    for transition, sex_map in hr_map.items():
        if not isinstance(sex_map, dict):
            continue
        for sex in ('all', 'female', 'male'):
            sex_map[sex] = 1.0


def build_counterfactual_config(cfg: dict) -> dict:
    """Return a deep-copied config with preventable risks removed."""
    cf = copy.deepcopy(cfg)
    risk_defs = cf.setdefault('risk_factors', {})
    for risk_name, meta in risk_defs.items():
        if risk_name not in PREVENTABLE_RISKS:
            continue
        if not isinstance(meta, dict):
            continue
        _zero_prevalence(meta)
        _neutralise_hazard_ratios(meta)
    return cf


def summarise(metrics_baseline: dict, metrics_cf: dict) -> pd.DataFrame:
    inc_base = metrics_baseline.get('incident_onsets_total', 0.0)
    inc_cf = metrics_cf.get('incident_onsets_total', 0.0)
    inc_prev = inc_base - inc_cf
    inc_prev_share = (inc_prev / inc_base) if inc_base else 0.0

    def _stage(metrics, key):
        return metrics.get(key, 0.0)

    prevalence_cols = ['stage_mild', 'stage_moderate', 'stage_severe']
    rows = [
        ['incident_onsets_total', inc_base, inc_cf, inc_prev, inc_prev_share],
    ]

    for col in prevalence_cols:
        base_val = _stage(metrics_baseline, col)
        cf_val = _stage(metrics_cf, col)
        preventable = base_val - cf_val
        share = (preventable / base_val) if base_val else 0.0
        rows.append([col, base_val, cf_val, preventable, share])

    return pd.DataFrame(
        rows,
        columns=[
            'metric',
            'baseline',
            'counterfactual_no_preventable_risks',
            'preventable_component',
            'preventable_share',
        ],
    )


def main():
    print("Running baseline model...")
    baseline_results = run_model(general_config, seed=DEFAULT_SEED)
    baseline_metrics = extract_psa_metrics(baseline_results)

    print("Building counterfactual with preventable risks removed...")
    cf_config = build_counterfactual_config(general_config)
    cf_results = run_model(cf_config, seed=DEFAULT_SEED)
    cf_metrics = extract_psa_metrics(cf_results)

    summary_df = summarise(baseline_metrics, cf_metrics)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(RESULTS_PATH) as writer:
        pd.DataFrame([baseline_metrics]).to_excel(writer, sheet_name="BaselineMetrics", index=False)
        pd.DataFrame([cf_metrics]).to_excel(writer, sheet_name="CounterfactualMetrics", index=False)
        summary_df.to_excel(writer, sheet_name="PreventableBreakdown", index=False)

    print("\nDone.")
    print(f"Saved results to {RESULTS_PATH}")
    print("\nKey results:")
    print(summary_df.to_string(index=False, formatters={'preventable_share': '{:.2%}'.format}))


if __name__ == "__main__":
    main()
