"""
Master Script - One-Way Sensitivity Analyses (Baseline + Growth Scenario)

Runs both one-way sensitivity analyses sequentially:
1. Baseline scenario (50% stable PD prevalence)
2. Growth scenario (50% → 61.25% PD prevalence)

Both analyses use:
- Full population (10,787,479 agents)
- Deterministic approach (1 run per HR value)
- HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)

Total runtime: ~6 hours (3 hours per scenario)

Output files:
- pd_sensitivity_analysis.xlsx (baseline)
- pd_sensitivity_analysis_growth.xlsx (growth)
"""

import sys
import io
from pathlib import Path
from datetime import datetime
import traceback
import copy

# Set UTF-8 encoding for output (Windows compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'seed': 42,
    'output_dir': Path('sensitivity_analysis_results'),
    'log_file': 'sensitivity_analysis_log.txt',
}

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Simple logger that writes to both console and file"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear log file
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"Sensitivity Analysis Log - Started {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        """Log a message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"

        print(formatted)

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")

    def section(self, title: str):
        """Log a section header"""
        separator = "=" * 80
        self.log("")
        self.log(separator)
        self.log(title)
        self.log(separator)
        self.log("")


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def enable_growth_scenario(config: dict) -> dict:
    """Enable time-varying periodontal disease prevalence in config"""
    cfg = copy.deepcopy(config)

    # Enable prevalence schedule for periodontal disease
    pd_cfg = cfg['risk_factors']['periodontal_disease']
    if 'prevalence_schedule' in pd_cfg:
        pd_cfg['prevalence_schedule']['use'] = True

    return cfg


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_baseline_sensitivity(logger: Logger, config: dict) -> bool:
    """One-way SA for baseline scenario — loads baseline from pkl, runs only HR=1.07/1.38."""
    logger.section("ANALYSIS 1: Baseline Scenario (50% Stable PD Prevalence)")

    excel_file = Path('pd_sensitivity_analysis.xlsx')
    if excel_file.exists():
        logger.log(f"Output already exists — skipping: {excel_file}")
        return True

    logger.log("Approach: Load baseline from pkl; run HR=1.07 and HR=1.38 only")
    logger.log("HR values: 1.07 (low), 1.21 (baseline loaded), 1.38 (high)")
    logger.log("")

    try:
        import pandas as pd
        from IBM_PD_AD_v3 import general_config, run_model, extract_psa_metrics, load_results_compressed

        seed = config['seed']
        original_pop = general_config.get('population', 10787479)
        hr_low  = 1.07
        hr_high = 1.38

        def set_pd_hr(cfg: dict, onset_hr: float) -> dict:
            c = copy.deepcopy(cfg)
            hr_map = c['risk_factors']['periodontal_disease'].setdefault('hazard_ratios', {})
            hr_map.setdefault('onset', {})
            if isinstance(hr_map['onset'], dict):
                hr_map['onset']['female'] = onset_hr
                hr_map['onset']['male'] = onset_hr
            else:
                hr_map['onset'] = {'female': onset_hr, 'male': onset_hr, 'all': onset_hr}
            return c

        def make_metrics(result: dict, param_name: str, value_type: str, prevalence: float) -> dict:
            m = extract_psa_metrics(result)
            m['parameter']  = param_name
            m['value_type'] = value_type
            m['replicate']  = 0
            m['prevalence'] = prevalence
            return m

        # Baseline (HR=1.21, 50% stable) — load from run_baseline_and_scenarios.py
        baseline_pkl = Path("results") / "results_pd_baseline.pkl.gz"
        if baseline_pkl.exists():
            logger.log(f"Loading baseline from {baseline_pkl}...")
            baseline_result = load_results_compressed(baseline_pkl)
        else:
            logger.log("results_pd_baseline.pkl.gz not found — running baseline model...")
            stable_cfg = copy.deepcopy(general_config)
            stable_cfg['risk_factors']['periodontal_disease']['prevalence_schedule']['use'] = False
            baseline_result = run_model(stable_cfg, seed=seed, return_agents=False)

        baseline_metrics = make_metrics(baseline_result, 'baseline', 'baseline', 0.50)
        baseline_qalys   = baseline_metrics['total_qalys_combined']
        logger.log(f"  Baseline QALYs (HR=1.21): {baseline_qalys:,.0f}")

        # Build stable base config (prevalence schedule off) for HR variants
        stable_base = copy.deepcopy(general_config)
        stable_base['risk_factors']['periodontal_disease']['prevalence_schedule']['use'] = False

        # Low HR (1.07, 50% stable) — new run
        logger.log(f"\nRunning low HR scenario (HR={hr_low})...")
        low_cfg     = set_pd_hr(stable_base, hr_low)
        low_result  = run_model(low_cfg, seed=seed, return_agents=False)
        low_metrics = make_metrics(low_result, 'onset_hr', 'low', 0.50)
        low_qalys   = low_metrics['total_qalys_combined']
        logger.log(f"  Low HR QALYs (HR={hr_low}): {low_qalys:,.0f}  (Delta={low_qalys - baseline_qalys:+,.0f})")

        # High HR (1.38, 50% stable) — new run
        logger.log(f"\nRunning high HR scenario (HR={hr_high})...")
        high_cfg     = set_pd_hr(stable_base, hr_high)
        high_result  = run_model(high_cfg, seed=seed, return_agents=False)
        high_metrics = make_metrics(high_result, 'onset_hr', 'high', 0.50)
        high_qalys   = high_metrics['total_qalys_combined']
        logger.log(f"  High HR QALYs (HR={hr_high}): {high_qalys:,.0f}  (Delta={high_qalys - baseline_qalys:+,.0f})")

        df = pd.DataFrame([baseline_metrics, low_metrics, high_metrics])
        excel_file = Path('pd_sensitivity_analysis.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            metadata = {
                'Parameter': ['Analysis Type', 'Scenario', 'HR Low', 'HR Baseline', 'HR High',
                               'Population', 'Random Seed', 'Date Generated'],
                'Value':     ['One-Way Sensitivity Analysis (Deterministic)',
                               'Baseline (50% stable PD prevalence)',
                               hr_low, 1.21, hr_high,
                               f'{original_pop:,}', seed,
                               datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

        logger.log(f"\n✓ Baseline sensitivity analysis complete!")
        logger.log(f"✓ Results saved to: {excel_file}")
        logger.log("")
        return True

    except Exception as e:
        logger.log(f"✗ Baseline analysis FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def run_growth_sensitivity(logger: Logger, config: dict) -> bool:
    """One-way SA for growth scenario — loads all three HR variants from pre-computed pkls."""
    logger.section("ANALYSIS 2: Growth Scenario (50% → 61.25% PD Prevalence)")

    excel_file = Path('pd_sensitivity_analysis_growth.xlsx')
    if excel_file.exists():
        logger.log(f"Output already exists — skipping: {excel_file}")
        return True

    logger.log("Approach: Load pre-computed full-population runs from run_baseline_and_scenarios.py")
    logger.log("HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)")
    logger.log("")

    try:
        import pandas as pd
        from IBM_PD_AD_v3 import general_config, run_model, extract_psa_metrics, load_results_compressed

        seed = config['seed']
        original_pop = general_config.get('population', 10787479)
        hr_low  = 1.07
        hr_high = 1.38

        PKL_BASELINE = Path("results") / "results_pd_growth.pkl.gz"
        PKL_LOW      = Path("results") / "results_pd_growth_hr_low.pkl.gz"
        PKL_HIGH     = Path("results") / "results_pd_growth_hr_high.pkl.gz"

        def set_pd_hr(cfg: dict, onset_hr: float) -> dict:
            c = copy.deepcopy(cfg)
            hr_map = c['risk_factors']['periodontal_disease'].setdefault('hazard_ratios', {})
            hr_map.setdefault('onset', {})
            if isinstance(hr_map['onset'], dict):
                hr_map['onset']['female'] = onset_hr
                hr_map['onset']['male'] = onset_hr
            else:
                hr_map['onset'] = {'female': onset_hr, 'male': onset_hr, 'all': onset_hr}
            return c

        base_growth_cfg = enable_growth_scenario(general_config)

        def load_or_run(pkl_path: Path, run_cfg: dict, label: str) -> dict:
            if pkl_path.exists():
                logger.log(f"  Loading {label} from {pkl_path}...")
                return load_results_compressed(pkl_path)
            logger.log(f"  {pkl_path} not found — running {label} model...")
            return run_model(run_cfg, seed=seed, return_agents=False)

        baseline_result = load_or_run(PKL_BASELINE, base_growth_cfg,                    "baseline growth (HR=1.21)")
        low_result      = load_or_run(PKL_LOW,      set_pd_hr(base_growth_cfg, hr_low),  f"growth HR={hr_low}")
        high_result     = load_or_run(PKL_HIGH,     set_pd_hr(base_growth_cfg, hr_high), f"growth HR={hr_high}")

        def make_metrics(result: dict, param_name: str, value_type: str) -> dict:
            m = extract_psa_metrics(result)
            m['parameter']  = param_name
            m['value_type'] = value_type
            m['replicate']  = 0
            m['scenario']   = 'growth'
            return m

        baseline_metrics = make_metrics(baseline_result, 'baseline', 'baseline')
        low_metrics      = make_metrics(low_result,      'onset_hr',  'low')
        high_metrics     = make_metrics(high_result,     'onset_hr',  'high')

        baseline_qalys = baseline_metrics['total_qalys_combined']
        low_qalys      = low_metrics['total_qalys_combined']
        high_qalys     = high_metrics['total_qalys_combined']
        logger.log(f"  Baseline QALYs (HR=1.21): {baseline_qalys:,.0f}")
        logger.log(f"  Low HR QALYs  (HR={hr_low}):  {low_qalys:,.0f}  (Delta={low_qalys - baseline_qalys:+,.0f})")
        logger.log(f"  High HR QALYs (HR={hr_high}): {high_qalys:,.0f}  (Delta={high_qalys - baseline_qalys:+,.0f})")

        df = pd.DataFrame([baseline_metrics, low_metrics, high_metrics])
        excel_file = Path('pd_sensitivity_analysis_growth.xlsx')

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            metadata = {
                'Parameter': ['Analysis Type', 'Scenario', 'Parameter Varied',
                               'HR Low', 'HR Baseline', 'HR High',
                               'Approach', 'Population', 'Prevalence',
                               'Random Seed', 'Date Generated'],
                'Value':     ['One-Way Sensitivity Analysis (Deterministic)',
                               'Growth Scenario (Elamin & Anash 2023)',
                               'PD Onset Hazard Ratio',
                               hr_low, 1.21, hr_high,
                               'Full population, loaded from pre-computed pkl files',
                               f'{original_pop:,}', '50% (2023) → 61.25% (2040)',
                               seed, datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

        logger.log(f"\n✓ Growth sensitivity analysis complete!")
        logger.log(f"✓ Results saved to: {excel_file}")
        logger.log("")
        return True

    except Exception as e:
        logger.log(f"✗ Growth analysis FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run both sensitivity analyses sequentially"""

    # Initialize
    start_time = datetime.now()
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / CONFIG['log_file']
    logger = Logger(log_path)

    logger.section("ONE-WAY SENSITIVITY ANALYSES - START")
    logger.log("This script runs both one-way sensitivity analyses:")
    logger.log("  1. Baseline scenario (50% stable PD prevalence)")
    logger.log("  2. Growth scenario (50% → 61.25% PD prevalence)")
    logger.log("")
    logger.log("Analysis approach:")
    logger.log("  - Full population (10,787,479 agents)")
    logger.log("  - Deterministic (1 run per HR value)")
    logger.log("  - HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)")
    logger.log("")
    logger.log(f"Configuration: {CONFIG}")
    logger.log(f"Start time: {start_time}")
    logger.log(f"Log file: {log_path.absolute()}")
    logger.log(f"Estimated total time: ~6 hours")

    # Track results
    results = {
        'Baseline Sensitivity Analysis': None,
        'Growth Sensitivity Analysis': None,
    }

    # Execute analyses
    results['Baseline Sensitivity Analysis'] = run_baseline_sensitivity(logger, CONFIG)
    results['Growth Sensitivity Analysis'] = run_growth_sensitivity(logger, CONFIG)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.section("ONE-WAY SENSITIVITY ANALYSES - COMPLETE")
    logger.log(f"End time: {end_time}")
    logger.log(f"Total duration: {duration}")
    logger.log("")
    logger.log("SUMMARY:")
    logger.log("-" * 80)

    for analysis_name, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        logger.log(f"  {analysis_name}: {status}")

    logger.log("-" * 80)
    logger.log("")
    logger.log("OUTPUT FILES:")
    logger.log("  Baseline (50% stable):")
    logger.log("    - pd_sensitivity_analysis.xlsx")
    logger.log("")
    logger.log("  Growth (50% → 61.25%):")
    logger.log("    - pd_sensitivity_analysis_growth.xlsx")
    logger.log("")
    logger.log(f"Full log saved to: {log_path.absolute()}")

    # Exit with error code if any analysis failed
    if any(result is False for result in results.values()):
        logger.log("\nWARNING: Some analyses failed. Check the log for details.", level="ERROR")
        return 1

    logger.log("\n✓ ALL SENSITIVITY ANALYSES COMPLETED SUCCESSFULLY!")
    logger.log("\nNEXT STEPS:")
    logger.log("  1. Compare sensitivity results between scenarios")
    logger.log("  2. Create tornado diagrams from Excel data")
    logger.log("  3. Report upper/lower bounds in manuscript")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
