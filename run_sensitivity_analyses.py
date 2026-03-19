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
    """Run one-way sensitivity analysis for baseline scenario (50% stable)"""
    logger.section("ANALYSIS 1: Baseline Scenario (50% Stable PD Prevalence)")
    logger.log("Approach: Full population, deterministic")
    logger.log("HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)")
    logger.log("Estimated time: ~3 hours")
    logger.log("")

    try:
        from IBM_PD_AD_v3 import general_config
        from pd_sensitivity_analysis import run_pd_sensitivity_analysis

        # Run baseline sensitivity analysis
        results = run_pd_sensitivity_analysis(
            general_config,
            population_fraction=1.0,    # Full population
            n_replicates=1,             # Single deterministic run per HR value
            combine_sexes=True,
            seed=config['seed'],
            paired_seeds=True,          # Use same seed across HR values
            prevalence_values=[0.50],   # Baseline: 50% stable prevalence
            n_jobs=1,                   # Sequential runs
            auto_export=True,           # Auto-export to Excel
            export_path=Path('pd_sensitivity_analysis.xlsx'),
            show_progress=True
        )

        logger.log("✓ Baseline sensitivity analysis complete!")
        logger.log(f"✓ Results saved to: pd_sensitivity_analysis.xlsx")
        logger.log(f"  - {len(results)} scenarios completed")
        logger.log("")
        return True

    except Exception as e:
        logger.log(f"✗ Baseline analysis FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def run_growth_sensitivity(logger: Logger, config: dict) -> bool:
    """Run one-way sensitivity analysis for growth scenario (50% → 61.25%)"""
    logger.section("ANALYSIS 2: Growth Scenario (50% → 61.25% PD Prevalence)")
    logger.log("Approach: Full population, deterministic")
    logger.log("HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)")
    logger.log("Prevalence: 50% (2023) → 61.25% (2040)")
    logger.log("Based on: Elamin & Anash (2023) projections")
    logger.log("Estimated time: ~3 hours")
    logger.log("")

    try:
        from IBM_PD_AD_v3 import general_config, run_model, extract_psa_metrics
        import pandas as pd

        seed = config['seed']
        original_pop = general_config.get('population', 10787479)

        # HR bounds from 95% CI
        hr_low = 1.07
        hr_high = 1.38

        # Enable growth scenario
        working_config = enable_growth_scenario(general_config)

        logger.log(f"Population: {original_pop:,} agents")
        logger.log("")

        def set_pd_hr(cfg: dict, onset_hr: float) -> dict:
            """Set the PD onset hazard ratio"""
            config_copy = copy.deepcopy(cfg)
            pd_meta = config_copy['risk_factors']['periodontal_disease']
            hr_map = pd_meta.setdefault('hazard_ratios', {})
            hr_map.setdefault('onset', {})
            if isinstance(hr_map['onset'], dict):
                hr_map['onset']['female'] = onset_hr
                hr_map['onset']['male'] = onset_hr
            else:
                hr_map['onset'] = {'female': onset_hr, 'male': onset_hr, 'all': onset_hr}
            return config_copy

        def run_scenario(cfg: dict, param_name: str, value_type: str) -> dict:
            """Run a single deterministic scenario"""
            result = run_model(cfg, seed=seed)
            metrics = extract_psa_metrics(result)
            metrics['parameter'] = param_name
            metrics['value_type'] = value_type
            metrics['replicate'] = 0
            metrics['scenario'] = 'growth'
            return metrics

        results_list = []

        # Baseline (HR = 1.21)
        logger.log("Running baseline scenario (HR=1.21)...")
        baseline_metrics = run_scenario(working_config, 'baseline', 'baseline')
        results_list.append(baseline_metrics)
        baseline_qalys = baseline_metrics['total_qalys_combined']
        logger.log(f"  ✓ Baseline QALYs: {baseline_qalys:,.0f}")
        logger.log("")

        # Low HR
        logger.log(f"Running low HR scenario (HR={hr_low})...")
        low_config = set_pd_hr(working_config, hr_low)
        low_metrics = run_scenario(low_config, 'onset_hr', 'low')
        results_list.append(low_metrics)
        low_qalys = low_metrics['total_qalys_combined']
        logger.log(f"  ✓ Low HR QALYs: {low_qalys:,.0f} (Delta={low_qalys - baseline_qalys:+,.0f})")
        logger.log("")

        # High HR
        logger.log(f"Running high HR scenario (HR={hr_high})...")
        high_config = set_pd_hr(working_config, hr_high)
        high_metrics = run_scenario(high_config, 'onset_hr', 'high')
        results_list.append(high_metrics)
        high_qalys = high_metrics['total_qalys_combined']
        logger.log(f"  ✓ High HR QALYs: {high_qalys:,.0f} (Delta={high_qalys - baseline_qalys:+,.0f})")
        logger.log("")

        # Compile results
        df = pd.DataFrame(results_list)
        logger.log("Results compiled (full population, no scaling required)")
        logger.log("")

        # Export to Excel
        excel_file = Path('pd_sensitivity_analysis_growth.xlsx')
        logger.log(f"Exporting results to {excel_file}...")

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

            # Add metadata sheet
            metadata = {
                'Parameter': [
                    'Analysis Type',
                    'Scenario',
                    'Parameter Varied',
                    'HR Low',
                    'HR Baseline',
                    'HR High',
                    'Approach',
                    'Population',
                    'Prevalence',
                    'Random Seed',
                    'Date Generated'
                ],
                'Value': [
                    'One-Way Sensitivity Analysis (Deterministic)',
                    'Growth Scenario (Elamin & Anash 2023)',
                    'PD Onset Hazard Ratio',
                    hr_low,
                    1.21,
                    hr_high,
                    'Full population, single deterministic run per HR value',
                    f'{original_pop:,}',
                    '50% (2023) → 61.25% (2040)',
                    seed,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        logger.log(f"✓ Growth sensitivity analysis complete!")
        logger.log(f"✓ Results saved to: {excel_file}")
        logger.log(f"  Sheets: Results, Metadata")
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
