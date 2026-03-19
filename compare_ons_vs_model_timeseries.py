"""
Compare Model Age Distribution vs ONS Projections Over Time

This script runs the model from 2023-2040 and creates time series plots
comparing the model's age distribution dynamics against ONS population projections
for each 5-year age band.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from IBM_PD_AD_v3 import run_model, general_config
from ons_projection_data import get_ons_projected_distribution


def extract_model_age_distributions_all_years(results):
    """
    Extract age distributions from model results for all years.

    Returns DataFrame with columns: year, age_band, model_proportion
    """
    df = results['incidence_by_year_sex_df']

    # Filter to get all years and both sexes
    df_filtered = df[df['sex'].isin(['male', 'female'])].copy()

    records = []
    for year in sorted(df_filtered['calendar_year'].unique()):
        year_df = df_filtered[df_filtered['calendar_year'] == year]

        # Group by age band (sum across sexes)
        grouped = (
            year_df.groupby(['age_lower', 'age_upper'], as_index=False)['population_alive_in_band']
            .sum()
            .rename(columns={'population_alive_in_band': 'population_count'})
        )

        # Calculate proportions
        total_pop = grouped['population_count'].sum()
        if total_pop > 0:
            grouped['model_proportion'] = grouped['population_count'] / total_pop

            # Add year and age band label
            for _, row in grouped.iterrows():
                age_lower = int(row['age_lower'])
                age_upper = row['age_upper']
                if pd.notna(age_upper) and age_upper != 100:
                    age_band = f"{age_lower}-{int(age_upper)}"
                else:
                    age_band = f"{age_lower}+"

                records.append({
                    'year': year,
                    'age_lower': age_lower,
                    'age_upper': 100 if pd.isna(age_upper) or age_upper == 100 else int(age_upper),
                    'age_band': age_band,
                    'model_proportion': row['model_proportion']
                })

    return pd.DataFrame(records)


def extract_ons_projections_all_years():
    """
    Extract ONS projections for all available years.

    Returns DataFrame with columns: year, age_band, ons_proportion
    """
    # All years from 2023 to 2040
    years = list(range(2023, 2041))

    records = []
    for year in years:
        try:
            distribution = get_ons_projected_distribution(year)
            for (age_lower, age_upper), proportion in distribution.items():
                age_band = f"{age_lower}-{age_upper if age_upper != 100 else '+'}"
                records.append({
                    'year': year,
                    'age_lower': age_lower,
                    'age_upper': age_upper,
                    'age_band': age_band,
                    'ons_proportion': proportion
                })
        except ValueError:
            # Year not available, skip
            pass

    return pd.DataFrame(records)


def plot_comparison_timeseries(model_df, ons_df, save_dir="plots", scenario_name=""):
    """
    Create time series plots comparing ONS projections vs model output for each age band.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get unique age bands
    age_bands = sorted(model_df['age_band'].unique())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, age_band in enumerate(age_bands):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Filter data for this age band
        model_band = model_df[model_df['age_band'] == age_band].sort_values('year')
        ons_band = ons_df[ons_df['age_band'] == age_band].sort_values('year')

        # Plot ONS projections
        ax.plot(ons_band['year'], ons_band['ons_proportion'] * 100,
                'o-', linewidth=2.5, markersize=8, label='ONS Projection',
                color='#2E86AB', alpha=0.8)

        # Plot model output
        ax.plot(model_band['year'], model_band['model_proportion'] * 100,
                's--', linewidth=2, markersize=6, label='Model Output',
                color='#A23B72', alpha=0.8)

        # Calculate RMSE
        merged = model_band.merge(ons_band, on='year', how='inner')
        if not merged.empty:
            rmse = np.sqrt(np.mean((merged['model_proportion'] - merged['ons_proportion'])**2)) * 100
            ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}%',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Proportion of 65+ Population (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Age Band: {age_band}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(2023, 2040)

    # Hide unused subplots
    for idx in range(len(age_bands), len(axes)):
        axes[idx].axis('off')

    title_suffix = f" ({scenario_name})" if scenario_name else ""
    plt.suptitle(f'Age Distribution Validation: ONS Projections vs Model Output{title_suffix}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save plot
    filename = f"ons_vs_model_timeseries{'_' + scenario_name.replace(' ', '_').lower() if scenario_name else ''}.png"
    plot_path = save_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTime series plot saved: {plot_path}")

    return plot_path


def print_summary_statistics(model_df, ons_df):
    """Print summary statistics of model fit to ONS projections."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: MODEL FIT TO ONS PROJECTIONS")
    print("="*80)

    # Merge data
    merged = model_df.merge(ons_df, on=['year', 'age_band'], how='inner')

    if merged.empty:
        print("No overlapping data to compare")
        return

    # Calculate errors
    merged['abs_error'] = (merged['model_proportion'] - merged['ons_proportion']) * 100
    merged['rel_error_pct'] = ((merged['model_proportion'] - merged['ons_proportion']) /
                               merged['ons_proportion']) * 100

    # Overall statistics
    rmse_overall = np.sqrt(np.mean(merged['abs_error']**2))
    mae_overall = np.mean(np.abs(merged['abs_error']))

    print(f"\nOverall Fit Metrics:")
    print(f"  RMSE: {rmse_overall:.3f} percentage points")
    print(f"  MAE:  {mae_overall:.3f} percentage points")

    # By age band
    print(f"\nFit Metrics by Age Band:")
    print(f"{'Age Band':<12} {'RMSE (pp)':<12} {'MAE (pp)':<12} {'Mean Rel Error %':<18}")
    print("-"*60)

    for age_band in sorted(merged['age_band'].unique()):
        band_data = merged[merged['age_band'] == age_band]
        rmse = np.sqrt(np.mean(band_data['abs_error']**2))
        mae = np.mean(np.abs(band_data['abs_error']))
        mean_rel_error = np.mean(band_data['rel_error_pct'])

        print(f"{age_band:<12} {rmse:<12.3f} {mae:<12.3f} {mean_rel_error:<18.2f}")

    print("="*80)


def run_comparison(config, seed=42, save_dir="plots", scenario_name=""):
    """
    Run full comparison of model vs ONS projections.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"RUNNING MODEL VS ONS COMPARISON{' - ' + scenario_name if scenario_name else ''}")
    print("="*80)

    # Set up config to run to 2040
    comparison_config = copy.deepcopy(config)
    base_year = int(comparison_config.get('base_year', 2023))
    timesteps_needed = 2040 - base_year
    comparison_config['number_of_timesteps'] = timesteps_needed

    print(f"\nRunning model: {base_year} to 2040 ({timesteps_needed} timesteps)")
    print(f"Population: {comparison_config.get('population', 0):,}")

    # Run model
    print("\nRunning model...")
    results = run_model(comparison_config, seed=seed, return_agents=False)
    print("Model run complete!")

    # Extract model age distributions
    print("\nExtracting model age distributions...")
    model_df = extract_model_age_distributions_all_years(results)

    # Extract ONS projections
    print("Extracting ONS projections...")
    ons_df = extract_ons_projections_all_years()

    # Print summary statistics
    print_summary_statistics(model_df, ons_df)

    # Create plots
    print("\nGenerating comparison plots...")
    plot_path = plot_comparison_timeseries(model_df, ons_df, save_dir, scenario_name)

    # Save detailed comparison table
    merged = model_df.merge(ons_df, on=['year', 'age_band'], how='inner')
    merged['abs_error_pp'] = (merged['model_proportion'] - merged['ons_proportion']) * 100
    merged['rel_error_pct'] = ((merged['model_proportion'] - merged['ons_proportion']) /
                               merged['ons_proportion']) * 100

    table_filename = f"ons_vs_model_detailed{'_' + scenario_name.replace(' ', '_').lower() if scenario_name else ''}.csv"
    table_path = save_dir / table_filename
    merged[['year', 'age_band', 'ons_proportion', 'model_proportion',
            'abs_error_pp', 'rel_error_pct']].to_csv(table_path, index=False, float_format='%.6f')
    print(f"Detailed comparison table saved: {table_path}")

    print(f"\n{'='*80}")
    print("[SUCCESS] Comparison complete!")
    print(f"{'='*80}\n")

    return {
        'model_df': model_df,
        'ons_df': ons_df,
        'merged': merged,
        'plot_path': plot_path,
        'table_path': table_path
    }


if __name__ == "__main__":
    # Run comparison with reduced population for faster execution
    # Use 100k instead of 10M+ for age distribution validation

    # Define periodontal prevalences to test
    prevalences = [0.25, 0.50, 0.75]

    # Store all results
    all_results = {}

    for prevalence in prevalences:
        print(f"\n{'='*80}")
        print(f"STARTING ANALYSIS FOR PERIODONTAL PREVALENCE: {prevalence*100:.0f}%")
        print(f"{'='*80}\n")

        # Create config for this prevalence
        comparison_config = copy.deepcopy(general_config)
        comparison_config['population'] = 100000  # Reduced from 10,787,479
        comparison_config['periodontal_prevalence'] = prevalence

        # Also reduce entrants proportionally (from 700k to ~6.5k)
        if 'open_population' in comparison_config and comparison_config['open_population'].get('use'):
            comparison_config['open_population']['entrants_per_year'] = 6500

        # Create scenario name
        scenario_name = f"PD_{int(prevalence*100)}pct"
        save_dir = f"full_analysis_results/ons_comparison/{scenario_name}"

        # Run comparison
        results = run_comparison(
            comparison_config,
            seed=42,
            save_dir=save_dir,
            scenario_name=scenario_name
        )

        # Store results
        all_results[f"{prevalence*100:.0f}%"] = results

        print(f"\n{'='*80}")
        print(f"COMPLETED ANALYSIS FOR {prevalence*100:.0f}% PREVALENCE")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")
    print("\nSummary of results:")
    for prev_label, result in all_results.items():
        print(f"  {prev_label}: {result['plot_path']}")
