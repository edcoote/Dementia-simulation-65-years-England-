# Dementia progression model (hazard-based), using basline data from 2023

import os
import random
import math
import copy
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional: tqdm for progress bars (will gracefully degrade if not installed)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm not installed"""
        return iterable

# General configuration

REPORTING_AGE_BANDS: List[Tuple[int, Optional[int]]] = [
    (65, 79),
    (80, None),
]

# Hazard reporting age bands for incidence outputs (inclusive upper bounds)
INCIDENCE_AGE_BANDS: List[Tuple[int, Optional[int]]] = [
    (65, 69),
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 89),
    (90, None),
]


def assign_age_to_reporting_band(age: float,
                                 bands: Optional[List[Tuple[int, Optional[int]]]] = None
                                 ) -> Optional[Tuple[int, Optional[int]]]:
    """Return the first reporting age band that contains the provided age."""
    lookup = bands if bands is not None else REPORTING_AGE_BANDS
    for lower, upper in lookup:
        if upper is None:
            if age >= lower:
                return (lower, upper)
        elif lower <= age <= upper:
            return (lower, upper)
    return None


def age_band_key(band: Tuple[int, Optional[int]]) -> str:
    """Stable key for dictionary columns derived from an age band."""
    lower, upper = band
    upper_str = str(upper) if upper is not None else "plus"
    return f"{lower}_{upper_str}"


def age_band_label(band: Tuple[int, Optional[int]]) -> str:
    """Human-readable label for age band legends."""
    lower, upper = band
    if upper is None:
        return f"{lower}+"
    return f"{lower}-{upper}"


def age_band_midpoint(band: Tuple[int, Optional[int]]) -> Optional[float]:
    """Return numeric midpoint for a closed interval; None for open-ended bands."""
    lower, upper = band
    if upper is None:
        return None
    return (lower + upper) / 2.0


def generate_placeholder_qx_schedule(start_year: int = 2023,
                                     end_year: int = 2040) -> Dict[int, Dict[str, Dict[int, float]]]:
    """
    Create a deterministic placeholder qx grid for years [start_year, end_year], ages 65-100, by sex.
    This is meant to be replaced with real ONS qx; it simply imposes a gentle improvement trend.
    """
    schedule: Dict[int, Dict[str, Dict[int, float]]] = {}
    for year in range(start_year, end_year + 1):
        improvement = (1.0 - 0.005) ** max(0, year - start_year)  # 0.5% annual improvement
        year_entry: Dict[str, Dict[int, float]] = {}
        for sex in ('female', 'male'):
            base65 = 0.010 if sex == 'female' else 0.011
            age_step = 0.0045 if sex == 'female' else 0.00495
            sex_entry: Dict[int, float] = {}
            for age in range(65, 101):
                qx = (base65 + age_step * (age - 65)) * improvement
                qx = min(0.6, max(0.0, qx))
                sex_entry[age] = round(qx, 6)
            year_entry[sex] = sex_entry
        schedule[year] = year_entry
    return schedule


def smooth_series(values: List[float], window: int = 3) -> List[float]:
    """Simple centered moving average smoothing."""
    if window <= 1 or not values:
        return values
    series = pd.Series(values, dtype=float)
    smoothed = (
        series.rolling(window=window, center=True, min_periods=1)
        .mean()
        .tolist()
    )
    return smoothed


def _beta_params_from_mean_rel_sd(mean: float, rel_sd: float) -> Optional[Tuple[float, float]]:
    """Return (alpha, beta) for a beta distribution given mean and relative SD."""
    if not (0.0 < mean < 1.0) or rel_sd <= 0.0:
        return None
    variance = (rel_sd * mean) ** 2
    max_variance = mean * (1.0 - mean)
    if variance >= max_variance:
        variance = max_variance * 0.999
    if variance <= 0.0:
        return None
    common = (mean * (1.0 - mean) / variance) - 1.0
    if common <= 0.0:
        return None
    alpha = mean * common
    beta_param = (1.0 - mean) * common
    if alpha <= 0.0 or beta_param <= 0.0:
        return None
    return alpha, beta_param


def _sample_probability_value(value: Any,
                              rel_sd: float,
                              rng: np.random.Generator) -> Any:
    """Sample a beta-distributed value around the provided probability."""
    try:
        base = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(base) or base <= 0.0:
        return value
    epsilon = 1e-6
    clipped = min(max(base, epsilon), 1.0 - epsilon)
    params = _beta_params_from_mean_rel_sd(clipped, rel_sd)
    if params is None:
        return value
    alpha, beta_param = params
    return float(rng.beta(alpha, beta_param))


def _gamma_params_from_mean_rel_sd(mean: float, rel_sd: float) -> Optional[Tuple[float, float]]:
    """Return (shape, scale) for a gamma distribution given mean and relative SD."""
    if mean <= 0.0 or rel_sd <= 0.0:
        return None
    variance = (rel_sd * mean) ** 2
    if variance <= 0.0:
        return None
    shape = (mean ** 2) / variance
    scale = variance / mean
    if shape <= 0.0 or scale <= 0.0:
        return None
    return shape, scale


def _sample_gamma_value(value: Any,
                        rel_sd: float,
                        rng: np.random.Generator) -> Any:
    """Sample a gamma-distributed value around the provided positive mean."""
    try:
        base = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(base) or base <= 0.0:
        return value
    params = _gamma_params_from_mean_rel_sd(base, rel_sd)
    if params is None:
        return value
    shape, scale = params
    return float(rng.gamma(shape, scale))


def _lognormal_params_from_ci(point_estimate: float,
                              lower: float,
                              upper: float) -> Optional[Tuple[float, float]]:
    """Return (mu, sigma) for lognormal given point estimate and symmetric 95% CI."""
    if point_estimate <= 0.0 or lower <= 0.0 or upper <= 0.0:
        return None
    sigma = (math.log(upper) - math.log(lower)) / (2.0 * 1.96)
    if sigma <= 0.0:
        return None
    mu = math.log(point_estimate)
    return mu, sigma


def _sample_lognormal_from_ci(point_estimate: float,
                              lower: float,
                              upper: float,
                              rng: np.random.Generator) -> float:
    """Sample a lognormal value using the supplied CI; fall back to point estimate."""
    params = _lognormal_params_from_ci(point_estimate, lower, upper)
    if params is None:
        return point_estimate
    mu, sigma = params
    return float(rng.lognormal(mean=mu, sigma=sigma))


def _apply_beta_to_mapping(mapping: Dict[Any, Any],
                           rel_sd: float,
                           rng: np.random.Generator) -> None:
    """Recursively sample beta-distributed values for all numeric leaves."""
    for key, val in mapping.items():
        if isinstance(val, dict):
            _apply_beta_to_mapping(val, rel_sd, rng)
        else:
            mapping[key] = _sample_probability_value(val, rel_sd, rng)


def _sample_cost_structure(costs_cfg: Dict[str, Any],
                           rel_sd: float,
                           rng: np.random.Generator) -> None:
    """Apply gamma sampling to nested cost dictionaries."""
    for stage_meta in costs_cfg.values():
        if not isinstance(stage_meta, dict):
            continue
        for setting_meta in stage_meta.values():
            if not isinstance(setting_meta, dict):
                continue
            for payer, amount in setting_meta.items():
                setting_meta[payer] = _sample_gamma_value(amount, rel_sd, rng)


def _sample_risk_factor_prevalence(risk_defs: Dict[str, dict],
                                   rel_sd: float,
                                   rng: np.random.Generator) -> None:
    """Apply beta sampling to each risk-factor prevalence entry."""
    for meta in risk_defs.values():
        prevalence = meta.get('prevalence')
        if not isinstance(prevalence, dict):
            continue
        for sex_key, value in prevalence.items():
            prevalence[sex_key] = _sample_probability_value(value, rel_sd, rng)


def _sample_risk_factor_hazard_ratios(risk_defs: Dict[str, dict],
                                       rng: np.random.Generator) -> None:
    """Apply lognormal sampling to risk-factor hazard ratios when CI data is available."""
    for risk_name, meta in risk_defs.items():
        ci_lookup = RISK_FACTOR_HR_INTERVALS.get(risk_name, {})
        rr_def = meta.get('hazard_ratios')
        if not isinstance(rr_def, dict) or not ci_lookup:
            continue
        for transition, sex_map in rr_def.items():
            if not isinstance(sex_map, dict):
                continue
            ci_transition = ci_lookup.get(transition, {})
            if not ci_transition:
                continue
            for sex_key, value in sex_map.items():
                ci_tuple = ci_transition.get(sex_key) or ci_transition.get('all')
                if not ci_tuple:
                    continue
                point, lower, upper = ci_tuple
                sex_map[sex_key] = _sample_lognormal_from_ci(point, lower, upper, rng)


general_config = {
    'number_of_timesteps': 17,
    'population': 10787479,  # 65+ population only

    'time_step_years': 1,

     #(VERIFIED, ONS) # Anchor baseline stage mix ---
    'base_year': 2023,  # t=0 will be this year; t=1 => 2024, etc.

    "open_population": {
        "use": True,                 # set True to enable new entrants
        "entrants_per_year": 700000,  # 65+ entrants only (raised to support +3M population growth)
        "entrants_growth": {
            "use": True,
            # Annualized growth from +13.17% over 2019-2040[Data from ONS], applied from 2023 onward.
            "annual_rate": 0.0059088616,
            "reference_year": 2023,
        },
        "fixed_entry_age": 65,   # All new entrants enter at age 65 (realistic population inflow)
        "sex_distribution": None      # e.g. {"female":0.52,"male":0.48}
    },

    'store_individual_survival': False,    # disable huge per-person survival records by default
    'report_paf_to_terminal': False,      # suppress console logging of PAF summary
    'compute_paf_in_main': False,         # skip running the heavy PAF counterfactual unless needed
    'enable_constant_hazard_checks': False,  # disable relative-deviation diagnostics by default

    # Probabilistic sensitivity analysis configuration (sampling specs defined below)
    'psa': {
        'use': False,
        'two_level_only': False,      # if True, skip full-population PSA and only run two-level ANOVA PSA
        'iterations': 1000,
        'seed': 20231113,
        'n_jobs': None,             # Number of parallel jobs (None = use all CPU cores)
        'relative_sd_beta': 0.10,   # +/-10% relative SD for beta-distributed parameters
        'relative_sd_gamma': 0.10,  # +/-10% relative SD for gamma-distributed parameters
    },

    # Optional: override baseline (time step 0) summary metrics with known real-world data
    'initial_summary_overrides': {
        'deaths': 490000,
        'entrants': 700000,
        'incident_onsets': 90000,
    },

    'initial_stage_mix': {     #(VERIFIED, (Primary Care Dementia Data & Economic Impact Of Dementia CF Report))    # proportions by sex; each nested dict must sum to 1,
        'female': {
            'cognitively_normal': 0.90,
            'mild': 0.0452376,
            'moderate': 0.0339282,
            'severe': 0.0118656,
        },
        'male': {
            'cognitively_normal': 0.90,
            'mild': 0.0452376,
            'moderate': 0.0339282,
            'severe': 0.0118656,
        },
    },
    'initial_dementia_prevalence_by_age_band': {
        # Re-scaled again (~0.85x) to target ~0.85M baseline prevalent cases with current age mix
        (65, 79): {
            'female': 0.020655,
            'male': 0.019295,
        },
        (80, 100): {
            'female': 0.146285,
            'male': 0.10591,
        },
    },

    # (VERIFIED, ONS)  #  Sex mix used when initialising the cohort
    'sex_distribution': {
        'female': 0.54,
        'male': 0.46,
    },

    # Age-band distribution for total population (sex-specific weights, sums to 1.0 per sex)
    'population_age_band_weights_by_sex': {
        'female': {
            (65, 79): 0.523876,
            (80, 100): 0.587907,
        },
        'male': {
            (65, 79): 0.476124421,
            (80, 100): 0.412092542,
        },
    },
    # Dementia prevalence allocation by age band (used only for baseline prevalence draw)
    'initial_age_band_weights': {
        (65, 69): 0.12,
        (70, 74): 0.11,
        (75, 79): 0.23,
        (80, 84): 0.15,
        (85, 100): 0.39,
    },

    # #(VERIFED, NHS England)  #   Baseline annual probability of onset if no duration is provided for normal->mild
    'base_onset_probability': 0.0025,

    ## (VERIFIED, Tariot et al(2024)) # Mean durations (years) => baseline hazards = 1/duration under exponential assumption
    'stage_transition_durations': {
        # 'normal_to_mild': 6,
        'mild_to_moderate': 2.2,
        'moderate_to_severe': 2,
        'severe_to_death': 4,
    },

    # (VERIFIED, ONS) # Background mortality (annual absolute hazards by age band; replace with ONS by converting annual probability to hazard h = -ln(1-prob))
    'background_mortality_hazards_by_year':
        {       2023: {       'female': {       65: 0.00784,
                                                66: 0.00894,
                                                67: 0.00983,
                                                68: 0.01046,
                                                69: 0.01145,
                                                70: 0.01248,
                                                71: 0.001362,
                                                72: 0.01525,
                                                73: 0.01662,
                                                74: 0.01828,
                                                75: 0.02075,
                                                76: 0.02299,
                                                77: 0.02547,
                                                78: 0.02977,
                                                79: 0.03295,
                                                80: 0.03838,
                                                81: 0.04200,
                                                82: 0.04895,
                                                83: 0.05558,
                                                84: 0.06335,
                                                85: 0.06959,
                                                86: 0.08102,
                                                87: 0.09163,
                                                88: 0.10411,
                                                89: 0.11644,
                                                90: 0.13230,
                                                91: 0.15243,
                                                92: 0.17344,
                                                93: 0.18859,
                                                94: 0.20818,
                                                95: 0.22701,
                                                96: 0.24778,
                                                97: 0.27310,
                                                98: 0.30048,
                                                99: 0.32332,
                                                100: 0.34060},
                              'male': {       65: 0.01187,
                                              66: 0.01307,
                                              67: 0.01464,
                                              68: 0.01641,
                                              69: 0.01734,
                                              70: 0.01959,
                                              71: 0.02136,
                                              72: 0.02322,
                                              73: 0.02493,
                                              74: 0.02737,
                                              75: 0.03058,
                                              76: 0.03333,
                                              77: 0.03592,
                                              78: 0.04213,
                                              79: 0.04706,
                                              80: 0.05333,
                                              81: 0.05936,
                                              82: 0.06760,
                                              83: 0.07471,
                                              84: 0.08379,
                                              85: 0.09400,
                                              86: 0.10567,
                                              87: 0.11721,
                                              88: 0.13328,
                                              89: 0.15016,
                                              90: 0.16782,
                                              91: 0.18525,
                                              92: 0.20765,
                                              93: 0.22646,
                                              94: 0.25228,
                                              95: 0.27048,
                                              96: 0.29464,
                                              97: 0.32385,
                                              98: 0.33568,
                                              99: 0.35257,
                                              100: 0.38669}},
                2024: {       'female': {       65: 0.00778,
                                                66: 0.00854,
                                                67: 0.00938,
                                                68: 0.01032,
                                                69: 0.01137,
                                                70: 0.01253,
                                                71: 0.01380,
                                                72: 0.01517,
                                                73: 0.01666,
                                                74: 0.01832,
                                                75: 0.02021,
                                                76: 0.02245,
                                                77: 0.02508,
                                                78: 0.02817,
                                                79: 0.03181,
                                                80: 0.03611,
                                                81: 0.04116,
                                                82: 0.04703,
                                                83: 0.05372,
                                                84: 0.06118,
                                                85: 0.06941,
                                                86: 0.07854,
                                                87: 0.08886,
                                                88: 0.10068,
                                                89: 0.11416,
                                                90: 0.13011,
                                                91: 0.14896,
                                                92: 0.16963,
                                                93: 0.19192,
                                                94: 0.21625,
                                                95: 0.24244,
                                                96: 0.27003,
                                                97: 0.29843,
                                                98: 0.32711,
                                                99: 0.35554,
                                                100: 0.38342},
                              'male': {       65: 0.01209,
                                              66: 0.01321,
                                              67: 0.01448,
                                              68: 0.01590,
                                              69: 0.01748,
                                              70: 0.01922,
                                              71: 0.02108,
                                              72: 0.02305,
                                              73: 0.02514,
                                              74: 0.02737,
                                              75: 0.02983,
                                              76: 0.03268,
                                              77: 0.03610,
                                              78: 0.04029,
                                              79: 0.04534,
                                              80: 0.05124,
                                              81: 0.05797,
                                              82: 0.06547,
                                              83: 0.07369,
                                              84: 0.08259,
                                              85: 0.09229,
                                              86: 0.10305,
                                              87: 0.11514,
                                              88: 0.12882,
                                              89: 0.14427,
                                              90: 0.61659,
                                              91: 0.18086,
                                              92: 0.20158,
                                              93: 0.22410,
                                              94: 0.24967,
                                              95: 0.27760,
                                              96: 0.30650,
                                              97: 0.33596,
                                              98: 0.36545,
                                              99: 0.39420,
                                              100: 0.42197}},
                2025: {       'female': {       65: 0.00762,
                                                66: 0.00836,
                                                67: 0.00919,
                                                68: 0.01013,
                                                69: 0.01117,
                                                70: 0.01233,
                                                71: 0.01361,
                                                72: 0.01500,
                                                73: 0.01650,
                                                74: 0.01814,
                                                75: 0.01998,
                                                76: 0.02211,
                                                77: 0.02463,
                                                78: 0.02761,
                                                79: 0.03115,
                                                80: 0.03533,
                                                81: 0.04025,
                                                82: 0.04599,
                                                83: 0.05263,
                                                84: 0.06014,
                                                85: 0.06845,
                                                86: 0.07755,
                                                87: 0.08762,
                                                88: 0.09901,
                                                89: 0.11201,
                                                90: 0.12760,
                                                91: 0.14624,
                                                92: 0.16672,
                                                93: 0.18875,
                                                94: 0.21283,
                                                95: 0.23890,
                                                96: 0.26669,
                                                97: 0.29562,
                                                98: 0.32511,
                                                99: 0.35450,
                                                100: 0.38335},
                              'male': {       65: 0.01193,
                                              66: 0.01301,
                                              67: 0.01423,
                                              68: 0.01560,
                                              69: 0.01715,
                                              70: 0.01887,
                                              71: 0.02077,
                                              72: 0.02281,
                                              73: 0.02497,
                                              74: 0.02725,
                                              75: 0.02972,
                                              76: 0.03246,
                                              77: 0.03566,
                                              78: 0.03952,
                                              79: 0.04424,
                                              80: 0.04990,
                                              81: 0.05652,
                                              82: 0.06404,
                                              83: 0.07239,
                                              84: 0.08148,
                                              85: 0.09128,
                                              86: 0.10192,
                                              87: 0.11372,
                                              88: 0.12702,
                                              89: 0.14211,
                                              90: 0.15911,
                                              91: 0.17810,
                                              92: 0.19854,
                                              93: 0.22079,
                                              94: 0.24625,
                                              95: 0.27426,
                                              96: 0.30336,
                                              97: 0.33309,
                                              98: 0.36290,
                                              99: 0.39220,
                                              100: 0.42085}},
                2026: {       'female': {       65: 0.00748,
                                                66: 0.00820,
                                                67: 0.00901,
                                                68: 0.00993,
                                                69: 0.01096,
                                                70: 0.01212,
                                                71: 0.01339,
                                                72: 0.01479,
                                                73: 0.01631,
                                                74: 0.01797,
                                                75: 0.01979,
                                                76: 0.02186,
                                                77: 0.02427,
                                                78: 0.02713,
                                                79: 0.03055,
                                                80: 0.03462,
                                                81: 0.03942,
                                                82: 0.04502,
                                                83: 0.05153,
                                                84: 0.05899,
                                                85: 0.06735,
                                                86: 0.07653,
                                                87: 0.08657,
                                                88: 0.09770,
                                                89: 0.11024,
                                                90: 0.12535,
                                                91: 0.14365,
                                                92: 0.16394,
                                                93: 0.18579,
                                                94: 0.20962,
                                                95: 0.23545,
                                                96: 0.26315,
                                                97: 0.29231,
                                                98: 0.32232,
                                                99: 0.35247,
                                                100: 0.38230},
                              'male': {       65: 0.01179,
                                              66: 0.01284,
                                              67: 0.01401,
                                              68: 0.01534,
                                              69: 0.01683,
                                              70: 0.01852,
                                              71: 0.02040,
                                              72: 0.02247,
                                              73: 0.02470,
                                              74: 0.02706,
                                              75: 0.02958,
                                              76: 0.03232,
                                              77: 0.03542,
                                              78: 0.03905,
                                              79: 0.04344,
                                              80: 0.04876,
                                              81: 0.05513,
                                              82: 0.06253,
                                              83: 0.07090,
                                              84: 0.08013,
                                              85: 0.09013,
                                              86: 0.10088,
                                              87: 0.11257,
                                              88: 0.12557,
                                              89: 0.14028,
                                              90: 0.15693,
                                              91: 0.17562,
                                              92: 0.19575,
                                              93: 0.21769,
                                              94: 0.24294,
                                              95: 0.27089,
                                              96: 0.30008,
                                              97: 0.33003,
                                              98: 0.36018,
                                              99: 0.38993,
                                              100: 0.41897}},
                2027: {       'female': {       65: 0.00736,
                                                66: 0.00805,
                                                67: 0.00884,
                                                68: 0.00974,
                                                69: 0.01076,
                                                70: 0.01190,
                                                71: 0.01317,
                                                72: 0.01456,
                                                73: 0.01610,
                                                74: 0.01777,
                                                75: 0.01961,
                                                76: 0.02166,
                                                77: 0.02400,
                                                78: 0.02675,
                                                79: 0.03004,
                                                80: 0.03400,
                                                81: 0.03868,
                                                82: 0.04416,
                                                83: 0.05052,
                                                84: 0.05785,
                                                85: 0.06615,
                                                86: 0.07539,
                                                87: 0.08553,
                                                88: 0.09663,
                                                89: 0.10890,
                                                90: 0.12356,
                                                91: 0.14140,
                                                92: 0.16138,
                                                93: 0.18305,
                                                94: 0.20672,
                                                95: 0.23232,
                                                96: 0.25981,
                                                97: 0.28893,
                                                98: 0.31921,
                                                99: 0.34993,
                                                100: 0.38056},
                              'male': {       65: 0.01168,
                                              66: 0.01269,
                                              67: 0.01384,
                                              68: 0.01512,
                                              69: 0.01656,
                                              70: 0.01820,
                                              71: 0.02004,
                                              72: 0.02210,
                                              73: 0.02436,
                                              74: 0.02679,
                                              75: 0.02939,
                                              76: 0.03219,
                                              77: 0.03529,
                                              78: 0.03882,
                                              79: 0.04297,
                                              80: 0.04795,
                                              81: 0.05397,
                                              82: 0.06111,
                                              83: 0.06936,
                                              84: 0.07863,
                                              85: 0.08879,
                                              86: 0.09976,
                                              87: 0.1156,
                                              88: 0.12448,
                                              89: 0.13890,
                                              90: 0.15517,
                                              91: 0.17352,
                                              92: 0.19335,
                                              93: 0.21498,
                                              94: 0.23998,
                                              95: 0.26777,
                                              96: 0.29695,
                                              97: 0.32700,
                                              98: 0.35738,
                                              99: 0.38742,
                                              100: 0.41678}},
                2028: {       'female': {       65: 0.00727,
                                                66: 0.00793,
                                                67: 0.00869,
                                                68: 0.00957,
                                                69: 0.01057,
                                                70: 0.01170,
                                                71: 0.01295,
                                                72: 0.01434,
                                                73: 0.01587,
                                                74: 0.01755,
                                                75: 0.01941,
                                                76: 0.02147,
                                                77: 0.02380,
                                                78: 0.02648,
                                                79: 0.02967,
                                                80: 0.03349,
                                                81: 0.03805,
                                                82: 0.04342,
                                                83: 0.04966,
                                                84: 0.05684,
                                                85: 0.06501,
                                                86: 0.07420,
                                                87: 0.08441,
                                                88: 0.09561,
                                                89: 0.10786,
                                                90: 0.12228,
                                                91: 0.13970,
                                                92: 0.15923,
                                                93: 0.18061,
                                                94: 0.20413,
                                                95: 0.22963,
                                                96: 0.25694,
                                                97: 0.28589,
                                                98: 0.31620,
                                                99: 0.34725,
                                                100: 0.37850},
                              'male': {       65: 0.01159,
                                              66: 0.01259,
                                              67: 0.01370,
                                              68: 0.01495,
                                              69: 0.01635,
                                              70: 0.01793,
                                              71: 0.01973,
                                              72: 0.02175,
                                              73: 0.02400,
                                              74: 0.02647,
                                              75: 0.02914,
                                              76: 0.03202,
                                              77: 0.03518,
                                              78: 0.03872,
                                              79: 0.04277,
                                              80: 0.04751,
                                              81: 0.05318,
                                              82: 0.05998,
                                              83: 0.06798,
                                              84: 0.07713,
                                              85: 0.08734,
                                              86: 0.09849,
                                              87: 0.11054,
                                              88: 0.12359,
                                              89: 0.13794,
                                              90: 0.15394,
                                              91: 0.17195,
                                              92: 0.19144,
                                              93: 0.21276,
                                              94: 0.23755,
                                              95: 0.26517,
                                              96: 0.29425,
                                              97: 0.32430,
                                              98: 0.35477,
                                              99: 0.38503,
                                              100: 0.41475}},
                2029: {       'female': {       65: 0.00719,
                                                66: 0.00783,
                                                67: 0.00857,
                                                68: 0.00942,
                                                69: 0.01039,
                                                70: 0.01150,
                                                71: 0.01274,
                                                72: 0.01412,
                                                73: 0.01564,
                                                74: 0.01732,
                                                75: 0.01918,
                                                76: 0.02126,
                                                77: 0.02360,
                                                78: 0.02627,
                                                79: 0.02939,
                                                80: 0.03310,
                                                81: 0.03753,
                                                82: 0.04277,
                                                83: 0.04889,
                                                84: 0.05594,
                                                85: 0.06397,
                                                86: 0.07304,
                                                87: 0.08320,
                                                88: 0.09447,
                                                89: 0.10683,
                                                90: 0.12123,
                                                91: 0.13844,
                                                92: 0.15757,
                                                93: 0.17852,
                                                94: 0.20177,
                                                95: 0.22717,
                                                96: 0.25440,
                                                97: 0.28320,
                                                98: 0.31335,
                                                99: 0.34446,
                                                100: 0.37601},
                              'male': {       65: 0.01153,
                                              66: 0.01250,
                                              67: 0.01359,
                                              68: 0.01481,
                                              69: 0.01617,
                                              70: 0.01772,
                                              71: 0.01946,
                                              72: 0.02143,
                                              73: 0.02364,
                                              74: 0.02610,
                                              75: 0.02881,
                                              76: 0.03177,
                                              77: 0.03501,
                                              78: 0.03861,
                                              79: 0.04267,
                                              80: 0.04731,
                                              81: 0.05275,
                                              82: 0.05919,
                                              83: 0.06684,
                                              84: 0.07575,
                                              85: 0.08585,
                                              86: 0.09706,
                                              87: 0.10931,
                                              88: 0.12262,
                                              89: 0.13711,
                                              90: 0.15305,
                                              91: 0.17081,
                                              92: 0.18995,
                                              93: 0.21094,
                                              94: 0.23549,
                                              95: 0.26296,
                                              96: 0.29191,
                                              97: 0.32191,
                                              98: 0.35244,
                                              99: 0.38284,
                                              100: 0.41296}},
                2030: {       'female': {       65: 0.00713,
                                                66: 0.00775,
                                                67: 0.00846,
                                                68: 0.00929,
                                                69: 0.01023,
                                                70: 0.01131,
                                                71: 0.001254,
                                                72: 0.01390,
                                                73: 0.01541,
                                                74: 0.01707,
                                                75: 0.01893,
                                                76: 0.02102,
                                                77: 0.02337,
                                                78: 0.02605,
                                                79: 0.02915,
                                                80: 0.03279,
                                                81: 0.03710,
                                                82: 0.04220,
                                                83: 0.04819,
                                                84: 0.05512,
                                                85: 0.06302,
                                                86: 0.07193,
                                                87: 0.08196,
                                                88: 0.09318,
                                                89: 0.10561,
                                                90: 0.12014,
                                                91: 0.13737,
                                                92: 0.15630,
                                                93: 0.17683,
                                                94: 0.19968,
                                                95: 0.22482,
                                                96: 0.25195,
                                                97: 0.28069,
                                                98: 0.31072,
                                                99: 0.34171,
                                                100: 0.37330},
                              'male': {       65: 0.01147,
                                              66: 0.01242,
                                              67: 0.01349,
                                              68: 0.01468,
                                              69: 0.01602,
                                              70: 0.01753,
                                              71: 0.01923,
                                              72: 0.02115,
                                              73: 0.02331,
                                              74: 0.02573,
                                              75: 0.02843,
                                              76: 0.03142,
                                              77: 0.03474,
                                              78: 0.03843,
                                              79: 0.04255,
                                              80: 0.04720,
                                              81: 0.05252,
                                              82: 0.05872,
                                              83: 0.06602,
                                              84: 0.07457,
                                              85: 0.08442,
                                              86: 0.09553,
                                              87: 0.10784,
                                              88: 0.12135,
                                              89: 0.13610,
                                              90: 0.15220,
                                              91: 0.16991,
                                              92: 0.18881,
                                              93: 0.20944,
                                              94: 0.23370,
                                              95: 0.26067,
                                              96: 0.28977,
                                              97: 0.31968,
                                              98: 0.35014,
                                              99: 0.38061,
                                              100: 0.41081}},
                2031: {       'female': {       65: 0.00707,
                                                66: 0.00768,
                                                67: 0.00837,
                                                68: 0.00917,
                                                69: 0.01009,
                                                70: 0.01114,
                                                71: 0.01234,
                                                72: 0.01368,
                                                73: 0.01517,
                                                74: 0.01682,
                                                75: 0.01867,
                                                76: 0.02075,
                                                77: 0.02310,
                                                78: 0.02579,
                                                79: 0.02890,
                                                80: 0.03252,
                                                81: 0.03676,
                                                82: 0.04174,
                                                83: 0.04759,
                                                84: 0.05438,
                                                85: 0.06215,
                                                86: 0.07092,
                                                87: 0.08079,
                                                88: 0.09187,
                                                89: 0.10425,
                                                90: 0.11886,
                                                91: 0.13623,
                                                92: 0.15520,
                                                93: 0.17554,
                                                94: 0.19799,
                                                95: 0.22272,
                                                96: 0.24962,
                                                97: 0.27830,
                                                98: 0.30832,
                                                99: 0.33923,
                                                100: 0.37080},
                              'male': {       65: 0.01140,
                                              66: 0.01235,
                                              67: 0.01340,
                                              68: 0.01457,
                                              69: 0.01589,
                                              70: 0.01736,
                                              71: 0.01902,
                                              72: 0.02090,
                                              73: 0.02301,
                                              74: 0.02538,
                                              75: 0.02804,
                                              76: 0.03102,
                                              77: 0.03438,
                                              78: 0.03813,
                                              79: 0.04233,
                                              80: 0.04704,
                                              81: 0.05237,
                                              82: 0.05846,
                                              83: 0.06550,
                                              84: 0.07369,
                                              85: 0.08319,
                                              86: 0.09405,
                                              87: 0.10627,
                                              88: 0.11984,
                                              89: 0.13480,
                                              90: 0.15116,
                                              91: 0.16905,
                                              92: 0.18790,
                                              93: 0.20826,
                                              94: 0.23220,
                                              95: 0.25922,
                                              96: 0.28783,
                                              97: 0.31761,
                                              98: 0.34800,
                                              99: 0.37841,
                                              100: 0.40860}},
                2032: {       'female': {       65: 0.00702,
                                                66: 0.00761,
                                                67: 0.000829,
                                                68: 0.00907,
                                                69: 0.00997,
                                                70: 0.01099,
                                                71: 0.01216,
                                                72: 0.01347,
                                                73: 0.01493,
                                                74: 0.01657,
                                                75: 0.01840,
                                                76: 0.02046,
                                                77: 0.02281,
                                                78: 0.02550,
                                                79: 0.02862,
                                                80: 0.03224,
                                                81: 0.03646,
                                                82: 0.04136,
                                                83: 0.04708,
                                                84: 0.05372,
                                                85: 0.06135,
                                                86: 0.06998,
                                                87: 0.07971,
                                                88: 0.09063,
                                                89: 0.10286,
                                                90: 0.11743,
                                                91: 0.13490,
                                                92: 0.15402,
                                                93: 0.17441,
                                                94: 0.19667,
                                                95: 0.22103,
                                                96: 0.24756,
                                                97: 0.27603,
                                                98: 0.30598,
                                                99: 0.33686,
                                                100: 0.36837},
                              'male': {       65: 0.01134,
                                              66: 0.01228,
                                              67: 0.01332,
                                              68: 0.01447,
                                              69: 0.01576,
                                              70: 0.01721,
                                              71: 0.01884,
                                              72: 0.02068,
                                              73: 0.02274,
                                              74: 0.02506,
                                              75: 0.02767,
                                              76: 0.03061,
                                              77: 0.03395,
                                              78: 0.03774,
                                              79: 0.04201,
                                              80: 0.04679,
                                              81: 0.05218,
                                              82: 0.05827,
                                              83: 0.06520,
                                              84: 0.07313,
                                              85: 0.08227,
                                              86: 0.09277,
                                              87: 0.10474,
                                              88: 0.11822,
                                              89: 0.13324,
                                              90: 0.14981,
                                              91: 0.16798,
                                              92: 0.18698,
                                              93: 0.20728,
                                              94: 0.23100,
                                              95: 0.25777,
                                              96: 0.28615,
                                              97: 0.31579,
                                              98: 0.34610,
                                              99: 0.37643,
                                              100: 0.40662}},
                2033: {       'female': {       65: 0.00696,
                                                66: 0.00755,
                                                67: 0.00822,
                                                68: 0.00898,
                                                69: 0.00986,
                                                70: 0.01086,
                                                71: 0.01199,
                                                72: 0.01327,
                                                73: 0.01471,
                                                74: 0.01632,
                                                75: 0.01813,
                                                76: 0.02018,
                                                77: 0.02251,
                                                78: 0.02519,
                                                79: 0.02830,
                                                80: 0.03193,
                                                81: 0.03615,
                                                82: 0.04102,
                                                83: 0.04665,
                                                84: 0.05317,
                                                85: 0.06064,
                                                86: 0.06912,
                                                87: 0.07871,
                                                88: 0.08948,
                                                89: 0.10154,
                                                90: 0.11596,
                                                91: 0.13341,
                                                92: 0.15265,
                                                93: 0.17320,
                                                94: 0.19551,
                                                95: 0.21970,
                                                96: 0.24586,
                                                97: 0.27399,
                                                98: 0.30376,
                                                99: 0.33460,
                                                100: 0.36612},
                              'male': {       65: 0.01126,
                                              66: 0.01220,
                                              67: 0.01324,
                                              68: 0.01438,
                                              69: 0.01565,
                                              70: 0.01707,
                                              71: 0.01867,
                                              72: 0.02048,
                                              73: 0.02250,
                                              74: 0.02478,
                                              75: 0.02733,
                                              76: 0.03023,
                                              77: 0.03352,
                                              78: 0.03729,
                                              79: 0.04159,
                                              80: 0.04644,
                                              81: 0.05189,
                                              82: 0.05803,
                                              83: 0.06495,
                                              84: 0.07277,
                                              85: 0.08165,
                                              86: 0.09180,
                                              87: 0.10341,
                                              88: 0.11665,
                                              89: 0.13157,
                                              90: 0.14820,
                                              91: 0.16659,
                                              92: 0.18586,
                                              93: 0.20631,
                                              94: 0.23002,
                                              95: 0.25658,
                                              96: 0.28472,
                                              97: 0.31416,
                                              98: 0.34431,
                                              99: 0.37458,
                                              100: 0.40467}},
                2034: {       'female': {       65: 0.00691,
                                                66: 0.00749,
                                                67: 0.00815,
                                                68: 0.00890,
                                                69: 0.00976,
                                                70: 0.01074,
                                                71: 0.01185,
                                                72: 0.01310,
                                                73: 0.01451,
                                                74: 0.01609,
                                                75: 0.01787,
                                                76: 0.01989,
                                                77: 0.02220,
                                                78: 0.02486,
                                                79: 0.02796,
                                                80: 0.03158,
                                                81: 0.03580,
                                                82: 0.04067,
                                                83: 0.04627,
                                                84: 0.05269,
                                                85: 0.06002,
                                                86: 0.06835,
                                                87: 0.07778,
                                                88: 0.08840,
                                                89: 0.10031,
                                                90: 0.11457,
                                                91: 0.13189,
                                                92: 0.15112,
                                                93: 0.17179,
                                                94: 0.19427,
                                                95: 0.21581,
                                                96: 0.24452,
                                                97: 0.27232,
                                                98: 0.30181,
                                                99: 0.33252,
                                                100: 0.36406},
                              'male': {       65: 0.01117,
                                              66: 0.01211,
                                              67: 0.01314,
                                              68: 0.01428,
                                              69: 0.01553,
                                              70: 0.01694,
                                              71: 0.01852,
                                              72: 0.02029,
                                              73: 0.02229,
                                              74: 0.02452,
                                              75: 0.02703,
                                              76: 0.02987,
                                              77: 0.03312,
                                              78: 0.03684,
                                              79: 0.04111,
                                              80: 0.04598,
                                              81: 0.05149,
                                              82: 0.05770,
                                              83: 0.06466,
                                              84: 0.07246,
                                              85: 0.08122,
                                              86: 0.09111,
                                              87: 0.10237,
                                              88: 0.11526,
                                              89: 0.12993,
                                              90: 0.14647,
                                              91: 0.16495,
                                              92: 0.18443,
                                              93: 0.20515,
                                              94: 0.22903,
                                              95: 0.25558,
                                              96: 0.28355,
                                              97: 0.31275,
                                              98: 0.34272,
                                              99: 0.37287,
                                              100: 0.40288}},
                2035: {       'female': {       65: 0.00685,
                                                66: 0.00743,
                                                67: 0.00808,
                                                68: 0.00882,
                                                69: 0.00967,
                                                70: 0.01063,
                                                71: 0.01172,
                                                72: 0.01295,
                                                73: 0.01432,
                                                74: 0.01587,
                                                75: 0.01762,
                                                76: 0.01962,
                                                77: 0.02190,
                                                78: 0.02453,
                                                79: 0.02761,
                                                80: 0.03121,
                                                81: 0.03541,
                                                82: 0.04027,
                                                83: 0.04586,
                                                84: 0.05224,
                                                85: 0.05948,
                                                86: 0.06767,
                                                87: 0.07694,
                                                88: 0.08740,
                                                89: 0.09916,
                                                90: 0.11326,
                                                91: 0.13043,
                                                92: 0.14955,
                                                93: 0.17022,
                                                94: 0.19282,
                                                95: 0.21723,
                                                96: 0.24331,
                                                97: 0.27097,
                                                98: 0.30014,
                                                99: 0.33060,
                                                100: 0.36201},
                              'male': {       65: 0.01107,
                                              66: 0.01201,
                                              67: 0.01304,
                                              68: 0.01417,
                                              69: 0.01542,
                                              70: 0.01681,
                                              71: 0.01837,
                                              72: 0.02012,
                                              73: 0.02208,
                                              74: 0.02428,
                                              75: 0.02675,
                                              76: 0.02955,
                                              77: 0.03274,
                                              78: 0.03641,
                                              79: 0.04063,
                                              80: 0.04548,
                                              81: 0.05100,
                                              82: 0.05725,
                                              83: 0.06427,
                                              84: 0.07210,
                                              85: 0.08084,
                                              86: 0.09061,
                                              87: 0.10161,
                                              88: 0.11415,
                                              89: 0.12847,
                                              90: 0.14478,
                                              91: 0.16318,
                                              92: 0.18276,
                                              93: 0.20368,
                                              94: 0.22780,
                                              95: 0.25456,
                                              96: 0.28255,
                                              97: 0.31160,
                                              98: 0.34137,
                                              99: 0.37134,
                                              100: 0.40138}},
                2036: {       'female': {       65: 0.00678,
                                                66: 0.00736,
                                                67: 0.00801,
                                                68: 0.00874,
                                                69: 0.00958,
                                                70: 0.01053,
                                                71: 0.01160,
                                                72: 0.01280,
                                                73: 0.01415,
                                                74: 0.01567,
                                                75: 0.01739,
                                                76: 0.01935,
                                                77: 0.02160,
                                                78: 0.02421,
                                                79: 0.02726,
                                                80: 0.03083,
                                                81: 0.03500,
                                                82: 0.03985,
                                                83: 0.04543,
                                                84: 0.05179,
                                                85: 0.05898,
                                                86: 0.06706,
                                                87: 0.07618,
                                                88: 0.08647,
                                                89: 0.09806,
                                                90: 0.11202,
                                                91: 0.12905,
                                                92: 0.14804,
                                                93: 0.16861,
                                                94: 0.19122,
                                                95: 0.21576,
                                                96: 0.24201,
                                                97: 0.26974,
                                                98: 0.29879,
                                                99: 0.32902,
                                                100: 0.36020},
                              'male': {       65: 0.01096,
                                              66: 0.01189,
                                              67: 0.01293,
                                              68: 0.01405,
                                              69: 0.01530,
                                              70: 0.01668,
                                              71: 0.01822,
                                              72: 0.01995,
                                              73: 0.02188,
                                              74: 0.02405,
                                              75: 0.02649,
                                              76: 0.02924,
                                              77: 0.03239,
                                              78: 0.03600,
                                              79: 0.04017,
                                              80: 0.04496,
                                              81: 0.05045,
                                              82: 0.05671,
                                              83: 0.06376,
                                              84: 0.07164,
                                              85: 0.08040,
                                              86: 0.09014,
                                              87: 0.10101,
                                              88: 0.11329,
                                              89: 0.12727,
                                              90: 0.14325,
                                              91: 0.16145,
                                              92: 0.18097,
                                              93: 0.20196,
                                              94: 0.22630,
                                              95: 0.25332,
                                              96: 0.28150,
                                              97: 0.31056,
                                              98: 0.34019,
                                              99: 0.36996,
                                              100: 0.39996}},
                2037: {       'female': {       65: 0.00671,
                                                66: 0.00729,
                                                67: 0.00793,
                                                68: 0.00866,
                                                69: 0.00949,
                                                70: 0.01043,
                                                71: 0.01149,
                                                72: 0.01267,
                                                73: 0.01400,
                                                74: 0.01549,
                                                75: 0.01717,
                                                76: 0.01910,
                                                77: 0.02132,
                                                78: 0.02390,
                                                79: 0.02691,
                                                80: 0.03045,
                                                81: 0.03459,
                                                82: 0.03940,
                                                83: 0.04495,
                                                84: 0.05130,
                                                85: 0.05846,
                                                86: 0.06648,
                                                87: 0.07549,
                                                88: 0.08563,
                                                89: 0.09705,
                                                90: 0.11083,
                                                91: 0.12774,
                                                92: 0.14660,
                                                93: 0.16705,
                                                94: 0.18957,
                                                95: 0.21413,
                                                96: 0.24053,
                                                97: 0.26843,
                                                98: 0.29757,
                                                99: 0.32772,
                                                100: 0.035868},
                              'male': {       65: 0.01085,
                                              66: 0.01178,
                                              67: 0.01280,
                                              68: 0.01393,
                                              69: 0.01517,
                                              70: 0.01654,
                                              71: 0.01807,
                                              72: 0.01978,
                                              73: 0.02169,
                                              74: 0.02384,
                                              75: 0.02624,
                                              76: 0.02895,
                                              77: 0.03206,
                                              78: 0.03562,
                                              79: 0.03973,
                                              80: 0.04446,
                                              81: 0.04989,
                                              82: 0.05611,
                                              83: 0.06316,
                                              84: 0.07106,
                                              85: 0.07985,
                                              86: 0.08960,
                                              87: 0.10044,
                                              88: 0.11258,
                                              89: 0.12630,
                                              90: 0.14195,
                                              91: 0.15987,
                                              92: 0.17920,
                                              93: 0.20015,
                                              94: 0.22459,
                                              95: 0.25182,
                                              96: 0.28022,
                                              97: 0.30945,
                                              98: 0.33913,
                                              99: 0.36882,
                                              100: 0.39870}},
                2038: {       'female': {       65: 0.00664,
                                                66: 0.00721,
                                                67: 0.00785,
                                                68: 0.00858,
                                                69: 0.00940,
                                                70: 0.01033,
                                                71: 0.01138,
                                                72: 0.01254,
                                                73: 0.01385,
                                                74: 0.01532,
                                                75: 0.01697,
                                                76: 0.01887,
                                                77: 0.02105,
                                                78: 0.02359,
                                                79: 0.02657,
                                                80: 0.03007,
                                                81: 0.03417,
                                                82: 0.03895,
                                                83: 0.04446,
                                                84: 0.05077,
                                                85: 0.05790,
                                                86: 0.06589,
                                                87: 0.07482,
                                                88: 0.08485,
                                                89: 0.09610,
                                                90: 0.10972,
                                                91: 0.12648,
                                                92: 0.14522,
                                                93: 0.16556,
                                                94: 0.18798,
                                                95: 0.21246,
                                                96: 0.23887,
                                                97: 0.26693,
                                                98: 0.29623,
                                                99: 0.32647,
                                                100: 0.35738},
                              'male': {       65: 0.01073,
                                              66: 0.01166,
                                              67: 0.01267,
                                              68: 0.01379,
                                              69: 0.01502,
                                              70: 0.01639,
                                              71: 0.01791,
                                              72: 0.01961,
                                              73: 0.02150,
                                              74: 0.02362,
                                              75: 0.02599,
                                              76: 0.02868,
                                              77: 0.03174,
                                              78: 0.03526,
                                              79: 0.03932,
                                              80: 0.04399,
                                              81: 0.04935,
                                              82: 0.05550,
                                              83: 0.06250,
                                              84: 0.07039,
                                              85: 0.07920,
                                              86: 0.08896,
                                              87: 0.09979,
                                              88: 0.11188,
                                              89: 0.12546,
                                              90: 0.14087,
                                              91: 0.15849,
                                              92: 0.17757,
                                              93: 0.19835,
                                              94: 0.22275,
                                              95: 0.25008,
                                              96: 0.27867,
                                              97: 0.30814,
                                              98: 0.33803,
                                              99: 0.36781,
                                              100: 0.39757}},
                2039: {       'female': {       65: 0.00657,
                                                66: 0.00714,
                                                67: 0.00777,
                                                68: 0.00849,
                                                69: 0.00931,
                                                70: 0.01023,
                                                71: 0.01126,
                                                72: 0.01242,
                                                73: 0.01371,
                                                74: 0.01516,
                                                75: 0.01679,
                                                76: 0.01865,
                                                77: 0.02080,
                                                78: 0.02330,
                                                79: 0.02624,
                                                80: 0.02971,
                                                81: 0.03377,
                                                82: 0.03849,
                                                83: 0.04396,
                                                84: 0.05022,
                                                85: 0.05730,
                                                86: 0.06525,
                                                87: 0.07414,
                                                88: 0.08409,
                                                89: 0.09522,
                                                90: 0.10868,
                                                91: 0.12528,
                                                92: 0.14390,
                                                93: 0.16413,
                                                94: 0.18645,
                                                95: 0.21084,
                                                96: 0.23718,
                                                97: 0.26527,
                                                98: 0.29473,
                                                99: 0.32512,
                                                100: 0.35613},
                              'male': {       65: 0.01060,
                                              66: 0.01153,
                                              67: 0.01254,
                                              68: 0.01365,
                                              69: 0.01487,
                                              70: 0.01623,
                                              71: 0.01775,
                                              72: 0.01943,
                                              73: 0.02131,
                                              74: 0.02341,
                                              75: 0.02575,
                                              76: 0.02841,
                                              77: 0.03143,
                                              78: 0.03491,
                                              79: 0.03892,
                                              80: 0.04353,
                                              81: 0.04883,
                                              82: 0.05491,
                                              83: 0.06184,
                                              84: 0.06967,
                                              85: 0.07845,
                                              86: 0.08821,
                                              87: 0.09904,
                                              88: 0.11111,
                                              89: 0.12462,
                                              90: 0.13989,
                                              91: 0.15731,
                                              92: 0.17613,
                                              93: 0.19667,
                                              94: 0.22092,
                                              95: 0.24823,
                                              96: 0.27695,
                                              97: 0.30662,
                                              98: 0.33670,
                                              99: 0.36665,
                                              100: 0.39647}},
                2040: {       'female': {       65: 0.00650,
                                                66: 0.00706,
                                                67: 0.00769,
                                                68: 0.00840,
                                                69: 0.00921,
                                                70: 0.01013,
                                                71: 0.01115,
                                                72: 0.01229,
                                                73: 0.01357,
                                                74: 0.01500,
                                                75: 0.01661,
                                                76: 0.01845,
                                                77: 0.02057,
                                                78: 0.02303,
                                                79: 0.02593,
                                                80: 0.02935,
                                                81: 0.03337,
                                                82: 0.03804,
                                                83: 0.04346,
                                                84: 0.04966,
                                                85: 0.05669,
                                                86: 0.06459,
                                                87: 0.07343,
                                                88: 0.08331,
                                                89: 0.09435,
                                                90: 0.10769,
                                                91: 0.12415,
                                                92: 0.14263,
                                                93: 0.16275,
                                                94: 0.18498,
                                                95: 0.20927,
                                                96: 0.26359,
                                                97: 0.26359,
                                                98: 0.29311,
                                                99: 0.32364,
                                                100: 0.35480},
                              'male': {       65: 0.01046,
                                              66: 0.01139,
                                              67: 0.01240,
                                              68: 0.01351,
                                              69: 0.01472,
                                              70: 0.01607,
                                              71: 0.01757,
                                              72: 0.01925,
                                              73: 0.02111,
                                              74: 0.02319,
                                              75: 0.02551,
                                              76: 0.02813,
                                              77: 0.03113,
                                              78: 0.02457,
                                              79: 0.03853,
                                              80: 0.04309,
                                              81: 0.04833,
                                              82: 0.05434,
                                              83: 0.06119,
                                              84: 0.06895,
                                              85: 0.07766,
                                              86: 0.08738,
                                              87: 0.09819,
                                              88: 0.11024,
                                              89: 0.12370,
                                              90: 0.13890,
                                              91: 0.15621,
                                              92: 0.17485,
                                              93: 0.19517,
                                              94: 0.21921,
                                              95: 0.24639,
                                              96: 0.27511,
                                              97: 0.30491,
                                              98: 0.33516,
                                              99: 0.36530,
                                              100: 0.39527}}},
    'background_mortality_hazards_by_year_is_qx': True,
    # Calibrates background mortality downward (target ~600k deaths vs ~1.3M baseline run)
    'background_mortality_scalar': 0.46,
    # Exact per-year scalars (takes precedence over growth/schedule)
    'background_mortality_scalar_by_year': {year: 0.46 for year in range(2023, 2041)},
    # Optional per-year scalar schedule (overrides growth/scalar if provided)
    # Example: {2023: 0.686, 2024: 0.686, 2025: 0.700}
    'background_mortality_scalar_schedule': {},
    # Optional proportional growth applied year-by-year when no schedule is provided
    # Example: annual_rate=0.01 means +1% per year vs reference_year
    'background_mortality_scalar_growth': {
        'use': False,
        'annual_rate': 0.01,
        'reference_year': 2023,
    },

    # Risk factor definitions with prevalence and onset hazard ratios only
    'risk_factors': {
        'socioeconomic_disadvantage': {
            'prevalence': {'female': 0.0, 'male': 0.0},  # union of low income & low SES (assuming independence)
            'hazard_ratios': {'onset': {'all': 2.01}},
        },
        'low_education': {
            'prevalence': {'female': 0.0, 'male': 0.0},
            'hazard_ratios': {'onset': {'all': 1.64}},
        },
        'hearing_difficulty': {
            'prevalence': {'female': 0.36, 'male': 0.36},
            'hazard_ratios': {'onset': {'all': 1.21}},
        },
        'hypertension': {
            'prevalence': {'female': 0.452, 'male': 0.452},
            'hazard_ratios': {'onset': {'all': 1.36}},
        },
        'obesity': {
            'prevalence': {'female': 0.243, 'male': 0.243},
            'hazard_ratios': {'onset': {'all': 1.1}},
        },
        'lifestyle': {
            'prevalence': {'female': 0.0, 'male': 0.0},  # combined prevalence from diet, activity, lifestyle score
            'hazard_ratios': {'onset': {'all': 1.08}},
        },
        'excessive_alcohol_consumption': {
            'prevalence': {'female': 0.0, 'male': 0.0},
            'hazard_ratios': {'onset': {'all': 1.30}},
        },
        'smoking': {
            'prevalence': {'female': 0.0, 'male': 0.0},
            'hazard_ratios': {'onset': {'all': 1.16}},
        },
        'depression': {
            'prevalence': {'female': 0.068, 'male': 0.068},
            'hazard_ratios': {'onset': {'all': 1.93}},
        },
        'social_isolation': {
            'prevalence': {'female': 0.0, 'male': 0.0},
            'hazard_ratios': {'onset': {'all': 1.34}},
        },
        'diabetes': {
            'prevalence': {'female': 0.045, 'male': 0.045},
            'hazard_ratios': {'onset': {'all': 2.06}},
        },
        'air_pollution': {
            'prevalence': {'female': 0.0, 'male': 0.0},
            'hazard_ratios': {'onset': {'all': 1.27}},
        },
        'APOE_e4_carrier': {
            'prevalence': {'female': 0.256, 'male': 0.256},
            'hazard_ratios': {'onset': {'all': 3.03}},
        },
        'periodontal_disease': {
            'prevalence': {'female': 0.50, 'male': 0.50},
            'hazard_ratios': {'onset': {'all': 1.21}},
            # Time-varying prevalence based on Elamin & Anash (2023)
            # 39.7% relative increase from 2020-2050 (30 years)
            # Adjusted to 2023-2040 (17 years): 22.5% relative increase
            'prevalence_schedule': {
                'use': False,  # Set to True to enable growth scenario
                'baseline_year': 2023,
                'baseline_prevalence': {'female': 0.50, 'male': 0.50},
                'target_year': 2040,
                'target_prevalence': {'female': 0.6125, 'male': 0.6125},  # 50% * 1.225
                'interpolation': 'linear',  # linear growth from baseline to target
            },
        },
    },

    # Living setting transitions (per-cycle probabilities; one-directional per current setting, fair assumption given nature of disease)
    # keyed by stage, then by (lower_age_inclusive, upper_age_inclusive_or_None) band  (VERIFIED, OHE)
    # 65+ only model - removed (35, 65) age band
    'living_setting_transition_probabilities': {
        'mild': {
            (65, None): {'to_institution': 0.066, 'to_home': 0},
        },
        'moderate': {
            (65, None): {'to_institution': 0.143, 'to_home': 0},
        },
        'severe': {
            (65, None): {'to_institution': 0.179, 'to_home': 0},
        },
    },

    # Utility norms by age band (baseline utilities, EQ-5D) split by sex
    # 65+ only model - removed ages below 65
    'utility_norms_by_age': {
        'female': {
            65: 0.78,
            75: 0.71,
        },
        'male': {
            65: 0.78,
            75: 0.75,
        },
    },

    # --- NEW: Parametric (Cox-style) age effects for hazards ---
    'age_hr_parametric': {
        'use': True,      # flip to False to use banded HRs above
        'ref_age': 65,    # baseline age where HR(age)=1
        'betas': {        # per-year log-hazard slopes (increase) (tune to data/literature)
            # Calibrated to target ~90k onsets in 2023-2024 under current run settings.
            'onset': 0.05,     #(CALIBRATED)
            'mild_to_moderate': 0,   #(VERIFIED, Biondo et al., 2022 - 3% increase in hazard of dementia per year of age = beta = ln(1.03)=0.03)
            'moderate_to_severe': 0,    #(VERIFIED, a 10-year age difference gives only 16% higher hazard of progressing to severe, observe slower symptom progression in very old patients - 33% increase over a 10-year age gap corresponds to ln(1.33)/10 = 0.029, which is too high)
            'severe_to_death': 0,   #(VERIFIED, an 80-year old with severe dementia has only slightly higher dementia-specific death hazard than a 70-year old with severe dementia, additional mortality so don't need to use such a large effect)
        },
    },

    'discount_rate_annual': 0.0,

    # Discounting convention for costs/QALYs:
    # - 'start': start-of-cycle discounting (first cycle undiscounted)
    # - 'mid': mid-cycle discounting (half-cycle correction)
    # - 'end': end-of-cycle discounting (first cycle discounted by 1 year when dt=1)
    'discounting_timing': 'end',

    # Caregiver utility table (stage/setting specific, age-invariant approximation) (VERIFIED, Reed et al., 2017)
    # Note: Caregivers only apply for home setting with mild/moderate/severe stages
    'stage_age_qalys': {
        'caregiver': {
            'mild': {
                'home': {0: 0.86},
            },
            'moderate': {
                'home': {0: 0.85},
            },
            'severe': {
                'home': {0: 0.82},
            },
        },
    },
    # QALY weights for dementia stages (fixed per stage or per sex) (VERIFIED, Mukadam et al., 2024)
    'dementia_stage_qalys': {
        'mild': 0.71,
        'moderate': 0.64,
        'severe': 0.38,

    },

    # Annual costs (GBP) by stage/setting  (VERIFIED, Annual costs of dementia)
    'costs': {
        'cognitively_normal': {
            'home': {'nhs': 0, 'informal': 0},
        },
        'mild': {
            'home': {'nhs': 7466.70, 'informal': 10189.55},
            'institution': {'nhs': 23144.27, 'informal': 874.93},
        },
        'moderate': {
            'home': {'nhs': 7180.18, 'informal': 33726.09},
            'institution': {'nhs': 15552.58, 'informal': 1643.14},
        },
        'severe': {
            'home': {'nhs': 7668.60, 'informal': 31523.39},
            'institution': {'nhs': 53084.13, 'informal': 501.88},
        },
    },
}

PSA_DEFAULT_RELATIVE_SD = 0.10

RISK_FACTOR_HR_INTERVALS: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {
    'socioeconomic_disadvantage': {'onset': {'all': (2.01, 1.88, 2.15)}},
    'low_education':            {'onset': {'all': (1.64, 1.52, 1.76)}},
    'hearing_difficulty':       {'onset': {'all': (1.21, 1.15, 1.27)}},
    'hypertension':             {'onset': {'all': (1.36, 1.28, 1.45)}},
    'obesity':                  {'onset': {'all': (1.1, 1.03, 1.18)}},
    'lifestyle':                {'onset': {'all': (1.08, 1.03, 1.2)}},
    'excessive_alcohol_consumption': {'onset': {'all': (1.30, 1.23, 1.37)}},
    'smoking':                  {'onset': {'all': (1.16, 1.10, 1.22)}},
    'depression':               {'onset': {'all': (1.93, 1.74, 2.13)}},
    'social_isolation':         {'onset': {'all': (1.34, 1.28, 1.41)}},
    'diabetes':                 {'onset': {'all': (2.06, 1.92, 2.22)},},
    'air_pollution':            {'onset': {'all': (1.27, 1.19, 1.35)}},
    'APOE_e4_carrier':          {'onset': {'all': (3.02, 2.88, 3.19)}},
    'periodontal_disease':      {'onset': {'all': (1.21, 1.07, 1.38)}},
}


# Constants & seed

DEMENTIA_STAGES = ['cognitively_normal', 'mild', 'moderate', 'severe', 'death']
LIVING_SETTINGS = ['home', 'institution']
random.seed(42)  # reproducibility

# -------- Weighted age samplers --------

def _normalize_weights(d: dict) -> dict:
    """Return a new dict with values normalized to sum to 1.0 (ignores non-positive weights)."""
    positive = {k: float(v) for k, v in d.items() if float(v) > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("All provided weights are zero or negative.")
    return {k: v / total for k, v in positive.items()}

def sample_age_from_weighted_ages(age_weights: Dict[int, float]) -> int:
    """
    Sample an exact age from a {age: weight} mapping.
    Example: {40: 0.2, 50: 0.15, 60: 0.25, ...}
    """
    w = _normalize_weights(age_weights)
    ages = list(w.keys())
    probs = list(w.values())
    # Use random.choices for weighted categorical draw
    return random.choices(population=ages, weights=probs, k=1)[0]

def sample_age_from_band_weights(band_weights: Dict[Tuple[int, int], float]) -> int:
    """
    Sample an age band by weight, then draw a uniform integer age within that band (inclusive).
    Example: {(40, 44): 0.2, (45, 49): 0.15, (50, 54): 0.25, ...}
    """
    w = _normalize_weights(band_weights)
    bands = list(w.keys())
    probs = list(w.values())
    low, high = random.choices(population=bands, weights=probs, k=1)[0]
    return random.randint(low, high)  # uniform within chosen band

# Sex and risk-factor helpers

def age_band_weights_for_year(open_pop_cfg: dict,
                              year: int,
                              baseline_weights: Dict[Tuple[int, int], float]) -> Optional[Dict[Tuple[int, int], float]]:
    """
    Return entrant age-band weights for the provided calendar year by scaling baseline
    weights with any milestone multipliers supplied in the open-population configuration.
    """
    multiplier_schedule = open_pop_cfg.get("age_band_multiplier_schedule")
    if not multiplier_schedule:
        # fall back to legacy fixed weights if provided
        schedule = open_pop_cfg.get("age_band_weights_schedule")
        if schedule:
            milestone_years = sorted(schedule)
            if not milestone_years:
                return open_pop_cfg.get("age_band_weights")
            if year <= milestone_years[0]:
                return schedule[milestone_years[0]]
            if year >= milestone_years[-1]:
                return schedule[milestone_years[-1]]
            for start, end in zip(milestone_years, milestone_years[1:]):
                if start <= year <= end:
                    span = end - start
                    weight = 0.0 if span <= 0 else (year - start) / span
                    bands = set(schedule[start]) | set(schedule[end])
                    return {
                        band: ((1.0 - weight) * schedule[start].get(band, 0.0) +
                               weight * schedule[end].get(band, 0.0))
                        for band in bands
                    }
        return open_pop_cfg.get("age_band_weights")

    milestone_years = sorted(multiplier_schedule)
    if not milestone_years:
        return baseline_weights

    if year <= milestone_years[0]:
        multipliers = multiplier_schedule[milestone_years[0]]
    elif year >= milestone_years[-1]:
        multipliers = multiplier_schedule[milestone_years[-1]]
    else:
        multipliers = None
        for start, end in zip(milestone_years, milestone_years[1:]):
            if start <= year <= end:
                span = end - start
                frac = 0.0 if span <= 0 else (year - start) / span
                bands = set(multiplier_schedule[start]) | set(multiplier_schedule[end])
                multipliers = {
                    band: ((1.0 - frac) * multiplier_schedule[start].get(band, 1.0) +
                           frac * multiplier_schedule[end].get(band, 1.0))
                    for band in bands
                }
                break
        if multipliers is None:
            multipliers = multiplier_schedule.get(year, {})

    scaled = {
        band: baseline_weights.get(band, 0.0) * multipliers.get(band, 1.0)
        for band in baseline_weights
    }
    total = sum(scaled.values())
    if total <= 0:
        return baseline_weights
    return {band: value / total for band, value in scaled.items()}

def background_mortality_scalar_for_year(config: dict, year: int) -> float:
    """
    Return background mortality scalar for a given year.
    If a schedule is provided, use it with linear interpolation between milestones.
    If no schedule is provided, apply proportional growth (if enabled).
    Falls back to 'background_mortality_scalar' or 1.0.
    """
    per_year = config.get("background_mortality_scalar_by_year") or {}
    if per_year:
        if year in per_year:
            return float(per_year[year])
        year_key = str(year)
        if year_key in per_year:
            return float(per_year[year_key])

    schedule = config.get("background_mortality_scalar_schedule") or {}
    if not schedule:
        growth_cfg = config.get("background_mortality_scalar_growth") or {}
        if growth_cfg.get("use", False):
            try:
                annual_rate = float(growth_cfg.get("annual_rate", 0.0))
            except (TypeError, ValueError):
                annual_rate = 0.0
            try:
                reference_year = int(growth_cfg.get("reference_year", year))
            except (TypeError, ValueError):
                reference_year = year
            base_scalar = float(config.get("background_mortality_scalar", 1.0) or 1.0)
            if annual_rate > -1.0:
                years_since_ref = year - reference_year
                return base_scalar * ((1.0 + annual_rate) ** years_since_ref)
        return float(config.get("background_mortality_scalar", 1.0) or 1.0)

    milestone_years = sorted(schedule)
    if not milestone_years:
        return float(config.get("background_mortality_scalar", 1.0) or 1.0)

    if year <= milestone_years[0]:
        return float(schedule[milestone_years[0]])
    if year >= milestone_years[-1]:
        return float(schedule[milestone_years[-1]])

    for start, end in zip(milestone_years, milestone_years[1:]):
        if start <= year <= end:
            span = end - start
            frac = 0.0 if span <= 0 else (year - start) / span
            start_val = float(schedule[start])
            end_val = float(schedule[end])
            return (1.0 - frac) * start_val + frac * end_val

    return float(config.get("background_mortality_scalar", 1.0) or 1.0)


def _canonicalize_sex(sex: str) -> str:
    """Normalize sex labels to the model's canonical keys."""
    return _canonical_sex_label(sex) if isinstance(sex, str) else "unspecified"


def _qx_to_hazard(qx: float) -> float:
    """Convert annual death probability qx to hazard, with guarding."""
    try:
        qx = float(qx)
    except (TypeError, ValueError):
        return 0.0
    qx = max(0.0, min(0.9999999999, qx))
    return -math.log(1.0 - qx)


def ensure_background_mortality_hazards_by_year(config: dict) -> None:
    """
    Populate config['background_mortality_hazards_by_year'] if qx values were pasted.
    """
    hazards_by_year = config.get('background_mortality_hazards_by_year') or {}
    if hazards_by_year:
        # Optional: convert if the user pasted qx instead of hazards
        if config.get('background_mortality_hazards_by_year_is_qx'):
            converted: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
            for year_key, sex_map in hazards_by_year.items():
                try:
                    year_int = int(year_key)
                except (TypeError, ValueError):
                    year_int = year_key
                if not isinstance(sex_map, dict):
                    continue
                for sex_key, age_map in sex_map.items():
                    if not isinstance(age_map, dict):
                        continue
                    for age_key, qx_val in age_map.items():
                        try:
                            age_int = int(age_key)
                        except (TypeError, ValueError):
                            age_int = age_key
                        converted[year_int][_canonicalize_sex(sex_key)][age_int] = _qx_to_hazard(qx_val)
            config['background_mortality_hazards_by_year'] = converted
        return

def add_new_entrants(population_state: Dict[int, dict],
                     config: dict,
                     next_id_start: int,
                     calendar_year: int) -> Tuple[int, int]:
    """Optionally add new individuals at the *start* of this timestep.

    Returns the next unused ID and the number of entrants added."""
    op = config.get("open_population", {}) or {}
    if not op.get("use", False):
        return next_id_start, 0

    base_entrants = float(op.get("entrants_per_year", 0) or 0)
    growth_cfg = op.get("entrants_growth", {}) or {}
    n_new = base_entrants
    if growth_cfg.get("use", False):
        try:
            annual_rate = float(growth_cfg.get("annual_rate", 0.0))
        except (TypeError, ValueError):
            annual_rate = 0.0
        try:
            reference_year = int(growth_cfg.get("reference_year", calendar_year))
        except (TypeError, ValueError):
            reference_year = calendar_year
        if annual_rate > -1.0:
            years_since_ref = calendar_year - reference_year
            n_new = base_entrants * ((1.0 + annual_rate) ** years_since_ref)
    n_new = int(round(n_new))
    if n_new <= 0:
        return next_id_start, 0

    # fall back to global config if open-pop overrides are not provided
    baseline_weights = config.get("initial_age_band_weights", {})
    age_band_weights = age_band_weights_for_year(op, calendar_year, baseline_weights) or baseline_weights
    sex_dist = op.get("sex_distribution") or config.get("sex_distribution", {})
    base_year = int(config.get('base_year', calendar_year))
    fixed_entry_age = op.get("fixed_entry_age")
    age_sampling_config = {
        "initial_age_band_weights": age_band_weights,
        "initial_age_range": config.get("initial_age_range", (35, 100)),
    }

    entrants_added = 0
    for j in range(n_new):
        sex = sample_sex(sex_dist)
        if fixed_entry_age is not None:
            age = int(fixed_entry_age)
        else:
            age = sample_age(age_sampling_config, sex)
        population_state[next_id_start] = {
            'ID': next_id_start,
            'age': age,
            'sex': sex,
            'risk_factors': assign_risk_factors(config['risk_factors'], age, sex, calendar_year),
            'dementia_stage': 'cognitively_normal',
            'time_in_stage': 0,
            'living_setting': 'home',
            'alive': True,
            'cumulative_qalys_patient': 0.0,
            'cumulative_qalys_caregiver': 0.0,
            'cumulative_costs_nhs': 0.0,
            'cumulative_costs_informal': 0.0,
            'calendar_year': calendar_year,
            'baseline_stage': 'cognitively_normal',
            'entry_age': age,
            'entry_time_step': max(0, calendar_year - base_year),
            'time_since_entry': 0.0,
            'ever_dementia': False,
            'age_at_onset': None,
        }
        set_stage_and_countdown(population_state[next_id_start], 'cognitively_normal', config)
        next_id_start += 1
        entrants_added += 1

    return next_id_start, entrants_added

def _canonical_sex_label(sex: Optional[str]) -> str:
    """Map free-text sex labels onto a small canonical vocabulary."""
    if sex is None:
        return 'unspecified'
    label = str(sex).strip().lower()
    if not label:
        return 'unspecified'
    if label in {'f', 'female', 'woman', 'women'}:
        return 'female'
    if label in {'m', 'male', 'man', 'men'}:
        return 'male'
    if label in {'all', 'any', 'either', 'both'}:
        return 'all'
    return label

def sample_sex(sex_distribution: Dict[str, float]) -> str:
    """Sample sex from a weight dictionary; defaults to 'unspecified' if absent."""
    if not sex_distribution:
        return 'unspecified'
    normalized = {_canonical_sex_label(k): float(v) for k, v in sex_distribution.items()}
    weights = _normalize_weights(normalized)
    labels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(population=labels, weights=probs, k=1)[0]

def get_stage_mix_for_sex(stage_mix_config: Optional[dict],
                          sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return the stage-mix weights to use for an individual of the provided sex."""
    if not stage_mix_config or not isinstance(stage_mix_config, dict):
        return None

    # Legacy support: already a mapping of stage -> weight
    if all(isinstance(v, (int, float)) for v in stage_mix_config.values()):
        return stage_mix_config

    canonical_sex = _canonical_sex_label(sex)
    for key, mix in stage_mix_config.items():
        if _canonical_sex_label(key) == canonical_sex and isinstance(mix, dict):
            return mix

    for fallback in ('all', 'any', 'either', 'both', 'default'):
        for key, mix in stage_mix_config.items():
            if (_canonical_sex_label(key) == fallback or key == fallback) and isinstance(mix, dict):
                return mix

    # Final fallback: return the first nested dictionary if present
    for mix in stage_mix_config.values():
        if isinstance(mix, dict):
            return mix

    return None


def get_stage_mix_for_age_and_sex(stage_mix_by_age: Optional[dict],
                                  age: float,
                                  sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return an age-conditional stage mix; falls back to sex/default keys."""
    if not stage_mix_by_age or not isinstance(stage_mix_by_age, dict):
        return None

    bands: List[Tuple[int, Optional[int]]] = []
    for key in stage_mix_by_age.keys():
        if isinstance(key, tuple) and len(key) == 2:
            lower, upper = key
            if isinstance(lower, (int, float)) and (upper is None or isinstance(upper, (int, float))):
                bands.append((int(lower), None if upper is None else int(upper)))
    bands.sort(key=lambda b: b[0])

    candidate = None
    if bands:
        chosen_band = assign_age_to_reporting_band(age, bands)
        if chosen_band is not None:
            candidate = stage_mix_by_age.get(chosen_band)

    if candidate is None:
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            if fallback in stage_mix_by_age:
                candidate = stage_mix_by_age[fallback]
                break

    if candidate is None:
        return None

    if isinstance(candidate, dict):
        return get_stage_mix_for_sex(candidate, sex)

    return None


def get_dementia_stage_weights_for_sex(stage_mix_config: Optional[dict],
                                       sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return normalized dementia-stage weights (excluding cognitively normal) for a given sex."""
    mix = get_stage_mix_for_sex(stage_mix_config, sex)
    if not mix or not isinstance(mix, dict):
        return None
    dementia_weights = {
        stage: weight for stage, weight in mix.items()
        if stage in {'mild', 'moderate', 'severe'}
    }
    if not dementia_weights:
        return None
    try:
        return _normalize_weights(dementia_weights)
    except ValueError:
        return None


def get_dementia_prevalence_for_age_and_sex(prevalence_config: Optional[dict],
                                            age: float,
                                            sex: Optional[str]) -> Optional[float]:
    """Fetch dementia prevalence for the given age/sex from banded configuration."""
    if not prevalence_config or not isinstance(prevalence_config, dict):
        return None

    bands: List[Tuple[int, Optional[int]]] = []
    for key in prevalence_config.keys():
        if isinstance(key, tuple) and len(key) == 2:
            lower, upper = key
            if isinstance(lower, (int, float)) and (upper is None or isinstance(upper, (int, float))):
                bands.append((int(lower), None if upper is None else int(upper)))
    bands.sort(key=lambda b: b[0])

    value = None
    if bands:
        band = assign_age_to_reporting_band(age, bands)
        if band is not None:
            value = prevalence_config.get(band)

    if value is None:
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            if fallback in prevalence_config:
                value = prevalence_config[fallback]
                break

    if isinstance(value, dict):
        sex_label = _canonical_sex_label(sex)
        for key, val in value.items():
            if _canonical_sex_label(key) == sex_label:
                try:
                    prevalence = float(val)
                except (TypeError, ValueError):
                    return None
                return max(0.0, min(1.0, prevalence))
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            for key, val in value.items():
                if _canonical_sex_label(key) == fallback or key == fallback:
                    try:
                        prevalence = float(val)
                    except (TypeError, ValueError):
                        return None
                    return max(0.0, min(1.0, prevalence))
        return None

    if value is None:
        return None

    try:
        prevalence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, prevalence))

def sample_stage_from_mix(stage_mix: Optional[Dict[str, float]],
                          default_stage: str = 'cognitively_normal') -> str:
    """Sample a dementia stage from weighted mix; fall back to default if weights invalid."""
    if not stage_mix:
        return default_stage
    try:
        weights = _normalize_weights(stage_mix)
    except ValueError:
        return default_stage

    stages = list(weights.keys())
    probs = list(weights.values())
    return random.choices(population=stages, weights=probs, k=1)[0]

def resolve_risk_value(value: Any,
                       age: Optional[int],
                       sex: Optional[str]) -> Any:
    """Recursively resolve nested risk metadata keyed by sex and/or age (scalars allowed)."""
    if isinstance(value, dict):
        sex_keys = [k for k in value.keys() if isinstance(k, str)]
        if sex_keys:
            if sex is not None:
                target = _canonical_sex_label(sex)
                for key in sex_keys:
                    if _canonical_sex_label(key) == target:
                        return resolve_risk_value(value[key], age, None)
            for fallback in ('all', 'any', 'either', 'both', 'default'):
                for key in sex_keys:
                    if _canonical_sex_label(key) == fallback or key == fallback:
                        return resolve_risk_value(value[key], age, None)

        band_items = [(key, nested) for key, nested in value.items()
                      if isinstance(key, tuple) and len(key) == 2]
        if band_items:
            if age is not None:
                for band, nested in band_items:
                    lo, hi = band
                    if lo <= age <= hi:
                        return resolve_risk_value(nested, age, None)
                # fallback: choose band with midpoint closest to age
                closest_nested = min(
                    band_items,
                    key=lambda item: abs((item[0][0] + item[0][1]) / 2.0 - age)
                )[1]
                return resolve_risk_value(closest_nested, age, None)
            return resolve_risk_value(band_items[0][1], age, None)

        numeric_items = [(key, nested) for key, nested in value.items()
                         if isinstance(key, (int, float))]
        if numeric_items:
            numeric_items.sort(key=lambda item: item[0])
            if age is not None:
                chosen = numeric_items[0][1]
                for threshold, nested in numeric_items:
                    if age >= threshold:
                        chosen = nested
                    else:
                        break
                return resolve_risk_value(chosen, age, None)
            return resolve_risk_value(numeric_items[0][1], age, None)

        if 'default' in value:
            return resolve_risk_value(value['default'], age, None)
        if 'all' in value:
            return resolve_risk_value(value['all'], age, None)
    return value

def get_prevalence_for_person(risk_meta: dict, age: int, sex: str, calendar_year: Optional[int] = None) -> float:
    """Fetch prevalence for a given age/sex combination (age ignored if scalar); clamps to [0, 1].

    If risk_meta contains a 'prevalence_schedule' with 'use': True, calculates time-varying prevalence.
    Otherwise falls back to static prevalence.
    """
    # Check if time-varying prevalence is enabled
    schedule = risk_meta.get('prevalence_schedule', {})
    if schedule.get('use', False) and calendar_year is not None:
        baseline_year = int(schedule.get('baseline_year', 2023))
        target_year = int(schedule.get('target_year', 2040))
        baseline_prev = schedule.get('baseline_prevalence', {})
        target_prev = schedule.get('target_prevalence', {})

        # Get baseline and target for this sex
        baseline = resolve_risk_value(baseline_prev, age, sex)
        target = resolve_risk_value(target_prev, age, sex)

        try:
            baseline = float(baseline)
            target = float(target)
        except (TypeError, ValueError):
            baseline = 0.5
            target = 0.6125

        # Linear interpolation
        if calendar_year <= baseline_year:
            prevalence = baseline
        elif calendar_year >= target_year:
            prevalence = target
        else:
            # Interpolate between baseline and target
            progress = (calendar_year - baseline_year) / (target_year - baseline_year)
            prevalence = baseline + progress * (target - baseline)
    else:
        # Use static prevalence
        raw = resolve_risk_value(risk_meta.get('prevalence', 0.0), age, sex)
        try:
            prevalence = float(raw)
        except (TypeError, ValueError):
            prevalence = 0.0

    return max(0.0, min(1.0, prevalence))

def get_hazard_ratio_for_person(risk_meta: dict,
                                 transition: str,
                                 age: int,
                                 sex: str) -> float:
    """Return the hazard ratio for the given transition, age, and sex (age ignored if scalar)."""
    rr_spec = risk_meta.get('hazard_ratios', {})
    transition_spec = rr_spec.get(transition, rr_spec.get('default', 1.0))
    raw = resolve_risk_value(transition_spec, age, sex)
    try:
        rr = float(raw)
    except (TypeError, ValueError):
        rr = 1.0
    return max(rr, 0.0)

# Hazard helpers

def prob_to_hazard(p: float, dt: float = 1.0) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float('inf')
    return -math.log(1.0 - p) / dt

def hazard_to_prob(h: float, dt: float = 1.0) -> float:
    if h <= 0.0:
        return 0.0
    return 1.0 - math.exp(-h * dt)

def base_hazard_from_duration(duration_years: float) -> float:
    if duration_years is None or duration_years <= 0.0:
        return 0.0
    return 1.0 / duration_years

def hazards_from_survival(times: List[float],
                          survival_probs: List[float]) -> pd.DataFrame:
    """
    Infer piecewise-constant hazards from survival probabilities at multiple time points.

    Parameters
    ----------
    times:
        Sequence of time points (years). Does not need to start at zero but must be strictly increasing.
    survival_probs:
        Survival probabilities corresponding to ``times`` (same length, positive).

    Returns
    -------
    DataFrame with one row per interval containing the start/end time, survival levels, interval length,
    implied hazard, and the deviation from the mean hazard across all intervals.
    """
    if len(times) != len(survival_probs):
        raise ValueError("times and survival_probs must have the same length.")
    if len(times) < 2:
        raise ValueError("At least two time points are required to infer hazards.")

    t = np.asarray(times, dtype=float)
    s = np.asarray(survival_probs, dtype=float)

    order = np.argsort(t)
    t = t[order]
    s = s[order]

    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(s)):
        raise ValueError("times and survival_probs must contain finite values only.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing.")
    if (s <= 0).any() or (s > 1).any():
        raise ValueError("survival probabilities must be in the interval (0, 1].")

    interval_rows: List[Dict[str, float]] = []
    prev_t = t[0]
    prev_s = s[0]
    for curr_t, curr_s in zip(t[1:], s[1:]):
        dt = curr_t - prev_t
        if dt <= 0:
            raise ValueError("Encountered non-positive interval length when computing hazards.")
        if curr_s <= 0 or prev_s <= 0:
            raise ValueError("Survival probabilities must stay strictly positive.")

        # interval hazard implied by exponential survival over the interval
        hazard = -math.log(curr_s / prev_s) / dt
        interval_rows.append({
            'time_start': float(prev_t),
            'time_end': float(curr_t),
            'interval_length': float(dt),
            'survival_start': float(prev_s),
            'survival_end': float(curr_s),
            'interval_hazard': float(hazard),
        })
        prev_t = curr_t
        prev_s = curr_s

    interval_df = pd.DataFrame(interval_rows)
    if interval_df.empty:
        return interval_df

    mean_hazard = interval_df['interval_hazard'].mean()
    if mean_hazard <= 0:
        interval_df['relative_deviation'] = np.nan
    else:
        interval_df['relative_deviation'] = (
            interval_df['interval_hazard'] - mean_hazard
        ) / mean_hazard
    interval_df['mean_hazard'] = mean_hazard
    return interval_df

def check_constant_hazard(times: List[float],
                          survival_probs: List[float],
                          tolerance: float = 0.05) -> Dict[str, Union[bool, float, pd.DataFrame]]:
    """
    Assess the constant-hazard assumption using survival probabilities across time.

    The function computes piecewise hazards using :func:`hazards_from_survival` and compares each
    interval's implied hazard against the mean hazard. Deviations larger than ``tolerance`` (relative)
    indicate that the exponential/constant-hazard assumption may not hold over the provided horizon.

    Parameters
    ----------
    times:
        Sequence of time points (years) for the survival probabilities.
    survival_probs:
        Survival probabilities observed at ``times``.
    tolerance:
        Maximum allowed relative deviation (default 5%). Set to a higher value if measurement noise is large.

    Returns
    -------
    Dictionary with the inferred mean hazard, the maximum absolute relative deviation, a boolean flag signalling
    whether the assumption holds within tolerance, and the interval DataFrame for further inspection.
    """
    interval_df = hazards_from_survival(times, survival_probs)
    if interval_df.empty:
        return {
            'mean_hazard': float('nan'),
            'max_relative_deviation': float('nan'),
            'within_tolerance': False,
            'intervals': interval_df,
        }

    rel_dev = interval_df['relative_deviation'].abs().replace([np.inf, -np.inf], np.nan).dropna()
    max_dev = rel_dev.max() if not rel_dev.empty else 0.0
    mean_hazard = float(interval_df['mean_hazard'].iloc[0])
    within = bool(max_dev <= tolerance) if np.isfinite(max_dev) else False
    return {
        'mean_hazard': mean_hazard,
        'max_relative_deviation': float(max_dev) if np.isfinite(max_dev) else float('nan'),
        'within_tolerance': within,
        'intervals': interval_df,
    }

def check_constant_hazard_from_model(model_results: dict,
                                     tolerance: float = 0.05,
                                     cohort: str = 'baseline',
                                     use_calendar_year: bool = False) -> Dict[str, Union[bool, float, pd.DataFrame]]:
    """
    Convenience wrapper that pulls survival data from ``model_results`` and runs ``check_constant_hazard``.

    Parameters
    ----------
    model_results:
        Output dictionary returned by :func:`run_model`.
    tolerance:
        Maximum allowed relative deviation (default 5%).
    cohort:
        Either ``'baseline'`` (default) to follow the initial cohort only, or ``'population'`` to include entrants.
    use_calendar_year:
        If ``True`` and calendar years are available, use them as the time axis instead of ``time_step``.
    """
    df = summaries_to_dataframe(model_results)
    if df.empty:
        raise ValueError("Model results do not contain summary history.")

    if cohort == 'population':
        series_name = 'population_alive'
        cohort_key = 'population'
    else:
        series_name = 'baseline_alive'
        cohort_key = 'baseline'

    if series_name not in df.columns:
        raise ValueError(f"Summary dataframe does not include '{series_name}'.")

    alive = df[series_name].fillna(0.0).to_numpy(dtype=float)
    if alive.size == 0 or alive[0] <= 0.0:
        raise ValueError(f"No positive counts found for '{series_name}'.")

    survival = alive / alive[0]
    valid_mask = survival > 0
    if valid_mask.sum() < 2:
        raise ValueError("Need at least two strictly positive survival points to assess hazards.")

    time_key = 'calendar_year' if use_calendar_year and 'calendar_year' in df.columns else 'time_step'
    times = df[time_key].to_numpy(dtype=float)

    times = times[valid_mask]
    survival = survival[valid_mask]

    check = check_constant_hazard(times.tolist(), survival.tolist(), tolerance=tolerance)
    check['cohort'] = cohort_key
    check['time_axis'] = time_key
    return check

def _value_from_age_table(age: float, table: Dict[Union[int, float], float]) -> float:
    thresholds = sorted(table)
    eligible = [a for a in thresholds if a <= age]
    key = eligible[-1] if eligible else thresholds[0]
    return float(table[key])

def get_age_specific_utility(age: float,
                             utility_norms: Union[Dict[Union[int, float], float], Dict[str, Any]],
                             sex: Optional[str] = None) -> float:
    """
    Return age-specific utility. Supports either a flat age->utility map or nested dict keyed by sex.
    Falls back to the first available mapping if sex-specific entry is missing.
    """
    if not utility_norms:
        return 0.0

    if isinstance(utility_norms, dict):
        # Flat age table (legacy)
        if all(isinstance(k, (int, float)) for k in utility_norms.keys()):
            return _value_from_age_table(age, utility_norms)

        sex_key = (sex or '').strip().lower()
        if sex_key and sex_key in utility_norms:
            sex_table = utility_norms[sex_key]
            if isinstance(sex_table, dict) and sex_table:
                return _value_from_age_table(age, sex_table)

        # Try generic 'all' entry
        if 'all' in utility_norms:
            sex_table = utility_norms['all']
            if isinstance(sex_table, dict) and sex_table:
                return _value_from_age_table(age, sex_table)

        # Fallback to first dict-like value
        for value in utility_norms.values():
            if isinstance(value, dict) and value:
                return _value_from_age_table(age, value)

        # Fallback to scalar
        try:
            return float(next(iter(utility_norms.values())))
        except (TypeError, StopIteration, ValueError):
            return 0.0

    # Scalar fallback
    try:
        return float(utility_norms)
    except (TypeError, ValueError):
        return 0.0

def get_stage_age_qaly(subject: str,
                       stage: str,
                       age: float,
                       setting: Optional[str],
                       stage_age_config: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Return a direct utility weight for the given subject/stage/age/setting if specified.
    Accepts either a direct age-threshold map (e.g. {65: 0.7}) or nested maps keyed by setting
    with optional 'default' fallback. Returns None if no override is configured.
    """
    if not stage_age_config:
        return None
    subject_data = stage_age_config.get(subject)
    if not isinstance(subject_data, dict):
        return None
    stage_data = subject_data.get(stage)
    if not isinstance(stage_data, dict) or not stage_data:
        return None

    def _as_age_map(candidate: Any) -> Optional[Dict[Union[int, float], float]]:
        if not isinstance(candidate, dict) or not candidate:
            return None
        if all(isinstance(k, (int, float)) for k in candidate.keys()):
            return candidate  # direct age map
        return None

    # direct age map at stage level
    age_map = _as_age_map(stage_data)
    if age_map is None:
        selected: Optional[Dict[Union[int, float], float]] = None
        if setting and setting in stage_data:
            selected = _as_age_map(stage_data[setting])
        if selected is None and 'default' in stage_data:
            selected = _as_age_map(stage_data['default'])
        if selected is None:
            # if only one nested dict exists, assume it is the intended age map
            nested_maps = [
                _as_age_map(v) for v in stage_data.values()
                if isinstance(v, dict)
            ]
            nested_maps = [m for m in nested_maps if m is not None]
            if len(nested_maps) == 1:
                selected = nested_maps[0]
        age_map = selected

    if not age_map:
        return None
    return get_age_specific_utility(age, age_map)

def _normalise_sex(sex: Optional[str]) -> str:
    return str(sex or '').strip().lower()

def get_qaly_by_age_and_sex(age: float,
                            sex: Optional[str],
                            config: dict) -> float:
    """Return healthy QALY weight using utility_norms_by_age (per sex if provided)."""
    utility_norms = config.get('utility_norms_by_age')
    value = get_age_specific_utility(age, utility_norms, sex)
    return value if value else 0.0

def get_dementia_stage_qaly(stage: str,
                            sex: Optional[str],
                            config: dict) -> Optional[float]:
    """Return dementia-stage QALY weight, supporting optional sex-specific entries."""
    stage_map = config.get('dementia_stage_qalys')
    if not isinstance(stage_map, dict) or not stage_map:
        return None

    # direct stage match
    if stage in stage_map and not isinstance(stage_map[stage], dict):
        try:
            return float(stage_map[stage])
        except (TypeError, ValueError):
            return None

    sex_key = _normalise_sex(sex)
    if stage in stage_map and isinstance(stage_map[stage], dict):
        entry = stage_map[stage]
        if sex_key and sex_key in entry:
            try:
                return float(entry[sex_key])
            except (TypeError, ValueError):
                return None
        for fallback_key in ('all', 'default'):
            if fallback_key in entry:
                try:
                    return float(entry[fallback_key])
                except (TypeError, ValueError):
                    return None
        for value in entry.values():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    # sex-first structure: {'female': {'mild': 0.7, ...}}
    if sex_key and sex_key in stage_map:
        sex_entry = stage_map[sex_key]
        if isinstance(sex_entry, dict) and stage in sex_entry:
            try:
                return float(sex_entry[stage])
            except (TypeError, ValueError):
                return None
        default_val = sex_entry.get('default') if isinstance(sex_entry, dict) else None
        if default_val is not None:
            try:
                return float(default_val)
            except (TypeError, ValueError):
                return None

    # generic fallback
    for fallback_key in ('all', 'default'):
        if fallback_key in stage_map:
            entry = stage_map[fallback_key]
            if isinstance(entry, dict) and stage in entry:
                try:
                    return float(entry[stage])
                except (TypeError, ValueError):
                    return None
            try:
                return float(entry)
            except (TypeError, ValueError):
                continue
    return None

def get_caregiver_qaly(age: float,
                       sex: Optional[str],
                       config: dict) -> Optional[float]:
    """Return caregiver QALY weight if caregiver tables are provided.
    Note: No caregiver-specific age tables are currently defined in config.
    This function exists for extensibility but returns None in current model."""
    return None

# NEW: unified age HR getter (parametric Cox-style)
def get_age_hr_for_transition(age: int, config: dict, transition: str) -> float:
    """Return age hazard ratio using exp(beta*(age-ref_age)) (optionally two-piece)."""
    param = config.get('age_hr_parametric', {})
    if param.get('use', False):
        ref_age = param.get('ref_age', 65)
        betas = param.get('betas', {}) or {}
        base_beta = betas.get(transition, 0.0)  # default: no age effect if missing
        piecewise = (param.get('two_piece') or {}).get(transition)
        if piecewise and 'break_age' in piecewise:
            break_age = int(piecewise['break_age'])
            beta_before = piecewise.get('beta_before', base_beta)
            beta_after = piecewise.get('beta_after', beta_before)
            if age <= break_age:
                return math.exp(beta_before * (age - ref_age))
            # ensure continuity at the breakpoint so the curve is capped smoothly
            ratio_at_break = math.exp(beta_before * (break_age - ref_age))
            return ratio_at_break * math.exp(beta_after * (age - break_age))
        return math.exp(base_beta * (age - ref_age))
    return 1.0  # fallback: no age effect

def apply_hazard_ratios(h_base: float,
                        risk_factors: Dict[str, bool],
                        risk_defs: Dict[str, dict],
                        transition_key: str,
                        age_hr: float,
                        age: int,
                        sex: str,
                        config: dict) -> float:
    h = h_base * age_hr
    for factor, active in risk_factors.items():
        if not active:
            continue
        rr = get_hazard_ratio_for_person(risk_defs.get(factor, {}), transition_key, age, sex)
        h *= rr
    return h

def transition_hazard_from_config(config: dict, person: dict, transition_key: str) -> float:
    """Return adjusted hazard for a named transition (duration -> h0, then multiply HRs)."""
    duration = config['stage_transition_durations'].get(transition_key)
    h0 = base_hazard_from_duration(duration)
    age_hr = get_age_hr_for_transition(person['age'], config, transition_key)  # CHANGED: parametric or banded
    return apply_hazard_ratios(
        h0,
        person['risk_factors'],
        config['risk_factors'],
        transition_key,
        age_hr,
        person['age'],
        person.get('sex', 'unspecified'),
        config,  # NEW
    )


# --- Deterministic countdown support --------------------------------------

def _stage_duration_steps(stage: str, config: dict) -> Optional[int]:
    """Return the configured number of time steps to stay in a stage (None/<=0 => no countdown)."""
    mapping = {
        'mild': 'mild_to_moderate',
        'moderate': 'moderate_to_severe',
        'severe': 'severe_to_death',
    }
    key = mapping.get(stage)
    if key is None:
        return None
    try:
        steps = config['stage_transition_durations'].get(key)
    except Exception:
        return None
    if steps is None:
        return None
    try:
        steps_int = int(round(float(steps)))
    except (TypeError, ValueError):
        return None
    return steps_int if steps_int > 0 else None


def set_stage_and_countdown(person: dict, stage: str, config: dict) -> None:
    """Set dementia stage and initialise the per-person deterministic countdown."""
    person['dementia_stage'] = stage
    person['time_in_stage'] = 0
    # Revert to hazard-driven progression (no deterministic countdown)
    person['remaining_stage_steps'] = None

def transition_prob_from_config(config: dict, person: dict, transition_key: str) -> float:
    """Convenience: hazard -> per-cycle probability."""
    h = transition_hazard_from_config(config, person, transition_key)
    return min(1.0, hazard_to_prob(h, dt=config['time_step_years']))

# Background mortality helpers

def get_background_mortality_hazard(age: int, hazard_table: Dict[int, float]) -> float:
    """Pick the hazard for the closest band <= age (falls back to smallest band if below)."""
    if not hazard_table:
        return 0.0
    thresholds = sorted(hazard_table)
    eligible = [a for a in thresholds if a <= age]
    key = eligible[-1] if eligible else thresholds[0]
    return hazard_table[key]

def get_dementia_mortality_multiplier(stage: str, mults: Dict[str, float]) -> float:
    """Multiply background hazard by a stage-specific factor (optional)."""
    if not mults:
        return 1.0
    return mults.get(stage, 1.0)

# Model storage helpers

def initialize_model_dictionary() -> Dict[int, dict]:
    """Container for per-timestep summary statistics."""
    return {}

def create_time_step_dictionary(model_dictionary: Dict[int, dict], time_step: int = 0, summary: Optional[dict] = None) -> None:
    """Store a per-timestep summary snapshot."""
    model_dictionary[time_step] = summary or {}

# Population init

def sample_age(config: dict, sex: Optional[str] = None) -> int:
    """
    Choose an age according to config:
      - If 'initial_age_weights' is provided: sample exact age by weight.
      - Else if 'population_age_band_weights_by_sex' is provided: sample band by sex-specific weights.
      - Else if 'initial_age_band_weights' is provided: sample within a band by weight.
      - Else fall back to uniform 'initial_age_range'.
    """
    if 'initial_age_weights' in config and config['initial_age_weights']:
        return sample_age_from_weighted_ages(config['initial_age_weights'])

    if sex is not None and 'population_age_band_weights_by_sex' in config:
        by_sex = config['population_age_band_weights_by_sex'] or {}
        weights = by_sex.get(sex) or by_sex.get(_canonical_sex_label(sex))
        if weights:
            return sample_age_from_band_weights(weights)

    if 'initial_age_band_weights' in config and config['initial_age_band_weights']:
        return sample_age_from_band_weights(config['initial_age_band_weights'])

    # fallback: uniform range
    lo, hi = config['initial_age_range']
    return random.randint(lo, hi)

def assign_risk_factors(risk_factors: Dict[str, dict], age: int, sex: str, calendar_year: Optional[int] = None) -> Dict[str, bool]:
    """Assign risk factors based on prevalence rates, optionally time-varying.

    Parameters:
    -----------
    risk_factors : dict
        Risk factor configuration dictionary
    age : int
        Person's age
    sex : str
        Person's sex ('female' or 'male')
    calendar_year : int, optional
        Calendar year for time-varying prevalence (if applicable)

    Returns:
    --------
    dict : Risk factor name -> boolean status
    """
    assigned = {}
    for rf, meta in risk_factors.items():
        prevalence = get_prevalence_for_person(meta, age, sex, calendar_year)
        assigned[rf] = random.random() < prevalence
    return assigned

def initialize_population(population: int,
                          config: dict) -> Tuple[Dict[int, dict], Counter]:
    base_year = int(config.get('base_year', 2023))
    stage_mix_config = config.get('initial_stage_mix', None)

    population_state: Dict[int, dict] = {}
    age_counter: Counter = Counter()

    for individual in range(population):
        sex = sample_sex(config.get('sex_distribution', {}))
        age = sample_age(config, sex)

        stage0: Optional[str] = None

        prevalence = get_dementia_prevalence_for_age_and_sex(
            config.get('initial_dementia_prevalence_by_age_band'),
            age,
            sex
        )
        if prevalence is not None:
            if random.random() < prevalence:
                dementia_stage_weights = get_dementia_stage_weights_for_sex(stage_mix_config, sex)
                if dementia_stage_weights:
                    stage0 = sample_stage_from_mix(dementia_stage_weights, default_stage='mild')
                else:
                    stage0 = 'mild'
            else:
                stage0 = 'cognitively_normal'

        if stage0 is None:
            stage_weights = get_stage_mix_for_sex(stage_mix_config, sex)
            stage0 = sample_stage_from_mix(stage_weights)

        population_state[individual] = {
            'ID': individual,
            'age': age,
            'sex': sex,
            'risk_factors': assign_risk_factors(config['risk_factors'], age, sex, base_year),
            'dementia_stage': stage0,
            'time_in_stage': 0,
            'living_setting': 'home',
            'alive': True,
            'cumulative_qalys_patient': 0.0,
            'cumulative_qalys_caregiver': 0.0,
            'cumulative_costs_nhs': 0.0,
            'cumulative_costs_informal': 0.0,
            'calendar_year': base_year,
            'baseline_stage': stage0,
            'entry_age': age,
            'entry_time_step': 0,
            'time_since_entry': 0.0,
            'ever_dementia': stage0 in ('mild', 'moderate', 'severe'),
            'age_at_onset': age if stage0 in ('mild', 'moderate', 'severe') else None,
        }
        # Initialise deterministic countdown for those already in dementia stages
        set_stage_and_countdown(population_state[individual], stage0, config)
        age_counter[age] += 1

    return population_state, age_counter

def advance_population_state(population_state: Dict[int, dict],
                             config: dict,
                             calendar_year: int) -> None:
    """Increment age/time_in_stage for alive individuals and roll calendar year."""
    dt = config['time_step_years']
    for person in population_state.values():
        person['calendar_year'] = calendar_year
        if person['alive']:
            person['age'] += dt
            person['time_in_stage'] += dt
            person['time_since_entry'] = person.get('time_since_entry', 0.0) + dt

# Accumulation (QALYs/costs)

def _clamp_utility(value: Any) -> float:
    """Clamp an absolute utility weight to [0, 1] with safe fallbacks."""
    try:
        u = float(value)
    except (TypeError, ValueError):
        return 0.0
    if u < 0.0:
        return 0.0
    if u > 1.0:
        return 1.0
    return u

def _discount_factor(time_step: int, config: dict) -> float:
    """Return the per-cycle discount factor according to config['discounting_timing']."""
    r = float(config.get('discount_rate_annual', 0.0) or 0.0)
    if r <= 0.0:
        return 1.0

    dt = float(config.get('time_step_years', 1.0) or 1.0)
    timing = str(config.get('discounting_timing', 'end') or 'end').strip().lower()
    if timing == 'start':
        exponent = max(0.0, (time_step - 1) * dt)
    elif timing == 'mid':
        exponent = max(0.0, (time_step - 0.5) * dt)
    else:  # 'end'
        exponent = max(0.0, time_step * dt)
    return 1.0 / ((1.0 + r) ** exponent)

def apply_stage_accumulations(individual_data: dict, config: dict, time_step: int) -> None:
    """
    Update cumulative QALYs and costs for the current cycle given stage and living setting.
    Applies NICE discounting at rate config['discount_rate_annual'] to this cycle's flows.
    Discount timing is controlled by config['discounting_timing'] in {'start','mid','end'}.
    """
    dt = float(config.get('time_step_years', 1.0) or 1.0)
    cycle_time_alive = float(individual_data.get('_cycle_time_alive', dt) or 0.0)
    if cycle_time_alive <= 0.0:
        return

    alive = bool(individual_data.get('alive', False))
    if alive:
        stage = individual_data['dementia_stage']
        setting = individual_data['living_setting']
    else:
        stage = individual_data.get('_cycle_stage_for_accrual')
        setting = individual_data.get('_cycle_setting_for_accrual')
        if stage is None or setting is None:
            return

    sex = individual_data.get('sex')
    age = individual_data['age']

    # Patient utility (absolute, clamped to [0, 1]); accrue for whole cohort including cognitively normal
    stage_age_qalys = config.get('stage_age_qalys')
    if stage == 'cognitively_normal':
        patient_weight = get_qaly_by_age_and_sex(age, sex, config)
    else:
        patient_weight = get_stage_age_qaly('patient', stage, age, setting, stage_age_qalys)
        if patient_weight is None:
            patient_stage = get_dementia_stage_qaly(stage, sex, config)
            if patient_stage is not None:
                patient_weight = patient_stage
            else:
                patient_weight = get_qaly_by_age_and_sex(age, sex, config)
    patient_weight = _clamp_utility(patient_weight)

    # Caregiver utility: use caregiver stage_age_qalys directly (no incremental baseline subtraction).
    caregiver_weight: float = 0.0
    if setting == 'home' and stage in ('mild', 'moderate', 'severe'):
        caregiver_utility = get_stage_age_qaly('caregiver', stage, age, setting, stage_age_qalys)
        if caregiver_utility is not None:
            caregiver_weight = _clamp_utility(caregiver_utility)

    costs = config['costs'].get(stage, {}).get(setting, {'nhs': 0.0, 'informal': 0.0})

    disc_factor = _discount_factor(time_step, config)

    # This cycle's (undiscounted) flows
    q_patient = patient_weight * cycle_time_alive
    q_caregiver = caregiver_weight * cycle_time_alive
    c_nhs = costs['nhs'] * cycle_time_alive
    c_informal = costs['informal'] * cycle_time_alive

    # Add discounted flows
    individual_data['cumulative_qalys_patient'] += q_patient * disc_factor
    individual_data['cumulative_qalys_caregiver'] += q_caregiver * disc_factor
    individual_data['cumulative_costs_nhs'] += c_nhs * disc_factor
    individual_data['cumulative_costs_informal'] += c_informal * disc_factor

    # Clear per-cycle accrual helpers so dead individuals do not repeatedly accrue in later cycles.
    for key in ('_cycle_time_alive', '_cycle_stage_for_accrual', '_cycle_setting_for_accrual'):
        if key in individual_data:
            individual_data.pop(key, None)

# Progression with mortality

def update_dementia_progression_optimized(population_state: Dict[int, dict],
                                config: dict,
                                time_step: int,
                                death_age_counter: Counter,
                                onset_tracker: Optional[Dict[str, Dict[str, int]]] = None,
                                age_band_exposure: Optional[Dict[Tuple[int, Optional[int]], float]] = None,
                                age_band_onsets: Optional[Dict[Tuple[int, Optional[int]], int]] = None,
                                age_band_exposure_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], float]]] = None,
                                age_band_onsets_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], int]]] = None
                                ) -> Tuple[int, int, Dict[Tuple[str, str], int], Dict[str, int], Dict[str, Dict[Tuple[int, Optional[int]], int]], Dict[str, Dict[Tuple[int, Optional[int]], int]]]:
    """
    OPTIMIZED: Advance dementia stages, apply mortality, update living settings,
    accumulate QALYs/costs, and count age bands - all in a single loop.

    This merges the functionality of:
    - update_dementia_progression()
    - update_stage_accumulations()
    - age band counting loop

    Returns: deaths, onsets, transition_counts, stage_start_counts, alive_counts_by_sex_band, prevalent_counts_by_sex_band
    """
    dt = config['time_step_years']
    base_year = int(config.get('base_year', 2023))
    calendar_year = base_year + time_step
    growth_cfg = config.get('incidence_growth') or {}
    if growth_cfg.get('use'):
        rate = float(growth_cfg.get('annual_rate', 0.0))
        ref_year = int(growth_cfg.get('reference_year', base_year))
        years_since_ref = max(0, calendar_year - ref_year)
        incidence_growth_multiplier = (1.0 + rate) ** years_since_ref
    else:
        incidence_growth_multiplier = 1.0
    deaths_this_step = 0
    onsets_this_step = 0
    transition_counter: Counter = Counter()
    stage_start_counts: Counter = Counter()

    # Initialize age band tracking dictionaries
    alive_counts_by_sex_band: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}
    prevalent_counts_by_sex_band: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}

    for person in population_state.values():
        if not person['alive']:
            continue

        # Per-cycle accrual helpers (used to ensure partial accrual in cycles where death occurs).
        person.pop('_cycle_time_alive', None)
        person.pop('_cycle_stage_for_accrual', None)
        person.pop('_cycle_setting_for_accrual', None)

        stage = person['dementia_stage']
        stage_start_counts[stage] += 1
        incidence_band: Optional[Tuple[int, Optional[int]]] = None
        sex_bucket = person.get('sex', 'unspecified')
        if stage == 'cognitively_normal':
            incidence_band = assign_age_to_reporting_band(person['age'], INCIDENCE_AGE_BANDS)
            if incidence_band is not None:
                if age_band_exposure is not None:
                    age_band_exposure[incidence_band] = age_band_exposure.get(incidence_band, 0.0) + dt
                if age_band_exposure_by_sex is not None:
                    exposure_by_band = age_band_exposure_by_sex.setdefault(sex_bucket, {})
                    exposure_by_band[incidence_band] = exposure_by_band.get(incidence_band, 0.0) + dt

        # --- Mortality step (competing risks if severe) ---
        # Year-specific hazard tables (if provided) override the base table
        bg_by_year = config.get('background_mortality_hazards_by_year') or {}
        bg_table_all = (
            bg_by_year.get(calendar_year)
            or bg_by_year.get(str(calendar_year))
            or {}
        )
        if isinstance(bg_table_all, dict) and any(isinstance(v, dict) for v in bg_table_all.values()):
            # nested by sex: choose table by person's sex, or fall back to 'all' or first available
            table = bg_table_all.get(sex_bucket) or bg_table_all.get('all')
            if table is None:
                # fall back to first nested dict if sex not found
                for v in bg_table_all.values():
                    if isinstance(v, dict):
                        table = v
                        break
            h_bg = get_background_mortality_hazard(person['age'], table or {})
        else:
            # flat table (backwards compatible)
            h_bg = get_background_mortality_hazard(person['age'], bg_table_all)

        # Optional scalar calibration (preserve age/sex shape, adjust level)
        h_bg *= background_mortality_scalar_for_year(config, calendar_year)

        # stage multiplier
        h_bg *= get_dementia_mortality_multiplier(stage, config.get('dementia_mortality_multipliers', {}))

        # Deterministic stage timing: only background mortality acts here
        h_total = h_bg

        p_death = hazard_to_prob(h_total, dt=dt)
        u_death = random.random()
        died_this_cycle = False
        transition_recorded = False
        if u_death < p_death:
            # Time alive within-cycle under a constant hazard, conditional on death in (0, dt].
            # With u_death < 1 - exp(-h_total*dt), t = -ln(1-u_death)/h_total lies in (0, dt].
            time_alive = dt
            if h_total > 0.0:
                try:
                    time_alive = -math.log(1.0 - u_death) / h_total
                except (ValueError, ZeroDivisionError, OverflowError):
                    time_alive = dt * 0.5
            time_alive = max(0.0, min(dt, float(time_alive)))
            person['_cycle_time_alive'] = time_alive
            person['_cycle_stage_for_accrual'] = stage
            person['_cycle_setting_for_accrual'] = person.get('living_setting')
            person['dementia_stage'] = 'death'
            person['alive'] = False
            person['living_setting'] = None
            death_age_counter[int(round(person['age']))] += 1
            deaths_this_step += 1
            transition_counter[(stage, 'death')] += 1
            died_this_cycle = True
            # OPTIMIZATION: Don't continue - need to do accumulations below

        # --- If still alive, apply stage progression (non-death transitions) ---
        if not died_this_cycle and stage == 'cognitively_normal':
            # Onset: either duration-driven (if provided) or base probability converted to hazard
            if 'normal_to_mild' in config['stage_transition_durations']:
                p = transition_prob_from_config(config, person, 'normal_to_mild')
            else:
                h0 = prob_to_hazard(config.get('base_onset_probability', 0.0), dt=dt)
                h0 *= incidence_growth_multiplier
                age_hr = get_age_hr_for_transition(person['age'], config, 'onset')  # CHANGED
                h = apply_hazard_ratios(
                    h0,
                    person['risk_factors'],
                    config['risk_factors'],
                    'onset',
                    age_hr,
                    person['age'],
                    person.get('sex', 'unspecified'),
                    config,  # NEW
                )
                p = hazard_to_prob(h, dt=dt)
            onset_triggered = False
            if random.random() < p:
                set_stage_and_countdown(person, 'mild', config)
                if not person.get('ever_dementia', False):
                    person['ever_dementia'] = True
                    person['age_at_onset'] = float(person.get('age', 0.0))
                elif person.get('age_at_onset') is None:
                    person['age_at_onset'] = float(person.get('age', 0.0))
                onsets_this_step += 1
                onset_triggered = True
                if onset_tracker is not None:
                    risk_flags = person.get('risk_factors', {})
                    for risk_name, counts in onset_tracker.items():
                        exposed = bool(risk_flags.get(risk_name, False))
                        bucket = 'with' if exposed else 'without'
                        counts[bucket] = counts.get(bucket, 0) + 1
                if onset_triggered and age_band_onsets is not None and incidence_band is not None:
                    age_band_onsets[incidence_band] = age_band_onsets.get(incidence_band, 0) + 1
                if onset_triggered and age_band_onsets_by_sex is not None and incidence_band is not None:
                    onset_by_band = age_band_onsets_by_sex.setdefault(sex_bucket, {})
                    onset_by_band[incidence_band] = onset_by_band.get(incidence_band, 0) + 1

        elif stage in ('mild', 'moderate', 'severe'):
            # Hazard-driven progression (no deterministic countdown)
            transition_map = {
                'mild': ('mild_to_moderate', 'moderate'),
                'moderate': ('moderate_to_severe', 'severe'),
                'severe': ('severe_to_death', 'death'),
            }
            key, next_stage = transition_map[stage]
            p = transition_prob_from_config(config, person, key)
            if random.random() < p:
                if next_stage == 'death':
                    # record death via progression hazard
                    person['_cycle_time_alive'] = dt
                    person['_cycle_stage_for_accrual'] = stage
                    person['_cycle_setting_for_accrual'] = person.get('living_setting')
                    person['dementia_stage'] = 'death'
                    person['alive'] = False
                    person['living_setting'] = None
                    death_age_counter[int(round(person['age']))] += 1
                    deaths_this_step += 1
                    transition_counter[(stage, 'death')] += 1
                    transition_recorded = True
                    died_this_cycle = True
                else:
                    set_stage_and_countdown(person, next_stage, config)
                    transition_counter[(stage, next_stage)] += 1
                    transition_recorded = True

        # Track transitions (already tracked for death above)
        if not died_this_cycle:
            end_stage = person['dementia_stage']
            if not transition_recorded:
                transition_counter[(stage, end_stage)] += 1

        # OPTIMIZATION: Merge living setting update and accumulations into same loop
        # (formerly done in separate update_stage_accumulations call)
        update_living_setting(person, config)
        apply_stage_accumulations(person, config, time_step)

        # OPTIMIZATION: Count alive/prevalent by age band in same loop
        # (formerly done in separate loop at lines 2511-2522)
        if person.get('alive', False):
            band = assign_age_to_reporting_band(person.get('age', 0.0), INCIDENCE_AGE_BANDS)
            if band is not None:
                sex_key = person.get('sex', 'unspecified')
                alive_bucket = alive_counts_by_sex_band.setdefault(sex_key, {})
                alive_bucket[band] = alive_bucket.get(band, 0) + 1
                if person.get('dementia_stage') in ('mild', 'moderate', 'severe'):
                    prevalent_bucket = prevalent_counts_by_sex_band.setdefault(sex_key, {})
                    prevalent_bucket[band] = prevalent_bucket.get(band, 0) + 1

    return deaths_this_step, onsets_this_step, dict(transition_counter), dict(stage_start_counts), alive_counts_by_sex_band, prevalent_counts_by_sex_band


# Backward-compatible wrapper for original function signature
def update_dementia_progression(population_state: Dict[int, dict],
                                config: dict,
                                time_step: int,
                                death_age_counter: Counter,
                                onset_tracker: Optional[Dict[str, Dict[str, int]]] = None,
                                age_band_exposure: Optional[Dict[Tuple[int, Optional[int]], float]] = None,
                                age_band_onsets: Optional[Dict[Tuple[int, Optional[int]], int]] = None,
                                age_band_exposure_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], float]]] = None,
                                age_band_onsets_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], int]]] = None
                                ) -> Tuple[int, int, Dict[Tuple[str, str], int], Dict[str, int]]:
    """
    BACKWARD-COMPATIBLE WRAPPER: Calls optimized version but returns only the original return values.
    Use update_dementia_progression_optimized() for the full optimized version.
    """
    deaths, onsets, trans, stage_starts, _, _ = update_dementia_progression_optimized(
        population_state, config, time_step, death_age_counter, onset_tracker,
        age_band_exposure, age_band_onsets, age_band_exposure_by_sex, age_band_onsets_by_sex
    )
    return deaths, onsets, trans, stage_starts


# Living setting transitions

def _select_living_setting_transition(config: Dict, stage: str, age: float) -> Dict[str, float]:
    """Return living setting transition probabilities for a stage/age, keeping backward compatibility."""
    stage_table = config.get('living_setting_transition_probabilities', {}).get(stage, {})
    if not stage_table:
        return {}
    # Legacy structure: flat dict per stage
    if isinstance(stage_table, dict) and (
        'to_institution' in stage_table or 'to_home' in stage_table
    ):
        return stage_table

    for band, probs in stage_table.items():
        if not isinstance(probs, dict):
            continue
        lower: Optional[float]
        upper: Optional[float]
        if isinstance(band, tuple) and len(band) == 2:
            lower, upper = band
        else:
            lower, upper = None, None
        lower = float(lower) if lower is not None else float('-inf')
        upper = float(upper) if upper is not None else float('inf')
        if lower <= age <= upper:
            return probs

    # Fallback in case no band matched (e.g. ages outside configured range)
    first = next(iter(stage_table.values()), {})
    return first if isinstance(first, dict) else {}


def update_living_setting(individual_data: dict, config: dict) -> None:
    if not individual_data['alive']:
        return
    stage = individual_data['dementia_stage']
    if stage in ['mild', 'moderate', 'severe']:
        probs = _select_living_setting_transition(config, stage, float(individual_data.get('age', 0.0)))
        current = individual_data['living_setting']
        if current == 'home' and random.random() < probs.get('to_institution', 0.0):
            individual_data['living_setting'] = 'institution'
        # Institution is absorbing for dementia stages; no transition back to home
    elif stage == 'cognitively_normal':
        individual_data['living_setting'] = 'home'

# Per-cycle accumulation wrapper

def update_stage_accumulations(population_state: Dict[int, dict],
                               time_step: int,
                               config: dict) -> None:
    """Apply living transitions, then add (discounted) QALYs/costs for the cycle."""
    for person in population_state.values():
        update_living_setting(person, config)
        apply_stage_accumulations(person, config, time_step)

# Reporting utils

def summarize_population_state(population_state: Dict[int, dict],
                               time_step: int,
                               base_year: int,
                               entrants: int = 0,
                               deaths: int = 0,
                               new_onsets: int = 0,
                               config: Optional[dict] = None) -> dict:
    """Aggregate key metrics for the current time step."""
    stage_counter = Counter()
    living_counter = Counter()
    age_band_dementia_counter = Counter()

    alive_count = 0
    age_alive_sum = 0.0
    baseline_alive_count = 0
    dementia_age_sum = 0.0
    dementia_count = 0
    total_qalys_patient = 0.0
    total_qalys_caregiver = 0.0
    total_costs_nhs = 0.0
    total_costs_informal = 0.0

    risk_counts_alive: Dict[str, int] = {}
    risk_counts_dementia: Dict[str, int] = {}
    risk_factor_names: List[str] = []
    if isinstance(config, dict):
        risk_factor_names = sorted((config.get('risk_factors') or {}).keys())
        risk_counts_alive = {name: 0 for name in risk_factor_names}
        risk_counts_dementia = {name: 0 for name in risk_factor_names}

    for person in population_state.values():
        stage = person['dementia_stage']
        stage_counter[stage] += 1
        total_qalys_patient += person['cumulative_qalys_patient']
        total_qalys_caregiver += person['cumulative_qalys_caregiver']
        total_costs_nhs += person['cumulative_costs_nhs']
        total_costs_informal += person['cumulative_costs_informal']

        if person['alive']:
            alive_count += 1
            age_alive_sum += person['age']
            living_counter[person.get('living_setting', 'unknown')] += 1
            has_dementia = stage in ('mild', 'moderate', 'severe')
            if has_dementia:
                dementia_age_sum += person['age']
                dementia_count += 1
                band = assign_age_to_reporting_band(person['age'])
                if band is not None:
                    age_band_dementia_counter[band] += 1
            if person.get('entry_time_step', 0) == 0:
                baseline_alive_count += 1

            if risk_factor_names:
                flags = person.get('risk_factors') or {}
                for risk_name in risk_factor_names:
                    if bool(flags.get(risk_name, False)):
                        risk_counts_alive[risk_name] += 1
                        if has_dementia:
                            risk_counts_dementia[risk_name] += 1

    mean_age_alive = age_alive_sum / alive_count if alive_count else 0.0
    mean_age_dementia = dementia_age_sum / dementia_count if dementia_count else 0.0

    summary = {
        'time_step': time_step,
        'calendar_year': base_year + time_step,
        'population_total': len(population_state),
        'population_alive': alive_count,
        'baseline_alive': baseline_alive_count,
        'entrants': entrants,
        'deaths': deaths,
        'incident_onsets': new_onsets,
        'incidence_per_1000_alive': (new_onsets / alive_count * 1000.0) if alive_count else 0.0,
        'total_qalys_patient': total_qalys_patient,
        'total_qalys_caregiver': total_qalys_caregiver,
        'total_qalys_total': total_qalys_patient + total_qalys_caregiver,
        'total_costs_nhs': total_costs_nhs,
        'total_costs_informal': total_costs_informal,
        'total_costs_nhs_perspective': total_costs_nhs,
        'total_costs_societal': total_costs_nhs + total_costs_informal,
        'dementia_cases_total': dementia_count,
        'dementia_prevalence_alive': (dementia_count / alive_count) if alive_count else 0.0,
        'mean_age_alive': mean_age_alive,
        'mean_age_dementia': mean_age_dementia,
    }

    for stage in DEMENTIA_STAGES:
        summary[f'stage_{stage}'] = stage_counter.get(stage, 0)

    for setting in LIVING_SETTINGS:
        summary[f'living_{setting}'] = living_counter.get(setting, 0)
    summary['living_unknown'] = living_counter.get('unknown', 0)

    for band in REPORTING_AGE_BANDS:
        summary[f'ad_cases_age_{age_band_key(band)}'] = age_band_dementia_counter.get(band, 0)

    if risk_factor_names:
        for risk_name in risk_factor_names:
            exposed_alive = int(risk_counts_alive.get(risk_name, 0) or 0)
            exposed_dementia = int(risk_counts_dementia.get(risk_name, 0) or 0)
            summary[f'risk_prev_alive_{risk_name}'] = (exposed_alive / alive_count) if alive_count else 0.0
            summary[f'risk_prev_dementia_{risk_name}'] = (exposed_dementia / dementia_count) if dementia_count else 0.0

    return summary

def generate_output(summary_history: Dict[int, dict], time_step: int) -> None:
    summary = summary_history.get(time_step)
    if summary is None:
        print(f'Time step {time_step}: no summary available')
        return

    print(f"Time step {time_step} (calendar year {summary.get('calendar_year', 'n/a')})")
    print('Stage counts:')
    for stage in DEMENTIA_STAGES:
        key = f'stage_{stage}'
        if key in summary:
            print(f"  {stage}: {summary[key]}")
    print(f"Alive count: {summary.get('population_alive', 0)}")
    print(f"Entrants this step: {summary.get('entrants', 0)}")
    print(f"Deaths this step: {summary.get('deaths', 0)}")
    if 'mean_age_alive' in summary:
        print(f"Mean age (alive): {summary['mean_age_alive']:.2f}")
    if 'mean_age_dementia' in summary:
        print(f"Mean age (dementia stages): {summary['mean_age_dementia']:.2f}")
    if 'incident_onsets' in summary:
        print(f"New onsets this step: {summary.get('incident_onsets', 0)}")

# Main run

def compute_lifetime_risk_by_entry_age(population_state: Dict[int, dict],
                                       restrict_to_cognitively_normal: bool = True) -> List[dict]:
    """
    Aggregate the lifetime dementia risk by entry age.

    Parameters
    ----------
    population_state: mapping of person ID to their final record.
    restrict_to_cognitively_normal: include only individuals who entered as cognitively normal.

    Returns
    -------
    A list of records sorted by entry age with keys:
        entry_age, population, dementia_cases, lifetime_risk.
    """
    total_by_age: Counter = Counter()
    dementia_by_age: Counter = Counter()

    for person in population_state.values():
        entry_age = person.get('entry_age')
        if entry_age is None:
            entry_age = person.get('age')
        if entry_age is None:
            continue
        try:
            entry_age_int = int(round(float(entry_age)))
        except (TypeError, ValueError):
            continue

        baseline_stage = person.get('baseline_stage', 'cognitively_normal')
        if restrict_to_cognitively_normal and baseline_stage != 'cognitively_normal':
            continue

        total_by_age[entry_age_int] += 1

        ever_dementia = bool(person.get('ever_dementia', False))
        if baseline_stage in ('mild', 'moderate', 'severe'):
            ever_dementia = True
        if ever_dementia:
            dementia_by_age[entry_age_int] += 1

    records: List[dict] = []
    for age in sorted(total_by_age.keys()):
        population = total_by_age[age]
        if population <= 0:
            continue
        dementia_cases = dementia_by_age.get(age, 0)
        risk = dementia_cases / population if population else 0.0
        records.append({
            'entry_age': age,
            'population': population,
            'dementia_cases': dementia_cases,
            'lifetime_risk': risk,
        })

    return records

def collect_individual_survival(population_state: Dict[int, dict]) -> List[dict]:
    records: List[dict] = []
    for person in population_state.values():
        baseline_stage = person.get('baseline_stage', person.get('dementia_stage', 'unknown'))
        record = {
            'ID': person.get('ID'),
            'baseline_stage': baseline_stage,
            'time': float(person.get('time_since_entry', 0.0)),
            'event': 0 if person.get('alive', False) else 1,
            'entry_time_step': int(person.get('entry_time_step', 0)),
        }
        records.append(record)
    return records


def summaries_to_dataframe(model_results: dict) -> pd.DataFrame:
    """Convert stored summaries into a tidy dataframe."""
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    rows = []
    for time_step in sorted(summaries.keys()):
        summary = summaries[time_step].copy()
        rows.append(summary)
    return pd.DataFrame(rows)

def run_model(config: dict, seed: Optional[int] = None, return_agents: bool = False) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ensure_background_mortality_hazards_by_year(config)
    number_of_timesteps = config['number_of_timesteps'] + 1
    population = config['population']
    base_year = int(config.get('base_year', 2023))

    summary_history = initialize_model_dictionary()
    population_state, initial_age_counter = initialize_population(population, config)
    death_age_counter: Counter = Counter()
    transition_history: Dict[int, dict] = {}
    risk_onset_tracker: Dict[str, Dict[str, int]] = {
        name: {'with': 0, 'without': 0} for name in config.get('risk_factors', {})
    }
    incidence_age_exposure: Dict[Tuple[int, Optional[int]], float] = defaultdict(float)
    incidence_age_onsets: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)

    baseline_summary = summarize_population_state(population_state, 0, base_year, entrants=0, deaths=0, config=config)
    baseline_overrides = config.get('initial_summary_overrides') or {}
    if baseline_overrides:
        baseline_summary.update(baseline_overrides)
        if 'incident_onsets' in baseline_overrides:
            alive = baseline_summary.get('population_alive', 0) or 0
            baseline_summary['incident_onsets'] = baseline_overrides['incident_onsets']
            baseline_summary['incidence_per_1000_alive'] = (
                (baseline_summary['incident_onsets'] / alive) * 1000.0 if alive else 0.0
            )
        if 'deaths' in baseline_overrides:
            baseline_summary['deaths'] = baseline_overrides['deaths']
        if 'entrants' in baseline_overrides:
            baseline_summary['entrants'] = baseline_overrides['entrants']
        if 'calendar_year' in baseline_overrides:
            baseline_summary['calendar_year'] = baseline_overrides['calendar_year']
        if 'time_step' in baseline_overrides:
            baseline_summary['time_step'] = baseline_overrides['time_step']

    # Year-on-year (per-cycle) flows are defined as differences in cumulative discounted totals.
    baseline_summary.setdefault('year_qalys_patient', 0.0)
    baseline_summary.setdefault('year_qalys_caregiver', 0.0)
    baseline_summary.setdefault('year_qalys_total', 0.0)
    baseline_summary.setdefault('year_costs_nhs', 0.0)
    baseline_summary.setdefault('year_costs_informal', 0.0)
    baseline_summary.setdefault('year_costs_societal', 0.0)

    create_time_step_dictionary(summary_history, 0, baseline_summary)
    generate_output(summary_history, 0)

    next_id = len(population_state)
    yearly_incidence_records: List[dict] = []

    for time_step in range(1, number_of_timesteps):
        calendar_year = base_year + time_step
        advance_population_state(population_state, config, calendar_year)

        next_id, entrants_this_step = add_new_entrants(population_state, config, next_id, calendar_year)
        per_sex_exposure: Dict[str, Dict[Tuple[int, Optional[int]], float]] = {}
        per_sex_onsets: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}

        # OPTIMIZATION: Use combined function that merges progression + accumulations + counting
        deaths_this_step, onsets_this_step, transition_counts, stage_start_counts, alive_counts_by_sex_band, prevalent_counts_by_sex_band = update_dementia_progression_optimized(
            population_state,
            config,
            time_step,
            death_age_counter,
            risk_onset_tracker,
            incidence_age_exposure,
            incidence_age_onsets,
            per_sex_exposure,
            per_sex_onsets
        )
        # OPTIMIZATION: Removed separate update_stage_accumulations() call - now done in combined loop above
        # OPTIMIZATION: Removed separate age band counting loop - now done in combined loop above

        sexes_present = (
            set(per_sex_exposure.keys())
            | set(per_sex_onsets.keys())
            | set(alive_counts_by_sex_band.keys())
            | set(prevalent_counts_by_sex_band.keys())
            | {'female', 'male'}
        )
        sexes_present.discard('all')

        totals_per_band = defaultdict(lambda: {
            'person_years_at_risk': 0.0,
            'incident_onsets_at_risk': 0,
            'population_alive_in_band': 0,
            'prevalent_dementia_cases_in_band': 0,
        })

        for sex in sorted(sexes_present):
            exposure_by_band = per_sex_exposure.get(sex, {})
            onsets_by_band = per_sex_onsets.get(sex, {})
            alive_by_band = alive_counts_by_sex_band.get(sex, {})
            prevalence_by_band = prevalent_counts_by_sex_band.get(sex, {})
            for band in INCIDENCE_AGE_BANDS:
                lower, upper = band
                band_label = age_band_label(band)
                person_years = float(exposure_by_band.get(band, 0.0))
                incident_onsets = int(onsets_by_band.get(band, 0))
                alive_count = int(alive_by_band.get(band, 0))
                prevalent_count = int(prevalence_by_band.get(band, 0))
                record = {
                    'time_step': time_step,
                    'calendar_year': calendar_year,
                    'sex': sex,
                    'age_band': band_label,
                    'age_lower': lower,
                    'age_upper': upper,
                    'person_years_at_risk': person_years,
                    'incident_onsets_at_risk': incident_onsets,
                    'population_alive_in_band': alive_count,
                    'prevalent_dementia_cases_in_band': prevalent_count,
                }
                yearly_incidence_records.append(record)
                totals_metrics = totals_per_band[band]
                totals_metrics['person_years_at_risk'] += person_years
                totals_metrics['incident_onsets_at_risk'] += incident_onsets
                totals_metrics['population_alive_in_band'] += alive_count
                totals_metrics['prevalent_dementia_cases_in_band'] += prevalent_count

        for band in INCIDENCE_AGE_BANDS:
            lower, upper = band
            band_label = age_band_label(band)
            totals_metrics = totals_per_band[band]
            record_all = {
                'time_step': time_step,
                'calendar_year': calendar_year,
                'sex': 'all',
                'age_band': band_label,
                'age_lower': lower,
                'age_upper': upper,
                'person_years_at_risk': float(totals_metrics['person_years_at_risk']),
                'incident_onsets_at_risk': int(totals_metrics['incident_onsets_at_risk']),
                'population_alive_in_band': int(totals_metrics['population_alive_in_band']),
                'prevalent_dementia_cases_in_band': int(totals_metrics['prevalent_dementia_cases_in_band']),
            }
            yearly_incidence_records.append(record_all)

        transition_history[time_step] = {
            'transition_counts': transition_counts,
            'stage_start_counts': stage_start_counts,
        }

        summary = summarize_population_state(population_state,
                                             time_step,
                                             base_year,
                                             entrants=entrants_this_step,
                                             deaths=deaths_this_step,
                                             new_onsets=onsets_this_step,
                                             config=config)

        prev = summary_history.get(time_step - 1, {}) or {}
        summary['year_qalys_patient'] = float(summary.get('total_qalys_patient', 0.0) or 0.0) - float(prev.get('total_qalys_patient', 0.0) or 0.0)
        summary['year_qalys_caregiver'] = float(summary.get('total_qalys_caregiver', 0.0) or 0.0) - float(prev.get('total_qalys_caregiver', 0.0) or 0.0)
        summary['year_qalys_total'] = summary['year_qalys_patient'] + summary['year_qalys_caregiver']
        summary['year_costs_nhs'] = float(summary.get('total_costs_nhs', 0.0) or 0.0) - float(prev.get('total_costs_nhs', 0.0) or 0.0)
        summary['year_costs_informal'] = float(summary.get('total_costs_informal', 0.0) or 0.0) - float(prev.get('total_costs_informal', 0.0) or 0.0)
        summary['year_costs_societal'] = float(summary.get('total_costs_societal', 0.0) or 0.0) - float(prev.get('total_costs_societal', 0.0) or 0.0)

        create_time_step_dictionary(summary_history, time_step, summary)
        generate_output(summary_history, time_step)

    lifetime_risk_normal = compute_lifetime_risk_by_entry_age(population_state, restrict_to_cognitively_normal=True)
    lifetime_risk_all = compute_lifetime_risk_by_entry_age(population_state, restrict_to_cognitively_normal=False)
    if config.get('store_individual_survival', True):
        survival_records = collect_individual_survival(population_state)
    else:
        survival_records = []

    # Capture final alive/dementia counts by incidence age band
    age_band_alive_counts: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)
    age_band_dementia_counts: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)
    for person in population_state.values():
        if not person.get('alive', False):
            continue
        band = assign_age_to_reporting_band(float(person.get('age', 0.0)), INCIDENCE_AGE_BANDS)
        if band is None:
            continue
        age_band_alive_counts[band] += 1
        if person.get('dementia_stage') in ('mild', 'moderate', 'severe'):
            age_band_dementia_counts[band] += 1

    incidence_age_records: List[dict] = []
    for band in INCIDENCE_AGE_BANDS:
        exposure = float(incidence_age_exposure.get(band, 0.0))
        events = int(incidence_age_onsets.get(band, 0))
        hazard_per_year = (events / exposure) if exposure > 0 else 0.0
        probability_per_year = hazard_to_prob(hazard_per_year, dt=1.0)
        alive_count = int(age_band_alive_counts.get(band, 0))
        dementia_count = int(age_band_dementia_counts.get(band, 0))
        prevalence_value = (dementia_count / alive_count) if alive_count > 0 else 0.0
        lower, upper = band
        incidence_age_records.append({
            'age_band': age_band_label(band),
            'age_lower': lower,
            'age_upper': upper,
            'age_mid': age_band_midpoint(band),
            'person_years_at_risk': exposure,
            'incident_onsets': events,
             'cases_all': dementia_count,
             'population_all': alive_count,
             'prevalence': prevalence_value,
            'incidence_hazard_per_year': hazard_per_year,
            'incidence_probability_per_year': probability_per_year,
        })

    incidence_age_df = pd.DataFrame(incidence_age_records)
    if not incidence_age_df.empty:
        incidence_age_df['prevalence (smoothed)'] = smooth_series(
            incidence_age_df['prevalence'].tolist()
        )
        ref_hazard_series = incidence_age_df.loc[
            incidence_age_df['incidence_hazard_per_year'] > 0,
            'incidence_hazard_per_year'
        ]
        ref_hazard = float(ref_hazard_series.iloc[0]) if not ref_hazard_series.empty else float('nan')
        if ref_hazard > 0 and math.isfinite(ref_hazard):
            incidence_age_df['h/h_ref'] = incidence_age_df['incidence_hazard_per_year'] / ref_hazard
        else:
            incidence_age_df['h/h_ref'] = np.nan
        incidence_age_df['log(h/h_ref)'] = np.log(incidence_age_df['h/h_ref'].replace({0: np.nan}))
        incidence_age_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Reorder and rename columns for downstream consumers
        column_map = {
            'age_band': 'age band',
            'age_mid': 'age mid',
            'cases_all': 'cases (all)',
            'population_all': 'population (all)',
            'prevalence': 'prevalence',
            'incidence_hazard_per_year': 'Incidence hazard/yr'
        }
        incidence_age_df.rename(columns=column_map, inplace=True)
        desired_columns = [
            'age band',
            'age mid',
            'cases (all)',
            'population (all)',
            'prevalence',
            'prevalence (smoothed)',
            'Incidence hazard/yr',
            'h/h_ref',
            'log(h/h_ref)',
            'person_years_at_risk',
            'incident_onsets',
            'age_lower',
            'age_upper',
            'incidence_probability_per_year',
        ]
        # Keep consistent ordering for later selection
        existing_cols = [c for c in desired_columns if c in incidence_age_df.columns]
        remaining_cols = [c for c in incidence_age_df.columns if c not in existing_cols]
        incidence_age_df = incidence_age_df[existing_cols + remaining_cols]

    incidence_by_year_sex_df = pd.DataFrame(yearly_incidence_records)
    if not incidence_by_year_sex_df.empty:
        incidence_by_year_sex_df.sort_values(['calendar_year', 'sex', 'age_lower'], inplace=True)
        incidence_by_year_sex_df.reset_index(drop=True, inplace=True)
        denom = incidence_by_year_sex_df['population_alive_in_band'].replace(0, np.nan)
        incidence_by_year_sex_df['dementia_prevalence_in_band'] = (
            incidence_by_year_sex_df['prevalent_dementia_cases_in_band'] / denom
        ).fillna(0.0)

    onset_age_counter: Counter = Counter()
    for person in population_state.values():
        age_at_onset = person.get('age_at_onset')
        if age_at_onset is None:
            continue
        try:
            age_int = int(round(float(age_at_onset)))
        except (TypeError, ValueError):
            continue
        onset_age_counter[age_int] += 1

    # Build results dictionary
    results = {
        'summaries': summary_history,
        'initial_age_distribution': dict(initial_age_counter),
        'age_at_death_distribution': dict(death_age_counter),
        'individual_survival': survival_records,
        'transition_history': transition_history,
        'incident_onsets_by_risk_factor': risk_onset_tracker,
        'lifetime_risk_by_entry_age': lifetime_risk_normal,
        'lifetime_risk_by_entry_age_all': lifetime_risk_all,
        'incidence_by_age_band': incidence_age_records,
        'incidence_by_age_band_df': incidence_age_df,
        'incidence_by_year_sex': yearly_incidence_records,
        'incidence_by_year_sex_df': incidence_by_year_sex_df,
        'age_at_onset_distribution': dict(onset_age_counter),
        'age_band_alive_counts': {age_band_label(b): int(count) for b, count in age_band_alive_counts.items()},
        'age_band_dementia_counts': {age_band_label(b): int(count) for b, count in age_band_dementia_counts.items()},
        'age_band_incidence_summary': incidence_age_df[['age band',
                                                        'age mid',
                                                        'cases (all)',
                                                        'population (all)',
                                                        'prevalence',
                                                        'prevalence (smoothed)',
                                                        'Incidence hazard/yr',
                                                        'h/h_ref',
                                                        'log(h/h_ref)']].copy() if not incidence_age_df.empty else pd.DataFrame(),
    }

    # Optionally include agents data for validation (can be memory-intensive for large populations)
    if return_agents:
        # Memory-efficient chunked conversion for large populations
        chunk_size = 100000
        agent_chunks = []
        agent_ids = list(population_state.keys())

        for i in range(0, len(agent_ids), chunk_size):
            chunk_ids = agent_ids[i:i + chunk_size]
            chunk_data = {agent_id: population_state[agent_id] for agent_id in chunk_ids}
            agent_chunks.append(pd.DataFrame.from_dict(chunk_data, orient='index'))

        results['agents'] = pd.concat(agent_chunks, ignore_index=False) if agent_chunks else pd.DataFrame()

    return results


def _total_incident_onsets(model_results: dict) -> int:
    """Sum dementia onsets across all simulated time steps."""
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    total = 0
    for summary in summaries.values():
        total += int(summary.get('incident_onsets', 0) or 0)
    return total


def _replace_nested_values(structure: Any, value: float) -> Any:
    """Recursively replace all numeric leaves in a nested mapping with the provided value."""
    if isinstance(structure, dict):
        return {key: _replace_nested_values(nested, value) for key, nested in structure.items()}
    return float(value)


def _counterfactual_config_without_risk(config: dict, risk_factor: str) -> dict:
    """
    Create a deep copy of the configuration where the chosen risk factor has zero prevalence
    and neutral (1.0) hazard ratios across all transitions.
    """
    cfg = copy.deepcopy(config)
    risk_defs = cfg.get('risk_factors', {})
    meta = risk_defs.get(risk_factor)
    if not isinstance(meta, dict):
        return cfg

    meta['prevalence'] = 0.0
    rr_spec = meta.get('hazard_ratios', {})
    if isinstance(rr_spec, dict) and rr_spec:
        meta['hazard_ratios'] = _replace_nested_values(rr_spec, 1.0)
    else:
        meta['hazard_ratios'] = {
            'onset': 1.0,
            'mild_to_moderate': 1.0,
            'moderate_to_severe': 1.0,
            'severe_to_death': 1.0,
        }
    return cfg


def compute_population_attributable_fraction(config: dict,
                                             risk_factor: str,
                                             baseline_results: Optional[dict] = None,
                                             seed: Optional[int] = None) -> Optional[dict]:
    """
    Estimate the population attributable fraction (PAF) for a named risk factor by
    comparing baseline simulation results with a counterfactual scenario in which
    the risk factor is removed (zero prevalence and neutral hazard ratios).
    """
    baseline_results_local = baseline_results
    if baseline_results_local is None:
        baseline_results_local = run_model(copy.deepcopy(config), seed=seed)

    counterfactual_results = run_model(
        _counterfactual_config_without_risk(config, risk_factor),
        seed=seed,
    )

    baseline_onsets = _total_incident_onsets(baseline_results_local)
    counterfactual_onsets = _total_incident_onsets(counterfactual_results)
    if baseline_onsets <= 0:
        return None

    paf = (baseline_onsets - counterfactual_onsets) / baseline_onsets
    paf = max(0.0, min(1.0, paf))

    baseline_breakdown = (
        baseline_results_local
        .get('incident_onsets_by_risk_factor', {})
        .get(risk_factor, {})
        if isinstance(baseline_results_local, dict) else {}
    )

    return {
        'risk_factor': risk_factor,
        'baseline_onsets': baseline_onsets,
        'counterfactual_onsets': counterfactual_onsets,
        'paf': paf,
        'baseline_with_risk_onsets': baseline_breakdown.get('with', 0),
        'baseline_without_risk_onsets': baseline_breakdown.get('without', 0),
    }


# -------- Probabilistic sensitivity analysis (PSA) utilities --------

def apply_psa_draw(base_config: dict,
                   psa_cfg: Optional[dict] = None,
                   rng: Optional[np.random.Generator] = None) -> dict:
    """
    Return a deep-copied config with PSA sampling applied to costs, utilities,
    probabilities, and risk-factor parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
    psa_meta = psa_cfg or base_config.get('psa') or {}
    rel_beta = float(psa_meta.get('relative_sd_beta', PSA_DEFAULT_RELATIVE_SD))
    rel_gamma = float(psa_meta.get('relative_sd_gamma', PSA_DEFAULT_RELATIVE_SD))

    cfg = copy.deepcopy(base_config)

    base_prob = cfg.get('base_onset_probability')
    if base_prob is not None:
        cfg['base_onset_probability'] = _sample_probability_value(base_prob, rel_beta, rng)

    costs_cfg = cfg.get('costs')
    if isinstance(costs_cfg, dict):
        _sample_cost_structure(costs_cfg, rel_gamma, rng)

    for key in ('utility_norms_by_age', 'stage_age_qalys', 'dementia_stage_qalys'):
        mapping = cfg.get(key)
        if isinstance(mapping, dict):
            _apply_beta_to_mapping(mapping, rel_beta, rng)

    risk_defs = cfg.get('risk_factors')
    if isinstance(risk_defs, dict):
        _sample_risk_factor_prevalence(risk_defs, rel_beta, rng)
        _sample_risk_factor_hazard_ratios(risk_defs, rng)

    return cfg


def extract_psa_metrics(model_results: dict) -> dict:
    """
    Pull decision metrics (costs, QALYs, incidence, severity) from a single model run.
    """
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    if not summaries:
        return {}
    final_step = max(summaries)
    final_summary = summaries[final_step]

    total_incidence = 0
    for summary in summaries.values():
        total_incidence += int(summary.get('incident_onsets', 0) or 0)

    metrics = {
        'total_costs_nhs': float(final_summary.get('total_costs_nhs', 0.0) or 0.0),
        'total_costs_informal': float(final_summary.get('total_costs_informal', 0.0) or 0.0),
        'total_qalys_patient': float(final_summary.get('total_qalys_patient', 0.0) or 0.0),
        'total_qalys_caregiver': float(final_summary.get('total_qalys_caregiver', 0.0) or 0.0),
        'incident_onsets_total': float(total_incidence),
    }
    metrics['total_costs_all'] = metrics['total_costs_nhs'] + metrics['total_costs_informal']
    metrics['total_qalys_combined'] = metrics['total_qalys_patient'] + metrics['total_qalys_caregiver']

    for stage in ('mild', 'moderate', 'severe'):
        key = f'stage_{stage}'
        metrics[key] = float(final_summary.get(key, 0) or 0)

    return metrics


def summarize_psa_results(metrics_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute mean and 95% intervals for each numeric PSA metric.
    """
    if metrics_df is None or metrics_df.empty:
        return {}
    summary: Dict[str, dict] = {}
    for column in metrics_df.columns:
        if column == 'iteration':
            continue
        if not pd.api.types.is_numeric_dtype(metrics_df[column]):
            continue
        series = metrics_df[column].dropna()
        if series.empty:
            continue
        summary[column] = {
            'mean': float(series.mean()),
            'lower_95': float(series.quantile(0.025)),
            'upper_95': float(series.quantile(0.975)),
        }
    return summary


def _run_single_psa_iteration(args: Tuple[int, dict, dict, int]) -> dict:
    """
    Helper function to run a single PSA iteration in parallel.

    Args:
        args: Tuple of (draw_idx, base_config, psa_meta, draw_seed)

    Returns:
        Dictionary of metrics for this iteration
    """
    draw_idx, base_config, psa_meta, draw_seed = args
    # Create a new RNG for this draw to ensure reproducibility
    rng = np.random.default_rng(draw_seed)
    draw_config = apply_psa_draw(base_config, psa_meta, rng)
    # Generate a seed for the model run
    model_seed = int(rng.integers(0, 2**32 - 1))
    draw_results = run_model(draw_config, seed=model_seed)
    metrics = extract_psa_metrics(draw_results)
    metrics['iteration'] = draw_idx + 1
    return metrics


def run_probabilistic_sensitivity_analysis(base_config: dict,
                                           psa_cfg: Optional[dict] = None,
                                           *,
                                           collect_draw_level: bool = False,
                                           seed: Optional[int] = None,
                                           n_jobs: Optional[int] = None) -> dict:
    """
    Execute a Monte Carlo PSA using the provided configuration.
    Returns summary 95% intervals plus optional draw-level metrics.

    Args:
        base_config: Base configuration dictionary
        psa_cfg: PSA-specific configuration (optional). Can include 'original_population'
                 to automatically scale new entrants when running with reduced population.
        collect_draw_level: If True, include all draw-level metrics in output
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs. If None, uses psa_cfg['n_jobs'] or cpu_count().
                Set to 1 to disable parallelization.

    Returns:
        Dictionary with 'summary' (95% CI), 'iterations', and optionally 'draws'

    Note:
        When running PSA with reduced population (e.g., 1% for faster computation),
        set psa_cfg['original_population'] to the full population size. This will
        automatically scale new entrants proportionally to maintain correct dynamics.
    """
    psa_meta = copy.deepcopy(psa_cfg or base_config.get('psa') or {})
    if not psa_meta.get('use', False):
        raise ValueError("PSA is disabled; set config['psa']['use'] = True to run the analysis.")

    iterations = int(psa_meta.get('iterations', 1000))
    if iterations <= 0:
        raise ValueError("PSA iterations must be a positive integer.")

    # Determine number of parallel jobs (robust to None/Non-numeric inputs)
    if n_jobs is None:
        n_jobs = psa_meta.get('n_jobs')
    if n_jobs is None:
        n_jobs = cpu_count()
    try:
        n_jobs = int(n_jobs)
    except (TypeError, ValueError):
        n_jobs = cpu_count()
    n_jobs = max(1, n_jobs)  # Ensure at least 1

    base_seed = seed if seed is not None else psa_meta.get('seed')
    rng = np.random.default_rng(base_seed)

    # Check if population scaling is needed for new entrants
    # This ensures that when running PSA with reduced population (e.g., 1%),
    # the new entrants are also scaled proportionally
    working_config = base_config
    original_pop = psa_meta.get('original_population')
    current_pop = base_config.get('population')

    if original_pop and current_pop and original_pop != current_pop:
        print(f"\nScaling new entrants: population {current_pop:,} / {original_pop:,} = {current_pop/original_pop:.2%}")
        working_config = _with_scaled_population_and_entrants(
            base_config,
            new_population=current_pop,
            original_population=original_pop
        )
        # Log the scaling for transparency
        if 'open_population' in working_config:
            op = working_config['open_population']
            if op.get('use', False):
                scaled_entrants = op.get('entrants_per_year', 0)
                original_entrants = base_config.get('open_population', {}).get('entrants_per_year', 0)
                if original_entrants != scaled_entrants:
                    print(f"  New entrants scaled: {original_entrants:,} → {scaled_entrants:,} per year")

    # Pre-generate all seeds for reproducibility
    draw_seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(iterations)]

    # Prepare arguments for all iterations
    psa_args = [
        (draw_idx, working_config, psa_meta, draw_seeds[draw_idx])
        for draw_idx in range(iterations)
    ]

    print(f"\nRunning PSA with {iterations} iterations using {n_jobs} parallel job(s)...")
    if TQDM_AVAILABLE:
        print("Progress tracking enabled (tqdm installed)")
    else:
        print("Install tqdm for progress tracking: pip install tqdm")

    # Run PSA iterations in parallel (or serial if n_jobs=1)
    if n_jobs == 1:
        # Serial execution with progress bar
        draw_metrics = []
        for args in tqdm(psa_args, desc="PSA iterations", disable=not TQDM_AVAILABLE):
            draw_metrics.append(_run_single_psa_iteration(args))
    else:
        # Parallel execution
        with Pool(processes=n_jobs) as pool:
            if TQDM_AVAILABLE:
                # Use tqdm with parallel processing
                draw_metrics = list(tqdm(
                    pool.imap(_run_single_psa_iteration, psa_args),
                    total=iterations,
                    desc="PSA iterations"
                ))
            else:
                # No progress bar for parallel execution without tqdm
                draw_metrics = pool.map(_run_single_psa_iteration, psa_args)

    metrics_df = pd.DataFrame(draw_metrics)
    summary = summarize_psa_results(metrics_df)

    payload = {
        'summary': summary,
        'iterations': iterations,
        'n_jobs_used': n_jobs,
    }
    if collect_draw_level:
        payload['draws'] = metrics_df
    return payload


# Output compression utilities

def save_results_compressed(results: dict, filepath: Union[str, Path],
                           compression_level: int = 6) -> None:
    """
    Save model results to a compressed pickle file.

    Args:
        results: Dictionary of results to save
        filepath: Path where to save the file (should end with .pkl.gz)
        compression_level: gzip compression level (1-9, higher = more compression)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(filepath, 'wb', compresslevel=compression_level) as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Report file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved compressed results to {filepath} ({size_mb:.2f} MB)")


def load_results_compressed(filepath: Union[str, Path]) -> dict:
    """
    Load model results from a compressed pickle file.

    Args:
        filepath: Path to the compressed file

    Returns:
        Dictionary of results
    """
    filepath = Path(filepath)
    with gzip.open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def save_dataframe_compressed(df: pd.DataFrame, filepath: Union[str, Path],
                              format: str = 'parquet') -> None:
    """
    Save a DataFrame in compressed format.

    Args:
        df: DataFrame to save
        filepath: Path where to save
        format: 'parquet' (recommended) or 'csv' (with gzip)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
        df.to_parquet(filepath, compression='gzip', index=False)
    elif format == 'csv':
        if not str(filepath).endswith('.gz'):
            filepath = Path(str(filepath) + '.gz')
        df.to_csv(filepath, index=False, compression='gzip')
    else:
        raise ValueError(f"Unsupported format: {format}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved compressed DataFrame to {filepath} ({size_mb:.2f} MB)")


# Two-Level ANOVA-based PSA (O'Hagan et al. 2007 method)

def _with_scaled_population_and_entrants(base_config: dict,
                                         new_population: int,
                                         original_population: int) -> dict:
    """
    Return a deep copy of the config with population set to new_population and
    entrant counts scaled proportionally to preserve open-population dynamics
    during reduced-size PSA runs.
    """
    cfg = copy.deepcopy(base_config)
    cfg['population'] = new_population

    if original_population and original_population > 0 and new_population > 0:
        ratio = new_population / original_population

        op = cfg.get('open_population')
        if isinstance(op, dict) and op.get('use', False):
            entrants = op.get('entrants_per_year')
            if isinstance(entrants, (int, float)):
                scaled = int(round(entrants * ratio))
                op['entrants_per_year'] = max(0, scaled)

        overrides = cfg.get('initial_summary_overrides')
        if isinstance(overrides, dict):
            for key in ('entrants', 'incident_onsets', 'deaths'):
                if key in overrides:
                    base_val = overrides.get(key)
                    if isinstance(base_val, (int, float)):
                        scaled = int(round(base_val * ratio))
                        overrides[key] = max(0, scaled)

    return cfg


def estimate_variance_components_pilot(base_config: dict,
                                       psa_cfg: dict,
                                       n_outer: int = 10,
                                       n_inner_list: Optional[List[int]] = None,
                                       seed: Optional[int] = None) -> dict:
    """
    Pilot study to estimate variance components using ANOVA decomposition.

    This implements the O'Hagan et al. (2007) method for patient-level simulation models.
    Runs a small number of parameter sets (n_outer) with multiple population sizes (n_inner_list)
    to estimate:
    - σ²_between: Variance due to parameter uncertainty (PSA signal)
    - σ²_within: Variance due to stochastic patient sampling (noise)

    Args:
        base_config: Base configuration dictionary
        psa_cfg: PSA configuration
        n_outer: Number of parameter sets to sample (default 10)
        n_inner_list: List of population sizes to test (default [1000, 5000, 10000])
        seed: Random seed

    Returns:
        Dictionary with variance estimates and optimal n recommendation
    """
    if n_inner_list is None:
        n_inner_list = [1000, 5000, 10000]

    print(f"\n{'='*60}")
    print("PILOT STUDY: Estimating Variance Components (O'Hagan Method)")
    print(f"{'='*60}")
    print(f"Outer iterations (parameter sets): {n_outer}")
    print(f"Inner population sizes to test: {n_inner_list}")

    base_seed = seed if seed is not None else psa_cfg.get('seed', 42)
    rng = np.random.default_rng(base_seed)
    psa_meta = copy.deepcopy(psa_cfg)

    # Save original population size
    original_pop = base_config.get('population', 33167098)

    results_by_n = {}

    for n_inner in n_inner_list:
        print(f"\nTesting with n={n_inner} individuals per parameter set...")

        # Create modified config with reduced population
        test_config = _with_scaled_population_and_entrants(
            base_config,
            new_population=n_inner,
            original_population=original_pop
        )

        # Run n_outer parameter sets
        outcomes = []
        for i in range(n_outer):
            draw_seed = int(rng.integers(0, 2**32 - 1))
            draw_config = apply_psa_draw(test_config, psa_meta, rng)

            model_seed = int(rng.integers(0, 2**32 - 1))
            draw_results = run_model(draw_config, seed=model_seed)
            metrics = extract_psa_metrics(draw_results)
            outcomes.append(metrics)

        # Calculate variance components using ANOVA
        outcomes_df = pd.DataFrame(outcomes)

        # For each outcome metric, estimate variance
        variance_stats = {}
        for col in outcomes_df.columns:
            if col == 'iteration' or not pd.api.types.is_numeric_dtype(outcomes_df[col]):
                continue

            values = outcomes_df[col].dropna()
            if len(values) < 2:
                continue

            # Total variance at this sample size
            total_var = float(values.var())
            mean_val = float(values.mean())

            variance_stats[col] = {
                'mean': mean_val,
                'total_variance': total_var,
                'n': n_inner
            }

        results_by_n[n_inner] = variance_stats
        print(f"  Completed {n_outer} parameter sets with n={n_inner}")

    # Estimate σ²_between and σ²_within using variance at different n
    print(f"\n{'='*60}")
    print("Variance Component Estimates:")
    print(f"{'='*60}")

    variance_components = {}

    # For each metric, fit Var(ȳ) = σ²_between + σ²_within/n
    for metric in results_by_n[n_inner_list[0]].keys():
        try:
            # Extract variance vs 1/n data
            n_vals = []
            var_vals = []

            for n_inner in n_inner_list:
                if metric in results_by_n[n_inner]:
                    n_vals.append(n_inner)
                    var_vals.append(results_by_n[n_inner][metric]['total_variance'])

            if len(n_vals) < 2:
                continue

            # Linear regression: Var = σ²_between + σ²_within * (1/n)
            one_over_n = np.array([1.0/n for n in n_vals])
            var_array = np.array(var_vals)

            # Fit: y = a + b*x where y=Var, x=1/n, a=σ²_between, b=σ²_within
            coeffs = np.polyfit(one_over_n, var_array, 1)
            sigma_within_sq = coeffs[0]  # Slope
            sigma_between_sq = coeffs[1]  # Intercept

            # Ensure non-negative
            sigma_between_sq = max(0, sigma_between_sq)
            sigma_within_sq = max(0, sigma_within_sq)

            variance_components[metric] = {
                'sigma_between_sq': sigma_between_sq,
                'sigma_within_sq': sigma_within_sq,
                'variance_ratio': sigma_between_sq / sigma_within_sq if sigma_within_sq > 0 else 0
            }

            print(f"\n{metric}:")
            print(f"  σ²_between (parameter uncertainty): {sigma_between_sq:.6e}")
            print(f"  σ²_within (stochastic noise): {sigma_within_sq:.6e}")
            print(f"  Ratio (signal/noise): {variance_components[metric]['variance_ratio']:.4f}")

        except Exception as e:
            print(f"\nWarning: Could not estimate variance for {metric}: {e}")
            continue

    # Calculate optimal n for a given precision target
    # For 95% CI width to be dominated by parameter uncertainty (not MC noise):
    # We want σ²_within/n << σ²_between
    # Rule of thumb: σ²_within/n ≤ 0.1 * σ²_between

    print(f"\n{'='*60}")
    print("Optimal Sample Size Recommendations:")
    print(f"{'='*60}")

    optimal_n_recommendations = {}

    for metric, components in variance_components.items():
        sigma_between_sq = components['sigma_between_sq']
        sigma_within_sq = components['sigma_within_sq']

        if sigma_between_sq > 0:
            # n such that σ²_within/n = 0.1 * σ²_between
            n_optimal_conservative = int(np.ceil(sigma_within_sq / (0.1 * sigma_between_sq)))
            # n such that σ²_within/n = 0.25 * σ²_between (less conservative)
            n_optimal_moderate = int(np.ceil(sigma_within_sq / (0.25 * sigma_between_sq)))

            optimal_n_recommendations[metric] = {
                'conservative': min(n_optimal_conservative, original_pop),
                'moderate': min(n_optimal_moderate, original_pop)
            }
        else:
            optimal_n_recommendations[metric] = {
                'conservative': n_inner_list[-1],
                'moderate': n_inner_list[-1]
            }

    # Get representative recommendation (use first key outcome)
    key_metrics = ['total_qalys', 'total_costs', 'total_dementia_onsets']
    recommended_n = None

    for key_metric in key_metrics:
        if key_metric in optimal_n_recommendations:
            rec = optimal_n_recommendations[key_metric]
            recommended_n = rec['moderate']
            print(f"\nFor {key_metric}:")
            print(f"  Conservative n (MC noise < 10% of PSA signal): {rec['conservative']:,}")
            print(f"  Moderate n (MC noise < 25% of PSA signal): {rec['moderate']:,}")
            break

    if recommended_n is None and optimal_n_recommendations:
        # Fall back to first available metric
        first_metric = list(optimal_n_recommendations.keys())[0]
        recommended_n = optimal_n_recommendations[first_metric]['moderate']
        print(f"\nRecommended n (based on {first_metric}): {recommended_n:,}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Use n ≈ {recommended_n:,} individuals per PSA iteration")
    print(f"This is {original_pop/recommended_n:.1f}x smaller than full population ({original_pop:,})")
    print(f"{'='*60}\n")

    return {
        'variance_components': variance_components,
        'optimal_n_recommendations': optimal_n_recommendations,
        'recommended_n': recommended_n,
        'original_population': original_pop,
        'pilot_n_outer': n_outer,
        'pilot_n_inner_tested': n_inner_list,
        'results_by_n': results_by_n
    }


def run_two_level_psa(base_config: dict,
                      psa_cfg: Optional[dict] = None,
                      *,
                      n_outer: int = 1000,
                      n_inner: Optional[int] = None,
                      variance_pilot_results: Optional[dict] = None,
                      collect_draw_level: bool = False,
                      seed: Optional[int] = None,
                      n_jobs: Optional[int] = None) -> dict:
    """
    Two-level PSA using O'Hagan et al. (2007) ANOVA method.

    Dramatically reduces computational cost by using reduced population size (n_inner)
    per parameter set while maintaining accuracy of 95% confidence intervals.

    Args:
        base_config: Base configuration
        psa_cfg: PSA configuration
        n_outer: Number of PSA iterations (parameter sets)
        n_inner: Population size per iteration (if None, uses full population)
        variance_pilot_results: Results from estimate_variance_components_pilot()
        collect_draw_level: Whether to return all draw-level data
        seed: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with PSA results including 95% CIs
    """
    psa_meta = copy.deepcopy(psa_cfg or base_config.get('psa') or {})

    # Determine n_inner
    original_pop = base_config.get('population', 33167098)

    if n_inner is None:
        if variance_pilot_results and 'recommended_n' in variance_pilot_results:
            n_inner = variance_pilot_results['recommended_n']
            print(f"\nUsing recommended n={n_inner:,} from pilot study")
        else:
            n_inner = original_pop
            print(f"\nNo pilot results provided. Using full population n={n_inner:,}")
            print("Consider running estimate_variance_components_pilot() first for efficiency!")

    # Create modified config with reduced population
    psa_config = _with_scaled_population_and_entrants(
        base_config,
        new_population=n_inner,
        original_population=original_pop
    )

    print(f"\n{'='*60}")
    print("TWO-LEVEL PSA (O'Hagan Method)")
    print(f"{'='*60}")
    print(f"Outer iterations (N): {n_outer}")
    print(f"Inner population (n): {n_inner:,}")
    print(f"Total simulated individuals: {n_outer * n_inner:,}")
    print(f"Reduction vs full PSA: {(original_pop * n_outer) / (n_inner * n_outer):.1f}x fewer individuals")
    print(f"{'='*60}\n")

    # Run standard PSA but with reduced population
    results = run_probabilistic_sensitivity_analysis(
        psa_config,
        psa_meta,
        collect_draw_level=collect_draw_level,
        seed=seed,
        n_jobs=n_jobs
    )

    # Add metadata about two-level design
    results['two_level_psa'] = {
        'n_outer': n_outer,
        'n_inner': n_inner,
        'original_population': original_pop,
        'reduction_factor': original_pop / n_inner,
        'variance_pilot_used': variance_pilot_results is not None
    }

    if variance_pilot_results:
        results['two_level_psa']['variance_components'] = variance_pilot_results.get('variance_components', {})

    return results


def export_psa_results_to_excel(psa_results: dict,
                                 path: str = "PSA_Results.xlsx",
                                 include_draws: bool = False) -> None:
    """
    Export PSA results (from run_probabilistic_sensitivity_analysis or run_two_level_psa) to Excel.

    Args:
        psa_results: Results dictionary from PSA functions
        path: Output Excel file path
        include_draws: If True and draw-level data available, include in separate sheet
    """
    print(f"\nExporting PSA results to {path}...")

    summary = psa_results.get('summary', {})
    if not summary:
        print("No PSA summary data available; nothing exported.")
        return

    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        # Sheet 1: PSA Summary (mean and 95% CI)
        summary_rows = []
        for metric, stats in summary.items():
            summary_rows.append({
                'Metric': metric,
                'Mean': stats.get('mean', np.nan),
                'Lower_95%_CI': stats.get('lower_95', np.nan),
                'Upper_95%_CI': stats.get('upper_95', np.nan),
                'CI_Width': stats.get('upper_95', np.nan) - stats.get('lower_95', np.nan)
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="PSA_Summary", index=False)

        # Sheet 2: PSA Metadata
        metadata_rows = []
        metadata_rows.append({'Parameter': 'PSA Iterations',
                             'Value': psa_results.get('iterations', 'N/A')})
        metadata_rows.append({'Parameter': 'Parallel Jobs Used',
                             'Value': psa_results.get('n_jobs_used', 'N/A')})

        # Add two-level PSA info if available
        if 'two_level_psa' in psa_results:
            two_level_info = psa_results['two_level_psa']
            metadata_rows.append({'Parameter': 'Method',
                                 'Value': "Two-Level PSA (O'Hagan et al. 2007)"})
            metadata_rows.append({'Parameter': 'Outer Iterations (N)',
                                 'Value': two_level_info.get('n_outer', 'N/A')})
            metadata_rows.append({'Parameter': 'Inner Population (n)',
                                 'Value': two_level_info.get('n_inner', 'N/A')})
            metadata_rows.append({'Parameter': 'Original Population',
                                 'Value': two_level_info.get('original_population', 'N/A')})
            metadata_rows.append({'Parameter': 'Reduction Factor',
                                 'Value': f"{two_level_info.get('reduction_factor', 0):.1f}x"})
            metadata_rows.append({'Parameter': 'Variance Pilot Used',
                                 'Value': 'Yes' if two_level_info.get('variance_pilot_used') else 'No'})
        else:
            metadata_rows.append({'Parameter': 'Method',
                                 'Value': 'Standard PSA (full population)'})

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_excel(writer, sheet_name="PSA_Metadata", index=False)

        # Sheet 3: Variance Components (if two-level PSA)
        if 'two_level_psa' in psa_results and 'variance_components' in psa_results['two_level_psa']:
            variance_comp = psa_results['two_level_psa']['variance_components']
            variance_rows = []

            for metric, components in variance_comp.items():
                variance_rows.append({
                    'Metric': metric,
                    'Sigma_Between_Sq (Parameter Uncertainty)': components.get('sigma_between_sq', np.nan),
                    'Sigma_Within_Sq (Stochastic Noise)': components.get('sigma_within_sq', np.nan),
                    'Signal_to_Noise_Ratio': components.get('variance_ratio', np.nan)
                })

            if variance_rows:
                variance_df = pd.DataFrame(variance_rows)
                variance_df.to_excel(writer, sheet_name="Variance_Components", index=False)

        # Sheet 4: Draw-level data (optional, can be large)
        if include_draws and 'draws' in psa_results:
            draws_df = psa_results['draws']
            if isinstance(draws_df, pd.DataFrame) and not draws_df.empty:
                # Limit to first 10,000 rows if very large (Excel has limits)
                if len(draws_df) > 10000:
                    print(f"  Warning: Draw data has {len(draws_df)} rows. Only exporting first 10,000 to Excel.")
                    print(f"  Consider using save_dataframe_compressed() for full data.")
                    draws_df = draws_df.head(10000)

                draws_df.to_excel(writer, sheet_name="All_Draws", index=False)
                print(f"  Included {len(draws_df)} draw-level records")

    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"Successfully exported PSA results to {path} ({file_size_mb:.2f} MB)")


# PSA Visualization Functions (with confidence intervals)

def plot_psa_time_series_with_ci(baseline_results: dict,
                                   psa_results: dict,
                                   metric_name: str,
                                   ylabel: str,
                                   title: str,
                                   save_path: str,
                                   show: bool = False,
                                   scale_factor: float = 1.0) -> None:
    """
    Generic function to plot PSA time series with mean line and shaded 95% CI.

    Args:
        baseline_results: Single baseline model run results
        psa_results: PSA results with draw-level data
        metric_name: Name of metric in summary DataFrames
        ylabel: Y-axis label
        title: Plot title
        save_path: Where to save the plot
        show: Whether to display the plot
        scale_factor: Multiply values by this (e.g., 1e-6 for millions)
    """
    # Check if we have draw-level data
    if 'draws' not in psa_results or psa_results['draws'] is None:
        print(f"No draw-level data available for {title}. Cannot plot CI.")
        return

    # Get baseline time series
    baseline_df = summaries_to_dataframe(baseline_results)
    if baseline_df.empty or metric_name not in baseline_df.columns:
        print(f"Baseline data missing {metric_name}. Cannot plot {title}.")
        return

    # Extract time information
    if 'calendar_year' in baseline_df.columns:
        time_col = 'calendar_year'
        time_label = 'Calendar Year'
    elif 'time_step' in baseline_df.columns:
        time_col = 'time_step'
        time_label = 'Time Step'
    else:
        print(f"No time column found. Cannot plot {title}.")
        return

    time_points = baseline_df[time_col].values
    n_time_points = len(time_points)

    # The PSA draws contain scalar summaries, not time series
    # We need to run a helper to get time series from individual model runs
    # For now, we'll work with what we have - this is a limitation
    # Let me create a workaround by re-running a subset of draws to get time series

    print(f"Note: Full time-series PSA plotting requires storing time-series data from each draw.")
    print(f"      Currently showing baseline only for {title}.")
    print(f"      For full PSA visualization, consider storing draw-level time series.")

    # Plot baseline as a single line for now
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_values = baseline_df[metric_name].values * scale_factor

    ax.plot(time_points, baseline_values, 'b-', linewidth=2, label='Baseline')

    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    save_or_show(save_path, show, title)


def plot_psa_incidence_with_ci(psa_results: dict,
                                 base_config: dict,
                                 save_path: str = "plots/psa_incidence_ci.png",
                                 show: bool = False,
                                 n_sample_draws: int = 100) -> None:
    """
    Plot dementia incidence over time with mean and 95% CI shaded region.

    Since PSA draws don't automatically store time-series data, this function
    re-runs a sample of parameter sets to generate the time series.

    Args:
        psa_results: PSA results (must include 'draws' with parameters)
        base_config: Base configuration
        save_path: Output file path
        show: Whether to display plot
        n_sample_draws: Number of draws to sample for time series (default 100)
    """
    if 'draws' not in psa_results or psa_results['draws'] is None:
        print("No draw-level data available. Cannot plot PSA incidence with CI.")
        return

    draws_df = psa_results['draws']

    if len(draws_df) < n_sample_draws:
        n_sample_draws = len(draws_df)

    print(f"\nGenerating incidence time series from {n_sample_draws} PSA draws...")
    print("This may take a few minutes...")

    # Sample draws
    sampled_draws = draws_df.sample(n=n_sample_draws, random_state=42)

    # We need to reconstruct configs and re-run - this is expensive
    # For now, let's create a simpler version that uses stored summary metrics
    print("Note: Full time-series PSA requires re-running models or storing time-series per draw.")
    print("      Consider using summary metrics (total incidence) or implementing time-series storage.")
    print("      Plotting not implemented for time-series with CI yet.")


def plot_psa_summary_metrics(psa_results: dict,
                              save_path: str = "plots/psa_summary_with_ci.png",
                              show: bool = False) -> None:
    """
    Plot key summary metrics from PSA as bar charts with error bars (95% CI).

    This plots scalar summaries like total costs, total QALYs, total incidence.

    Args:
        psa_results: PSA results dictionary
        save_path: Output file path
        show: Whether to display plot
    """
    summary = psa_results.get('summary', {})
    if not summary:
        print("No PSA summary data available. Cannot plot.")
        return

    # Select key metrics to plot
    metrics_to_plot = {
        'total_costs': ('Total Costs', '£', 1e-9),  # Billions
        'total_qalys': ('Total QALYs', 'QALYs', 1e-6),  # Millions
        'total_dementia_onsets': ('Total Dementia Onsets', 'Cases', 1e-3),  # Thousands
    }

    # Filter to available metrics
    available_metrics = {k: v for k, v in metrics_to_plot.items() if k in summary}

    if not available_metrics:
        print("No plottable metrics found in PSA summary.")
        return

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))

    if len(available_metrics) == 1:
        axes = [axes]

    for ax, (metric_key, (label, unit, scale)) in zip(axes, available_metrics.items()):
        stats = summary[metric_key]

        mean = stats['mean'] * scale
        lower = stats['lower_95'] * scale
        upper = stats['upper_95'] * scale
        error = [[mean - lower], [upper - mean]]

        ax.bar([0], [mean], yerr=error, capsize=10, color='steelblue', alpha=0.7, width=0.5)

        ax.set_ylabel(f"{label} ({unit})", fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        ax.text(0, mean, f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(0, lower, f'{lower:.2f}', ha='center', va='top', fontsize=9, style='italic')
        ax.text(0, upper, f'{upper:.2f}', ha='center', va='bottom', fontsize=9, style='italic')

    plt.suptitle('PSA Summary Metrics with 95% Confidence Intervals', fontsize=16, fontweight='bold', y=1.02)

    save_or_show(save_path, show, "PSA Summary Metrics")


def plot_psa_tornado(psa_results: dict,
                      metric: str = 'total_qalys',
                      save_path: str = "plots/psa_tornado.png",
                      show: bool = False,
                      top_n: int = 10) -> None:
    """
    Create a tornado diagram showing parameter sensitivity (placeholder for future implementation).

    Args:
        psa_results: PSA results
        metric: Which outcome metric to analyze
        save_path: Output path
        show: Whether to display
        top_n: Number of top parameters to show
    """
    print("Tornado diagram requires storing parameter values per draw.")
    print("This is a placeholder for future implementation.")
    print("Consider correlation analysis between parameters and outcomes.")


# Visuals

def save_or_show(save_path, show=False, label="plot"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved {label} to {save_path.resolve()}")


def plot_ad_prevalence(model_results, save_path="plots/ad_prevalence.png", show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping prevalence plot.")
        return

    ad_cols = [f'stage_{stage}' for stage in ('mild', 'moderate', 'severe')]
    for col in ad_cols:
        if col not in df.columns:
            df[col] = 0

    df['ad_cases'] = df[ad_cols].sum(axis=1)
    # Use alive population each cycle as denominator; fall back to total if alive not tracked
    denom_series = df['population_alive'] if 'population_alive' in df.columns else df['population_total']
    denom_series = denom_series.replace(0, np.nan)
    df['prevalence_pct'] = 100.0 * df['ad_cases'] / denom_series
    df['prevalence_pct'] = df['prevalence_pct'].fillna(0.0)
    x = df['calendar_year'] if 'calendar_year' in df.columns else df['time_step']

    max_prev = float(df['prevalence_pct'].max()) if not df.empty else 0.0
    upper_limit = 5.0 if max_prev <= 0 else max(5.0, max_prev * 1.1)

    plt.figure()
    plt.plot(x, df['prevalence_pct'], marker='o')
    plt.ylabel("Alzheimer's prevalence (%)")
    plt.xlabel('Year' if 'calendar_year' in df.columns else 'Time step (years)')
    plt.ylim(0, upper_limit)
    plt.title("Alzheimer's prevalence over time")
    save_or_show(save_path, show, label="prevalence plot")


def plot_ad_incidence(model_results,
                      save_path="plots/ad_incidence.png",
                      show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping incidence plot.")
        return
    if 'incident_onsets' not in df.columns:
        print("Incident onset counts not available; skipping incidence plot.")
        return

    df = df.copy()
    axis_col: Optional[str]
    if 'calendar_year' in df.columns:
        axis_col = 'calendar_year'
    elif 'time_step' in df.columns:
        axis_col = 'time_step'
    else:
        print("No time axis available; skipping incidence plot.")
        return

    df = df.sort_values(axis_col).reset_index(drop=True)

    baseline_value = df[axis_col].iloc[0]
    post_baseline_df = df.loc[df[axis_col] != baseline_value].copy()
    if post_baseline_df.empty:
        print("Only baseline data available; skipping incidence plot.")
        return
    df = post_baseline_df

    df['incident_onsets'] = df['incident_onsets'].fillna(0)
    if 'incidence_per_1000_alive' not in df.columns:
        alive = df.get('population_alive', pd.Series(dtype=float)).replace(0, np.nan)
        df['incidence_per_1000_alive'] = (df['incident_onsets'] / alive) * 1000.0
        df['incidence_per_1000_alive'] = df['incidence_per_1000_alive'].fillna(0.0)
    else:
        df['incidence_per_1000_alive'] = df['incidence_per_1000_alive'].fillna(0.0)

    x = df[axis_col]
    counts = df['incident_onsets']
    rates = df['incidence_per_1000_alive']

    fig, ax_count = plt.subplots()
    ax_count.bar(x, counts, width=0.6, alpha=0.4, label="New onsets (count)")
    if len(df) >= 2:
        x_numeric = pd.to_numeric(x, errors='coerce')
        if x_numeric.isna().any():
            x_numeric = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        x_numeric = x_numeric.astype(float)
        coeffs = np.polyfit(x_numeric.to_numpy(dtype=float), counts.to_numpy(dtype=float), 1)
        trend_counts = np.polyval(coeffs, x_numeric.to_numpy(dtype=float))
        ax_count.plot(x, trend_counts, color="tab:green", linestyle="--", linewidth=2, label="New onsets trend")
    ax_count.set_ylabel("New onsets (count)")

    ax_rate = ax_count.twinx()
    ax_rate.plot(x, rates, color="tab:red", marker='o', label="Incidence per 1,000 alive")
    ax_rate.set_ylabel("Incidence per 1,000 alive")

    max_count = float(counts.max()) if not counts.empty else 0.0
    max_rate = float(rates.max()) if not rates.empty else 0.0
    count_upper = max_count * 1.1 if max_count > 0 else 1.0
    rate_upper = max_rate * 1.1 if max_rate > 0 else 1.0
    ax_count.set_ylim(0.0, count_upper)
    ax_rate.set_ylim(0.0, rate_upper)

    ax_count.set_xlabel('Year' if 'calendar_year' in df.columns else 'Time step (years)')
    ax_count.set_title("Alzheimer's incidence over time")

    handles1, labels1 = ax_count.get_legend_handles_labels()
    handles2, labels2 = ax_rate.get_legend_handles_labels()
    ax_count.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    save_or_show(save_path, show, label="incidence plot")


def plot_age_specific_ad_cases(model_results,
                               age_bands: Optional[List[Tuple[int, Optional[int]]]] = None,
                               save_path: str = "plots/ad_age_specific_cases.png",
                               show: bool = False) -> None:
    """Stacked bar chart of dementia cases by age band for each simulated year."""
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping age-specific dementia plot.")
        return

    bands = age_bands if age_bands is not None else REPORTING_AGE_BANDS
    if not bands:
        print("No age bands configured for reporting; skipping age-specific dementia plot.")
        return

    index_field = 'calendar_year' if 'calendar_year' in df.columns else 'time_step'
    x_labels = df[index_field].tolist()
    x_positions = np.arange(len(x_labels))

    plt.figure()
    stacked_bottom = np.zeros(len(x_labels), dtype=float)

    for band in bands:
        column_name = f"ad_cases_age_{age_band_key(band)}"
        if column_name not in df.columns:
            df[column_name] = 0
        counts = df[column_name].fillna(0.0).to_numpy(dtype=float)
        plt.bar(x_positions,
                counts,
                bottom=stacked_bottom,
                width=0.6,
                label=age_band_label(band))
        stacked_bottom += counts

    plt.xticks(x_positions, x_labels, rotation=90, ha='center')
    plt.ylabel("Estimated dementia cases (count)")
    plt.xlabel('Year' if index_field == 'calendar_year' else 'Time step (years)')
    plt.title("Age-specific dementia cases over time")
    plt.legend(title="Age band")

    save_or_show(save_path, show, label="age-specific dementia cases plot")


def plot_dementia_prevalence_by_stage(model_results,
                                      stages: Optional[List[str]] = None,
                                      save_path: str = "plots/ad_stage_prevalence.png",
                                      show: bool = False) -> None:
    """Stacked bar chart showing dementia prevalence by stage for each simulated year."""
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping stage-specific prevalence plot.")
        return

    tracked_stages = stages if stages is not None else ['mild', 'moderate', 'severe']
    if not tracked_stages:
        print("No stages provided for prevalence plotting; skipping stage-specific prevalence plot.")
        return

    index_field = 'calendar_year' if 'calendar_year' in df.columns else 'time_step'
    x_labels = df[index_field].tolist()
    x_positions = np.arange(len(x_labels))

    denom = df['population_alive'] if 'population_alive' in df.columns else df['population_total']
    denom = denom.replace(0, np.nan)

    plt.figure()
    stacked_bottom = np.zeros(len(x_labels), dtype=float)

    for stage in tracked_stages:
        column_name = f'stage_{stage}'
        if column_name not in df.columns:
            df[column_name] = 0
        prevalence_series = (df[column_name].fillna(0.0) / denom).replace([np.inf, -np.inf], np.nan)
        prevalence_pct = prevalence_series.fillna(0.0) * 100.0
        values = prevalence_pct.to_numpy(dtype=float)

        plt.bar(
            x_positions,
            values,
            bottom=stacked_bottom,
            width=0.6,
            label=stage.replace('_', ' ').title()
        )
        stacked_bottom += values

    plt.xticks(x_positions, x_labels, rotation=90, ha='center')
    plt.ylabel("Share of alive population (%)")
    plt.xlabel('Year' if index_field == 'calendar_year' else 'Time step (years)')
    plt.title("Dementia prevalence by stage over time")
    y_max = stacked_bottom.max() if stacked_bottom.size else 0.0
    y_upper = max(5.0, y_max * 1.05) if y_max > 0 else 5.0
    plt.ylim(0, min(100.0, y_upper))
    plt.legend(title="Stage")

    save_or_show(save_path, show, label="stage-specific prevalence plot")


def plot_survival_curve(model_results, save_path='plots/survival_curve.png', show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping survival plot.")
        return

    series_name = 'baseline_alive' if 'baseline_alive' in df.columns else 'population_alive'
    if series_name not in df.columns or df[series_name].fillna(0).iloc[0] == 0:
        print("No individuals at baseline; cannot plot survival.")
        return

    alive = df[series_name].fillna(0)
    baseline_alive = alive.iloc[0]
    survival = alive / baseline_alive if baseline_alive else alive
    x = df['calendar_year'] if 'calendar_year' in df.columns else df['time_step']

    plt.figure()
    plt.step(x, survival, where='post')
    plt.ylim(0, 1.01)
    plt.xlabel("Year" if 'calendar_year' in df.columns else "Time step (years)")
    plt.ylabel("Survival proportion")
    title = "Survival curve"
    if series_name == 'baseline_alive':
        title += " (baseline cohort)"
    plt.title(title)
    save_or_show(save_path, show, label="survival curve")


def _kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(times)
    times = times[order]
    events = events[order]

    event_mask = events == 1
    unique_event_times = np.unique(times[event_mask])

    timeline = [0.0]
    survival = [1.0]
    current = 1.0

    for t in unique_event_times:
        at_risk = np.sum(times >= t)
        events_at_t = np.sum((times == t) & event_mask)
        if at_risk == 0:
            continue
        current *= (1.0 - events_at_t / at_risk)
        timeline.append(float(t))
        survival.append(current)

    max_time = float(times.max())
    if timeline[-1] < max_time:
        timeline.append(max_time)
        survival.append(current)

    return np.array(timeline), np.array(survival)


def plot_survival_by_baseline_stage(model_results,
                                    save_path='plots/survival_by_stage.png',
                                    show=False):
    records = model_results.get('individual_survival', []) if isinstance(model_results, dict) else []
    if not records:
        print("No individual-level survival records available; skipping Kaplan-Meier plot.")
        return

    survival_df = pd.DataFrame(records)
    if survival_df.empty or 'baseline_stage' not in survival_df or 'time' not in survival_df:
        print("Incomplete survival records; skipping Kaplan-Meier plot.")
        return

    plt.figure()
    unique_stages = [stage for stage in DEMENTIA_STAGES if stage in survival_df['baseline_stage'].unique()]
    for stage in unique_stages:
        stage_df = survival_df[survival_df['baseline_stage'] == stage]
        if stage_df.empty:
            continue
        times = stage_df['time'].to_numpy(dtype=float)
        events = stage_df['event'].to_numpy(dtype=int)
        t_points, surv = _kaplan_meier_curve(times, events)
        if t_points.size == 0:
            continue
        label = stage.replace('_', ' ').title()
        plt.step(t_points, surv, where='post', label=label)

    if not plt.gca().has_data():
        print("No valid Kaplan-Meier curves to plot.")
        plt.close()
        return

    plt.ylim(0, 1.01)
    plt.xlabel("Time since entry (years)")
    plt.ylabel("Survival proportion")
    plt.title("Kaplan-Meier survival by baseline dementia stage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="Kaplan-Meier survival by stage")


def plot_baseline_age_hist(model_results, bins=None, save_path="plots/baseline_age_hist.png", show=False):
    age_dist = model_results.get('initial_age_distribution', {}) if isinstance(model_results, dict) else {}
    if not age_dist:
        print("No baseline age distribution available; skipping histogram.")
        return

    ages = np.array(sorted(age_dist.keys()))
    counts = np.array([age_dist[a] for a in ages], dtype=float)

    plt.figure()
    if isinstance(bins, int) and bins > 0:
        hist, bin_edges = np.histogram(ages, bins=bins, weights=counts)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        plt.bar(centers, hist, width=widths, align='center')
    else:
        plt.bar(ages, counts, width=0.9, align='center')

    plt.xlabel("Age at baseline (years)")
    plt.ylabel("Count")
    plt.title("Baseline age distribution")
    save_or_show(save_path, show, label="baseline age histogram")


def plot_age_at_death_hist(model_results, bins=20,
                           save_path='plots/age_at_death_hist.png', show=False):
    age_dist = model_results.get('age_at_death_distribution', {}) if isinstance(model_results, dict) else {}
    if not age_dist:
        print("No deaths to plot (yet).")
        return

    ages = np.array(sorted(age_dist.keys()))
    counts = np.array([age_dist[a] for a in ages], dtype=float)

    plt.figure()
    if isinstance(bins, int) and bins > 0:
        hist, bin_edges = np.histogram(ages, bins=bins, weights=counts)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        plt.bar(centers, hist, width=widths, align='center')
    else:
        plt.bar(ages, counts, width=0.9, align='center')

    plt.xlabel("Age at death (years)")
    plt.ylabel("Count")
    plt.title("Distribution of age at death")
    save_or_show(save_path, show, label="age-at-death histogram")


def _compute_cost_qaly_metrics(model_results):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        return df

    df = df.copy()
    denom = df['population_total'].replace(0, np.nan)
    df['cumulative_costs_nhs_mean'] = df['total_costs_nhs'] / denom
    df['cumulative_costs_informal_mean'] = df['total_costs_informal'] / denom
    df['cumulative_qalys_patient_mean'] = df['total_qalys_patient'] / denom
    df['cumulative_qalys_caregiver_mean'] = df['total_qalys_caregiver'] / denom

    df['total_costs_mean'] = df['cumulative_costs_nhs_mean'] + df['cumulative_costs_informal_mean']
    df['total_qalys_mean'] = df['cumulative_qalys_patient_mean'] + df['cumulative_qalys_caregiver_mean']
    df['total_costs_total'] = df['total_costs_nhs'] + df['total_costs_informal']
    df['total_qalys_total'] = df['total_qalys_patient'] + df['total_qalys_caregiver']
    df.fillna(0, inplace=True)
    return df


def plot_costs_per_person_over_time(model_results,
                                    save_path="plots/costs_per_person_over_time.png",
                                    show=False):
    """Mean cumulative costs per person (NHS, Informal, Total) over time."""
    agg = _compute_cost_qaly_metrics(model_results)
    if agg.empty:
        print("No summary data available; skipping cost plot.")
        return

    x = agg['calendar_year'] if 'calendar_year' in agg.columns else agg['time_step']

    plt.figure()
    plt.plot(x, agg['cumulative_costs_nhs_mean'], marker="o", linestyle="-", label="NHS (GBP/person)")
    plt.plot(x, agg['cumulative_costs_informal_mean'], marker="o", linestyle="--", label="Informal (GBP/person)")
    plt.plot(x, agg['total_costs_mean'], marker="o", linestyle="-.", label="Total (GBP/person)")
    plt.xlabel('Year' if 'calendar_year' in agg.columns else 'Time step (years)')
    plt.ylabel('Mean cumulative cost per person (GBP)')
    plt.title('Mean cumulative costs per person over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="costs per person")


def plot_qalys_per_person_over_time(model_results,
                                    save_path="plots/qalys_per_person_over_time.png",
                                    show=False):
    """Mean cumulative QALYs per person (Patient, Caregiver, Total) over time."""
    agg = _compute_cost_qaly_metrics(model_results)
    if agg.empty:
        print("No summary data available; skipping QALY plot.")
        return

    x = agg['calendar_year'] if 'calendar_year' in agg.columns else agg['time_step']

    plt.figure()
    plt.plot(x, agg['cumulative_qalys_patient_mean'], marker="o", linestyle="-", label="Patient QALYs/person")
    plt.plot(x, agg['cumulative_qalys_caregiver_mean'], marker="o", linestyle="--", label="Caregiver QALYs/person")
    plt.plot(x, agg['total_qalys_mean'], marker="o", linestyle="-.", label="Total QALYs/person")
    plt.xlabel('Year' if 'calendar_year' in agg.columns else 'Time step (years)')
    plt.ylabel('Mean cumulative QALYs per person')
    plt.title('Mean cumulative QALYs per person over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="QALYs per person")

# ---------- Hazard-ratio visualisers (use base_onset_probability path) ----------

def _mock_person(age: int, risk_flags: dict, sex: str = "female") -> dict:
    return {
        "age": age,
        "sex": _canonical_sex_label(sex),
        "risk_factors": risk_flags,
        "dementia_stage": "cognitively_normal",
        "time_in_stage": 0,
        "living_setting": "home",
        "alive": True,
    }

def _onset_hazard_from_base_prob(config: dict,
                                 age: int,
                                 risk_flags: dict,
                                 sex: str = "female") -> float:
    """
    Use base_onset_probability (per-cycle) -> hazard, then apply age & risk-factor HRs.
    This path is used when 'normal_to_mild' is NOT defined in stage_transition_durations.
    """
    dt = config["time_step_years"]
    p0 = config.get("base_onset_probability", 0.0)
    h0 = prob_to_hazard(p0, dt=dt)
    age_hr = get_age_hr_for_transition(age, config, "onset")  # CHANGED: parametric/banded
    person = _mock_person(age, risk_flags, sex=sex)
    h = apply_hazard_ratios(
        h0,
        person["risk_factors"],
        config["risk_factors"],
        "onset",
        age_hr,
        person["age"],
        person.get("sex", "unspecified"),
        config,  # NEW
    )
    return h

def plot_onset_hazard_vs_age_from_base_prob(config: dict,
                                            risk_profiles: dict = None,
                                            ages: np.ndarray | list = None,
                                            save_path: str = "plots/onset_hazard_vs_age.png",
                                            show: bool = False):
    """Plot the adjusted ONSET hazard vs age using base_onset_probability (no duration needed)."""
    if risk_profiles is None:
        risk_profiles = {
            "None": {},
            "Periodontal": {"periodontal_disease": True},
            "Smoking": {"smoking": True},
            "Smoking + Periodontal": {"smoking": True, "periodontal_disease": True},
        }
    if ages is None:
        ages = np.arange(50, 91, 5)

    plt.figure()
    for label, flags in risk_profiles.items():
        if isinstance(flags, dict):
            profile_sex = _canonical_sex_label(flags.get('sex', 'female'))
            risk_flags = {k: v for k, v in flags.items() if k != 'sex'}
        else:
            profile_sex = 'female'
            risk_flags = {}
        hazards = [
            _onset_hazard_from_base_prob(config, age, risk_flags, sex=profile_sex)
            for age in ages
        ]
        plt.plot(ages, hazards, marker="o", label=label)

    plt.xlabel("Age (years)")
    plt.ylabel("Onset hazard (per year)")
    plt.title("Onset hazard vs age (from base_onset_probability)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show(save_path, show, label="onset hazard vs age")

def plot_onset_probability_vs_age_from_base_prob(config: dict,
                                                 risk_profiles: dict = None,
                                                 ages: np.ndarray | list = None,
                                                 save_path: str = "plots/onset_probability_vs_age.png",
                                                 show: bool = False):
    """Same as above, but converted to per-cycle probabilities for your Δt."""
    if risk_profiles is None:
        risk_profiles = {
            "None": {},
            "Periodontal": {"periodontal_disease": True},
            "Smoking": {"smoking": True},
            "Smoking + Periodontal": {"smoking": True, "periodontal_disease": True},
        }
    if ages is None:
        ages = np.arange(50, 91, 5)

    dt = config["time_step_years"]
    plt.figure()
    for label, flags in risk_profiles.items():
        if isinstance(flags, dict):
            profile_sex = _canonical_sex_label(flags.get('sex', 'female'))
            risk_flags = {k: v for k, v in flags.items() if k != 'sex'}
        else:
            profile_sex = 'female'
            risk_flags = {}
        probs = [
            hazard_to_prob(
                _onset_hazard_from_base_prob(config, age, risk_flags, sex=profile_sex),
                dt=dt,
            )
            for age in ages
        ]
        plt.plot(ages, probs, marker="o", label=label)

    plt.xlabel("Age (years)")
    plt.ylabel(f"Per-cycle probability (Δt={dt} y)")
    plt.title("Onset probability vs age (from base_onset_probability)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show(save_path, show, label="onset probability vs age")

# ===== Constant-hazard diagnostics (add below your imports) =====

def check_constant_hazard_from_transitions(model_results: dict,
                                           from_stage: str,
                                           to_stage: str,
                                           *,
                                           dt: float = 1.0,
                                           tolerance: float = 0.05) -> dict:
    """
    Uses transition_history to test if the per-step hazard for a specific transition is ~constant over time.
    Returns: dict with mean_hazard, max_relative_deviation, within_tolerance, and a DataFrame of interval hazards.
    """
    th = model_results.get('transition_history', {}) or {}
    rows = []
    for t in sorted(th.keys()):
        payload = th[t] or {}
        starts = payload.get('stage_start_counts', {}) or {}
        trans  = payload.get('transition_counts', {}) or {}
        start_n = float(starts.get(from_stage, 0.0))
        if start_n <= 0:
            continue
        trans_n  = float(trans.get((from_stage, to_stage), 0.0))
        # per-interval probability for that step
        p = trans_n / start_n
        p = max(0.0, min(1.0, p))
        # convert probability -> hazard under exponential assumption for interval length dt
        h = -math.log(max(1e-12, 1.0 - p)) / dt
        rows.append({
            'time_step': t,
            'start_count': start_n,
            'transition_count': trans_n,
            'probability': p,
            'interval_hazard': h,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            'mean_hazard': float('nan'),
            'max_relative_deviation': float('nan'),
            'within_tolerance': False,
            'intervals': df,
        }

    mean_h = float(df['interval_hazard'].mean())
    if mean_h > 0:
        df['relative_deviation'] = (df['interval_hazard'] - mean_h) / mean_h
        max_dev = float(df['relative_deviation'].abs().max())
        within  = bool(max_dev <= tolerance)
    else:
        df['relative_deviation'] = float('nan')
        max_dev = float('nan')
        within  = False

    return {
        'mean_hazard': mean_h,
        'max_relative_deviation': max_dev,
        'within_tolerance': within,
        'intervals': df,
    }


def plot_transition_interval_hazards(intervals_df: pd.DataFrame,
                                     *,
                                     title: str,
                                     save_path: str = "plots/transition_interval_hazards.png",
                                     show: bool = False) -> None:
    """
    Simple line plot of per-step interval hazards with a horizontal line at the mean.
    Saves to file using your existing save_or_show() helper.
    """
    if intervals_df is None or intervals_df.empty:
        print(f"No intervals to plot for: {title}")
        return

    # Choose x from calendar_year if available via summaries; otherwise use time_step
    x = intervals_df['time_step']
    y = intervals_df['interval_hazard']
    mean_y = float(y.mean())

    plt.figure()
    plt.plot(x, y, marker='o', label='Interval hazard')
    plt.axhline(mean_y, linestyle='--', linewidth=1.5, label=f"Mean = {mean_y:.4f}")
    plt.xlabel("Time step (years)")
    plt.ylabel("Interval hazard (/year)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_or_show(save_path, show, label=title)


def run_constant_hazard_diagnostics(model_results: dict,
                                    config: dict,
                                    *,
                                    tolerance: float = 0.05,
                                    show_plots: bool = False) -> None:
    if not config.get('enable_constant_hazard_checks', False):
        return
    """
    Runs:
      (1) Whole-cohort survival constant-hazard check (your existing helper)
      (2) Per-transition constant-hazard checks for key transitions
    Prints a concise summary and saves plots under plots/.
    """
    dt = float(config.get('time_step_years', 1.0))

    # --- (1) Whole-cohort survival constant-hazard check ---
    try:
        cohort_check = check_constant_hazard_from_model(
            model_results,
            tolerance=tolerance,
            cohort='baseline',          # change to 'population' if you prefer
            use_calendar_year=True
        )
        print("\n[Whole-cohort survival] Constant-hazard check")
        print(f"  Time axis: {cohort_check.get('time_axis')}")
        print(f"  Mean hazard: {cohort_check['mean_hazard']:.6f} /year")
        print(f"  Max relative deviation: {cohort_check['max_relative_deviation']:.2%}")
        print(f"  Within tolerance (±{tolerance:.0%}): {cohort_check['within_tolerance']}")
        # Optional quick plot of piecewise hazards reconstructed from survival:
        intervals = cohort_check.get('intervals')
        if intervals is not None and not intervals.empty:
            # reuse the generic plotting helper by renaming columns to match expectation
            tmp = intervals.rename(columns={'time_start': 'time_step',
                                            'interval_hazard': 'interval_hazard'}).copy()
            # use time_end index as proxy for step index to avoid duplicate x
            tmp['time_step'] = np.arange(len(tmp), dtype=float)
            plot_transition_interval_hazards(
                tmp[['time_step', 'interval_hazard']].copy(),
                title="Whole-cohort survival: piecewise hazards",
                save_path="plots/whole_cohort_piecewise_hazards.png",
                show=show_plots
            )
    except ValueError as exc:
        print(f"\n[Whole-cohort survival] Check unavailable: {exc}")

    # --- (2) Per-transition constant-hazard checks ---
    transitions_to_test = [
        ('mild', 'moderate'),
        ('moderate', 'severe'),
        ('severe', 'death'),
    ]

    for (frm, to) in transitions_to_test:
        res = check_constant_hazard_from_transitions(
            model_results, frm, to, dt=dt, tolerance=tolerance
        )
        print(f"\n[{frm} → {to}] Constant-hazard check")
        print(f"  Mean hazard: {res['mean_hazard']:.6f} /year")
        print(f"  Max relative deviation: {res['max_relative_deviation']:.2%}")
        print(f"  Within tolerance (±{tolerance:.0%}): {res['within_tolerance']}")

        # Plot per-step interval hazards
        plot_transition_interval_hazards(
            res['intervals'],
            title=f"{frm.title()} → {to.title()} interval hazards",
            save_path=f"plots/{frm}_to_{to}_interval_hazards.png",
            show=show_plots
        )

# Lifetime risk visuals

def plot_lifetime_risk_by_entry_age(model_results,
                                    save_path: str = "plots/lifetime_risk_by_age.png",
                                    show: bool = False,
                                    *,
                                    include_all: bool = False,
                                    min_population: int = 25) -> None:
    """Line chart of lifetime dementia risk (%) by entry age."""
    data_key = 'lifetime_risk_by_entry_age_all' if include_all else 'lifetime_risk_by_entry_age'
    records = model_results.get(data_key, []) if isinstance(model_results, dict) else []
    if not records:
        target = "all entrants" if include_all else "cognitively normal entrants"
        print(f"No lifetime risk data for {target}; skipping lifetime risk plot.")
        return

    df = pd.DataFrame(records)
    if df.empty:
        print("Lifetime risk data frame is empty; skipping lifetime risk plot.")
        return

    if min_population > 1:
        df = df[df['population'] >= min_population]
        if df.empty:
            print(f"No ages meet the minimum population threshold ({min_population}); skipping lifetime risk plot.")
            return

    df = df.sort_values('entry_age').reset_index(drop=True)
    df['lifetime_risk_pct'] = df['lifetime_risk'] * 100.0

    plt.figure()
    plt.plot(df['entry_age'], df['lifetime_risk_pct'], marker='o')
    plt.xlabel("Age at entry (years)")
    plt.ylabel("Lifetime dementia risk (%)")
    title_suffix = " (all entrants)" if include_all else " (baseline cognitively normal)"
    plt.title(f"Lifetime dementia risk by entry age{title_suffix}")
    max_risk_pct = float(df['lifetime_risk_pct'].max()) if not df.empty else 0.0
    upper_limit = max(5.0, max_risk_pct * 1.1) if max_risk_pct > 0 else 5.0
    plt.ylim(0, upper_limit)
    plt.grid(True, alpha=0.3)

    save_or_show(save_path, show, label="lifetime dementia risk plot")

# Export

def export_results_to_excel(model_results, path="PD_AD_PD50.xlsx"):
    summaries = summaries_to_dataframe(model_results)
    if summaries.empty:
        print("No summary data available; nothing exported.")
        return

    calendar_lookup: Dict[int, Any] = {}
    if 'time_step' in summaries.columns:
        if 'calendar_year' in summaries.columns:
            calendar_lookup = dict(zip(summaries['time_step'], summaries['calendar_year']))
        else:
            calendar_lookup = dict(zip(summaries['time_step'], summaries['time_step']))

    # Ensure the output directory exists to avoid ExcelWriter errors
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(path) as writer:
        summaries.to_excel(writer, sheet_name="Summary", index=False)

        flow_cols = [
            'time_step',
            'calendar_year',
            'year_qalys_patient',
            'year_qalys_caregiver',
            'year_qalys_total',
            'year_costs_nhs',
            'year_costs_informal',
            'year_costs_societal',
        ]
        if set(['time_step', 'calendar_year']).issubset(summaries.columns):
            existing_flow_cols = [c for c in flow_cols if c in summaries.columns]
            if len(existing_flow_cols) >= 3:
                flows_df = summaries[existing_flow_cols].copy().sort_values('time_step')
                flows_df.to_excel(writer, sheet_name="YearlyFlows", index=False)

        risk_cols = [c for c in summaries.columns if str(c).startswith('risk_prev_')]
        if risk_cols and set(['time_step', 'calendar_year']).issubset(summaries.columns):
            risk_df = summaries[['time_step', 'calendar_year', *risk_cols]].copy().sort_values('time_step')
            risk_df.to_excel(writer, sheet_name="RiskFactorPrevalence", index=False)

        target_years = [2025, 2030, 2035, 2040]
        severity_columns = [col for col in ('stage_mild', 'stage_moderate', 'stage_severe') if col in summaries.columns]
        if severity_columns and 'calendar_year' in summaries.columns:
            severity_df = (
                summaries[summaries['calendar_year'].isin(target_years)]
                [['calendar_year', *severity_columns]]
                .copy()
            )
            rename_map = {
                'stage_mild': 'mild_cases',
                'stage_moderate': 'moderate_cases',
                'stage_severe': 'severe_cases',
            }
            severity_df.rename(columns=rename_map, inplace=True)
            case_columns = [rename_map.get(col, col) for col in severity_columns]
            severity_df = pd.DataFrame({'calendar_year': target_years}).merge(
                severity_df, on='calendar_year', how='left'
            )
            for col in case_columns:
                if col not in severity_df.columns:
                    severity_df[col] = np.nan
            severity_df['total_dementia_cases'] = severity_df[case_columns].sum(axis=1, min_count=1)
            ordered_columns = ['calendar_year', *case_columns, 'total_dementia_cases']
            severity_df = severity_df[ordered_columns]
            severity_df.to_excel(writer, sheet_name="SeverityPrevalence", index=False)

        # Add high-level incidence metrics for quick reference in the workbook.
        incidence_cols = {'incident_onsets', 'population_alive'}
        if incidence_cols.issubset(summaries.columns):
            cases_cols = [col for col in ['time_step', 'calendar_year', 'incident_onsets', 'population_alive'] if col in summaries.columns]
            cases_df = summaries[cases_cols].copy().sort_values('time_step' if 'time_step' in cases_cols else cases_cols[0])
            denom = cases_df['population_alive'].replace(0, np.nan)
            cases_df['cases_per_1k_population'] = (cases_df['incident_onsets'] / denom) * 1_000.0
            cases_df['cases_per_1k_population'] = cases_df['cases_per_1k_population'].fillna(0.0)
            cases_df.to_excel(writer, sheet_name="CasesPer1k", index=False)

        incidence_by_year_sex_df = model_results.get('incidence_by_year_sex_df')
        if isinstance(incidence_by_year_sex_df, pd.DataFrame) and not incidence_by_year_sex_df.empty:
            incidence_by_year_sex_df.to_excel(writer, sheet_name="IncidenceByYearSex", index=False)
            prevalence_cols = [
                'time_step',
                'calendar_year',
                'sex',
                'age_band',
                'population_alive_in_band',
                'prevalent_dementia_cases_in_band',
                'dementia_prevalence_in_band',
            ]
            existing_prev_cols = [c for c in prevalence_cols if c in incidence_by_year_sex_df.columns]
            if len(existing_prev_cols) >= 6:
                prevalence_df = incidence_by_year_sex_df[existing_prev_cols].copy()
                prevalence_df.to_excel(writer, sheet_name="PrevalenceByAgeSex", index=False)

        mean_age_cols = ['time_step', 'calendar_year', 'mean_age_alive', 'mean_age_dementia']
        if set(mean_age_cols).issubset(summaries.columns):
            mean_age_df = summaries[mean_age_cols].sort_values('time_step')
            mean_age_df.to_excel(writer, sheet_name="MeanAgeByTimeStep", index=False)

        age_dist = model_results.get('initial_age_distribution', {}) if isinstance(model_results, dict) else {}
        if age_dist:
            age_df = pd.DataFrame(
                sorted(age_dist.items()), columns=["age", "count"]
            )
            age_df.to_excel(writer, sheet_name="BaselineAgeDist", index=False)

        death_dist = model_results.get('age_at_death_distribution', {}) if isinstance(model_results, dict) else {}
        if death_dist:
            death_df = pd.DataFrame(
                sorted(death_dist.items()), columns=["age", "count"]
            )
            death_df.to_excel(writer, sheet_name="AgeAtDeathDist", index=False)

        onset_dist = model_results.get('age_at_onset_distribution', {}) if isinstance(model_results, dict) else {}
        if onset_dist:
            onset_df = pd.DataFrame(
                sorted(onset_dist.items()), columns=["age_at_onset", "count"]
            )
            onset_df.to_excel(writer, sheet_name="AgeAtOnsetDist", index=False)

        risk_onsets = model_results.get('incident_onsets_by_risk_factor', {}) if isinstance(model_results, dict) else {}
        if risk_onsets:
            rows = []
            for risk_name, counts in risk_onsets.items():
                with_risk = int(counts.get('with', 0) or 0)
                without_risk = int(counts.get('without', 0) or 0)
                total = with_risk + without_risk
                rows.append({
                    "risk_factor": risk_name,
                    "onsets_with_risk": with_risk,
                    "onsets_without_risk": without_risk,
                    "total_onsets": total,
                    "with_fraction": (with_risk / total) if total else 0.0,
                })
            risk_df = pd.DataFrame(rows)
            risk_df.to_excel(writer, sheet_name="RiskFactorOnsets", index=False)

        paf_summary = model_results.get('paf_summary') if isinstance(model_results, dict) else None
        if paf_summary:
            paf_df = pd.DataFrame([paf_summary])
            paf_df.to_excel(writer, sheet_name="PAFSummary", index=False)

        lifetime_risk_normal = model_results.get('lifetime_risk_by_entry_age', []) if isinstance(model_results, dict) else []
        if lifetime_risk_normal:
            lifetime_df = pd.DataFrame(lifetime_risk_normal)
            if not lifetime_df.empty:
                lifetime_df = lifetime_df.sort_values('entry_age').reset_index(drop=True)
                lifetime_df['lifetime_risk_pct'] = lifetime_df['lifetime_risk'] * 100.0
                lifetime_df.to_excel(writer, sheet_name="LifetimeRiskNormal", index=False)

        lifetime_risk_all = model_results.get('lifetime_risk_by_entry_age_all', []) if isinstance(model_results, dict) else []
        if lifetime_risk_all:
            lifetime_all_df = pd.DataFrame(lifetime_risk_all)
            if not lifetime_all_df.empty:
                lifetime_all_df = lifetime_all_df.sort_values('entry_age').reset_index(drop=True)
                lifetime_all_df['lifetime_risk_pct'] = lifetime_all_df['lifetime_risk'] * 100.0
                lifetime_all_df.to_excel(writer, sheet_name="LifetimeRiskAll", index=False)

        transition_history = model_results.get('transition_history', {}) if isinstance(model_results, dict) else {}
        if transition_history:
            rows: List[dict] = []
            for time_step in sorted(transition_history.keys()):
                payload = transition_history.get(time_step, {}) or {}
                start_counts = payload.get('stage_start_counts', {}) or {}
                transition_counts = payload.get('transition_counts', {}) or {}
                calendar_year = calendar_lookup.get(time_step)
                for from_stage in DEMENTIA_STAGES:
                    start_total = float(start_counts.get(from_stage, 0))
                    if start_total <= 0:
                        continue
                    for to_stage in DEMENTIA_STAGES:
                        count = float(transition_counts.get((from_stage, to_stage), 0))
                        probability = count / start_total if start_total > 0 else 0.0
                        rows.append({
                            'time_step': time_step,
                            'calendar_year': calendar_year,
                            'from_stage': from_stage,
                            'to_stage': to_stage,
                            'start_count': start_total,
                            'transition_count': count,
                            'transition_probability': probability,
                        })
            transition_df = pd.DataFrame(rows)
            if not transition_df.empty:
                prob_matrix = transition_df.pivot(
                    index=['time_step', 'calendar_year', 'from_stage'],
                    columns='to_stage',
                    values='transition_probability'
                ).reset_index().fillna(0.0)
                prob_matrix.columns.name = None
                prob_matrix.to_excel(writer, sheet_name="TransitionProbabilities", index=False)

                avg_prob_matrix = (
                    transition_df.groupby(['from_stage', 'to_stage'])['transition_probability']
                    .mean()
                    .unstack(fill_value=0.0)
                    .reset_index()
                )
                avg_prob_matrix.columns.name = None
                avg_prob_matrix.to_excel(writer, sheet_name="TransitionProbabilitiesAverage", index=False)

                count_matrix = transition_df.pivot(
                    index=['time_step', 'calendar_year', 'from_stage'],
                    columns='to_stage',
                    values='transition_count'
                ).reset_index().fillna(0.0)
                count_matrix.columns.name = None
                count_matrix.to_excel(writer, sheet_name="TransitionCounts", index=False)

                start_counts_df = transition_df[['time_step', 'calendar_year', 'from_stage', 'start_count']].drop_duplicates()
                start_counts_df.to_excel(writer, sheet_name="TransitionStarts", index=False)

        age_band_summary = model_results.get('age_band_incidence_summary')
        if isinstance(age_band_summary, pd.DataFrame) and not age_band_summary.empty:
            age_band_summary.to_excel(writer, sheet_name="AgeHazardSummary", index=False)

    print(f"Saved aggregated model results to {path}")

# Run & output


# Optional calibration plots (disabled by default; enable via RUN_CALIBRATION_PLOTS=1)
def run_calibration_prevalence_plots():
    data = [
        [35, 49, "F", 0.0001, 0.000776677],
        [50, 64, "F", 0.0012, 0.003178285],
        [65, 79, "F", 0.0178, 0.024883946],
        [80, None, "F", 0.1244, 0.13821326],
        [35, 49, "M", 0.0001, 0.000971348],
        [50, 64, "M", 0.0013, 0.003344429],
        [65, 79, "M", 0.0168, 0.023354982],
        [80, None, "M", 0.0910, 0.10122974],
    ]
    df = pd.DataFrame(data, columns=["age_lower", "age_upper", "sex", "obs", "pred"])
    df["age_band"] = df.apply(
        lambda r: f"{int(r.age_lower)}+" if pd.isna(r.age_upper)
        else f"{int(r.age_lower)}-{int(r.age_upper)}", axis=1
    )

    save_dir = Path(r"C:\Users\EdwardCoote\OneDrive\DementiaModel\plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    def ols_fit(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.column_stack((np.ones_like(x), x))
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        y_hat = X @ beta
        resid = y - y_hat
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum(resid ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"alpha": beta[0], "beta": beta[1], "r2": r2}

    def run_and_save(sub, sex_label):
        res = ols_fit(sub["pred"], sub["obs"])
        alpha, beta, r2 = res["alpha"], res["beta"], res["r2"]
        print(f"\nSex = {sex_label}")
        print(f"  alpha (intercept): {alpha:.6f}")
        print(f"  beta (slope):     {beta:.6f}")
        print(f"  R^2:               {r2:.3f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(sub["pred"], sub["obs"], s=90)
        xmax = max(sub["pred"].max(), sub["obs"].max()) * 1.05
        x = np.linspace(0, xmax, 100)
        plt.plot(x, x, "k--", label="1:1 line")
        plt.plot(x, alpha + beta * x, "r-", label=f"Fitted (beta={beta:.2f})")
        for _, row in sub.iterrows():
            plt.annotate(row["age_band"], (row["pred"], row["obs"]), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
        plt.xlabel("Predicted prevalence (2024)")
        plt.ylabel("Observed prevalence (2024)")
        plt.title(f"Calibration - Prevalence 2024 ({sex_label})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        file_path = save_dir / f"calibration_prevalence_{sex_label}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"  Plot saved to: {file_path}")
        return res

    res_f = run_and_save(df[df["sex"] == "F"], "Female")
    res_m = run_and_save(df[df["sex"] == "M"], "Male")
    return {"female": res_f, "male": res_m}


if __name__ == "__main__":
    run_seed = 42
    model_results = run_model(general_config, seed=run_seed)

    if general_config.get('enable_constant_hazard_checks', False):
        try:
            baseline_check = check_constant_hazard_from_model(model_results, tolerance=0.05, cohort='baseline')
            mean_hazard = baseline_check['mean_hazard']
            max_dev = baseline_check['max_relative_deviation']
            within = baseline_check['within_tolerance']
            print("\nBaseline cohort constant-hazard check:")
            print(f"  Mean hazard: {mean_hazard:.6f} per year")
            print(f"  Max relative deviation: {max_dev:.2%}")
            print(f"  Within tolerance (+/- 5%): {within}")
        except ValueError as exc:
            print(f"\nBaseline cohort constant-hazard check unavailable: {exc}")

    if general_config.get('compute_paf_in_main', False):
        paf_summary = compute_population_attributable_fraction(
            general_config,
            risk_factor='periodontal_disease',
            baseline_results=model_results,
            seed=run_seed,
        )
        if paf_summary:
            model_results['paf_summary'] = paf_summary
            if general_config.get('report_paf_to_terminal', False):
                total_onsets = paf_summary['baseline_onsets']
                with_periodontal = paf_summary['baseline_with_risk_onsets']
                without_periodontal = paf_summary['baseline_without_risk_onsets']
                paf_value = paf_summary['paf']
                print("\nPeriodontal disease population attributable fraction (PAF) summary:")
                print(f"  Total dementia onsets (baseline): {total_onsets}")
                print(f"  Dementia onsets with periodontal disease: {with_periodontal}")
                print(f"  Dementia onsets without periodontal disease: {without_periodontal}")
                print(f"  PAF attributable to periodontal disease: {paf_value:.2%}")
                print(f"  PAF for those without periodontal disease: {(1.0 - paf_value):.2%}")
        elif general_config.get('report_paf_to_terminal', False):
            print("Unable to compute periodontal disease PAF (no dementia onsets in baseline scenario).")

    psa_cfg = general_config.get('psa', {})
    if psa_cfg.get('use', False):
        run_standard_psa = not psa_cfg.get('two_level_only', False)

        if run_standard_psa:
            psa_results = run_probabilistic_sensitivity_analysis(
                general_config,
                psa_cfg,
                collect_draw_level=False,
                seed=run_seed,
            )
            summary = psa_results.get('summary', {})
            if summary:
                print("\nProbabilistic sensitivity analysis (95% CI):")
                focus_metrics = [
                    'total_costs_all',
                    'total_qalys_combined',
                    'incident_onsets_total',
                    'stage_mild',
                    'stage_moderate',
                    'stage_severe',
                ]
                for metric in focus_metrics:
                    stats = summary.get(metric)
                    if not stats:
                        continue
                    mean_val = stats.get('mean')
                    lo = stats.get('lower_95')
                    hi = stats.get('upper_95')
                    print(f"  {metric}: mean={mean_val:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")

        two_level_results = run_two_level_psa(
            general_config,
            psa_cfg,
            n_outer=psa_cfg.get('iterations', 1000),
            variance_pilot_results=None,
            collect_draw_level=False,
            seed=run_seed,
            n_jobs=psa_cfg.get('n_jobs')
        )
        two_level_summary = two_level_results.get('summary', {})
        if two_level_summary:
            print("\nTwo-level PSA (95% CI using O'Hagan method):")
            for metric in [
                'total_costs_all',
                'total_qalys_combined',
                'incident_onsets_total',
            ]:
                stats = two_level_summary.get(metric)
                if not stats:
                    continue
                mean_val = stats.get('mean')
                lo = stats.get('lower_95')
                hi = stats.get('upper_95')
                print(f"  {metric}: mean={mean_val:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")

    if general_config.get('enable_constant_hazard_checks', False):
        run_constant_hazard_diagnostics(
            model_results,
            general_config,
            tolerance=0.05,
            show_plots=False
        )

    # PLOTS DISABLED: Use generate_figures_2_3_4_from_model.py to create publication figures
    # The following plots are commented out to avoid generating unnecessary charts during batch processing.
    # All data needed for figures is exported to Excel (see export_results_to_excel below).
    # To re-enable plots for manual runs, uncomment the lines below:

    # plot_ad_prevalence(model_results, show=True)
    # plot_ad_incidence(model_results, show=True)
    # plot_age_specific_ad_cases(model_results, show=True)
    # plot_dementia_prevalence_by_stage(model_results, show=True)
    # plot_survival_curve(model_results, show=True)
    # plot_survival_by_baseline_stage(model_results, show=True)
    # plot_baseline_age_hist(model_results, show=True)
    # plot_age_at_death_hist(model_results, show=True)
    # plot_costs_per_person_over_time(model_results, show=True)
    # plot_qalys_per_person_over_time(model_results, show=True)
    # plot_lifetime_risk_by_entry_age(model_results, show=True)
    # plot_onset_hazard_vs_age_from_base_prob(general_config, show=True)
    # plot_onset_probability_vs_age_from_base_prob(general_config, show=True)

    # Export all model results to Excel for journal reproducibility
    # This includes: summary data, yearly flows, risk factor prevalence, and severity distribution
    excel_output_path = Path("results") / "Baseline_Model.xlsx"
    excel_output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting results to Excel: {excel_output_path}")

    def _save_fallback(exc: Exception):
        fallback_path = excel_output_path.with_name(f"{excel_output_path.stem}_export_failed.pkl.gz")
        try:
            save_results_compressed(model_results, fallback_path)
            print(f"Excel export failed ({exc}); raw results saved to {fallback_path} for recovery.")
        except Exception as save_exc:
            print(f"Excel export failed ({exc}) and fallback save also failed ({save_exc}).")

    try:
        export_results_to_excel(model_results, path=str(excel_output_path))
    except PermissionError as exc:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = excel_output_path.with_name(f"{excel_output_path.stem}_{timestamp}.xlsx")
        try:
            export_results_to_excel(model_results, path=str(alt_path))
            print(f"Primary Excel path locked ({exc}); wrote to {alt_path} instead.")
        except Exception as exc2:
            _save_fallback(exc2)
    except Exception as exc:
        _save_fallback(exc)

    if os.environ.get("RUN_CALIBRATION_PLOTS", "0") == "1":
        run_calibration_prevalence_plots()

