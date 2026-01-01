"""
Piecewise Regression Module

Functions for piecewise linear regression and trend detection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import warnings


def piecewise_regression(
    data: pd.DataFrame,
    time_col: str = "date",
    value_col: str = "value",
    max_knots: int = 3,
    min_segment_length: int = 30,
    edge_buffer: float = 0.05,
    grid_resolution: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Perform piecewise linear regression with automatic knot detection using grid search.

    This function fits continuous piecewise linear regression models with different numbers
    of knots (breakpoints) and selects the optimal model based on BIC (Bayesian Information
    Criterion). It uses a "broken stick" approach to ensure continuity at knot points.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    max_knots : int
        Maximum number of knots to test (default: 3)
    min_segment_length : int
        Minimum number of observations per segment (default: 30)
    edge_buffer : float
        Percentage of data to exclude from edges (default: 0.05)
    grid_resolution : int
        Number of candidate positions per knot (default: 20)
    verbose : bool
        Whether to print progress messages (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - optimal_knots: Optimal number of knots
        - knot_positions: List of knot indices
        - knot_dates: List of knot time values
        - bic_scores: DataFrame with BIC scores for each knot count
        - model: Dict with fitted model parameters
        - data: DataFrame with original data plus fitted values
        - all_results: List of all tested configurations
    """
    # Prepare data
    df = data.copy()
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df['time_index'] = np.arange(len(df))
    df['y'] = df[value_col].values

    n = len(df)

    # Define valid range for knots (exclude edges)
    min_idx = int(np.ceil(n * edge_buffer))
    max_idx = int(np.floor(n * (1 - edge_buffer)))

    def fit_piecewise(knots: List[int], data_df: pd.DataFrame) -> Dict:
        """
        Fit piecewise linear model given knot positions.
        Uses broken stick approach for continuous piecewise linear regression.
        """
        if len(knots) == 0:
            # No knots - simple linear regression
            X = data_df[['time_index']].values
            y = data_df['y'].values

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)
            residuals = y - y_pred
            rss = np.sum(residuals**2)

            return {
                'model': model,
                'rss': rss,
                'n_params': 2,  # intercept + slope
                'knots': [],
                'X_design': X,
                'fitted': y_pred
            }

        # Sort knots
        knots = sorted(knots)

        # Build design matrix for continuous piecewise linear (broken stick)
        X = np.ones((n, 1))  # Intercept
        X = np.column_stack([X, data_df['time_index'].values])  # First slope

        # Add broken stick terms (continuous piecewise linear)
        for k in knots:
            X = np.column_stack([X, np.maximum(data_df['time_index'].values - k, 0)])

        y = data_df['y'].values

        # Fit model using LinearRegression (without separate intercept)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred
        rss = np.sum(residuals**2)

        n_params = 2 + len(knots)  # intercept + initial slope + slope changes

        return {
            'model': model,
            'rss': rss,
            'n_params': n_params,
            'knots': knots,
            'X_design': X,
            'fitted': y_pred
        }

    def calc_bic(rss: float, n: int, k: int) -> float:
        """Calculate Bayesian Information Criterion."""
        return n * np.log(rss / n) + k * np.log(n)

    def generate_candidates(n_knots: int, min_idx: int, max_idx: int,
                          min_segment: int) -> List[List[int]]:
        """
        Generate candidate knot positions for grid search.
        Different strategies for 1, 2, and 3+ knots.
        """
        if n_knots == 0:
            return [[]]

        # Calculate minimum spacing required
        total_min_length = (n_knots + 1) * min_segment
        available_length = max_idx - min_idx + 1

        if total_min_length > available_length:
            warnings.warn(f"Cannot fit {n_knots} knots with min segment length {min_segment}")
            return []

        candidates = []

        if n_knots == 1:
            # For 1 knot, test positions with proper spacing
            n_positions = min(grid_resolution, max_idx - min_idx - 2 * min_segment)
            if n_positions > 0:
                positions = np.linspace(
                    min_idx + min_segment,
                    max_idx - min_segment,
                    num=int(n_positions)
                )
                positions = np.round(positions).astype(int)
                candidates = [[int(p)] for p in positions]

        elif n_knots == 2:
            # For 2 knots, create a proper grid search
            # Generate all candidate positions first
            available_range = max_idx - min_idx - 2 * min_segment
            n_positions = min(grid_resolution, available_range)

            if n_positions > 0:
                # Create uniform grid of candidate positions
                all_positions = np.linspace(
                    min_idx + min_segment,
                    max_idx - min_segment,
                    num=int(n_positions)
                )
                all_positions = np.round(all_positions).astype(int)
                all_positions = np.unique(all_positions)  # Remove any duplicates from rounding

                # Test all valid combinations where pos1 < pos2 and they're separated by min_segment
                for i, pos1 in enumerate(all_positions):
                    for pos2 in all_positions[i+1:]:
                        if pos2 - pos1 >= min_segment:
                            candidates.append([int(pos1), int(pos2)])

        else:
            # For 3+ knots, use a coarser grid
            segment_length = (max_idx - min_idx) / (n_knots + 1)

            # Create base positions evenly spaced
            base_positions = np.linspace(
                min_idx + segment_length,
                max_idx - segment_length,
                num=n_knots
            )
            base_positions = np.round(base_positions).astype(int)

            # Test variations around base positions
            search_window = int(np.round(segment_length * 0.3))

            if n_knots == 3:
                # For 3 knots, test variations
                offsets = np.linspace(-search_window, search_window, num=5)

                for offset1 in offsets:
                    for offset2 in offsets:
                        for offset3 in offsets:
                            pos = base_positions + np.array([offset1, offset2, offset3])
                            pos = np.round(pos).astype(int)

                            # Check minimum segment constraint
                            all_points = np.concatenate([[min_idx], pos, [max_idx]])
                            diffs = np.diff(all_points)

                            if np.all(diffs >= min_segment):
                                candidates.append(pos.tolist())
            else:
                # For 4+ knots, just use base positions
                candidates = [base_positions.tolist()]

        return candidates

    # Test different numbers of knots
    results = []
    overall_best_bic = np.inf

    for k in range(max_knots + 1):
        if verbose:
            print(f"Testing {k} knot(s)...")

        # Generate candidate knot positions
        candidates = generate_candidates(k, min_idx, max_idx, min_segment_length)

        if len(candidates) == 0:
            continue

        best_bic = np.inf
        best_knots = None
        best_model = None
        best_rss = None
        best_fitted = None

        # Test each candidate
        for knots in candidates:
            fit = fit_piecewise(knots, df)
            bic = calc_bic(fit['rss'], n, fit['n_params'])

            if bic < best_bic:
                best_bic = bic
                best_knots = knots
                best_model = fit
                best_rss = fit['rss']
                best_fitted = fit['fitted']

            # Update overall best
            if bic < overall_best_bic:
                overall_best_bic = bic

        results.append({
            'n_knots': k,
            'knots': best_knots,
            'bic': best_bic,
            'rss': best_rss,
            'model': best_model,
            'n_candidates': len(candidates)
        })

        if verbose:
            print(f"  Best BIC: {best_bic:.2f} | RSS: {best_rss:.2f} | "
                  f"Tested {len(candidates)} configurations")

    # Find optimal number of knots
    bic_values = [r['bic'] for r in results]
    optimal_idx = np.argmin(bic_values)
    optimal = results[optimal_idx]

    if verbose:
        print(f"\nOptimal model: {optimal['n_knots']} knot(s) with BIC = {optimal['bic']:.2f}")

    # Get fitted values for optimal model
    df['fitted'] = optimal['model']['fitted']

    # Convert knot indices back to original time values
    if optimal['knots'] and len(optimal['knots']) > 0:
        knot_dates = df[time_col].iloc[optimal['knots']].tolist()
    else:
        knot_dates = None

    # Create BIC scores dataframe
    bic_scores = pd.DataFrame({
        'n_knots': [r['n_knots'] for r in results],
        'bic': bic_values,
        'rss': [r['rss'] for r in results]
    })

    return {
        'optimal_knots': optimal['n_knots'],
        'knot_positions': optimal['knots'] if optimal['knots'] else [],
        'knot_dates': knot_dates,
        'bic_scores': bic_scores,
        'model': optimal['model'],
        'data': df,
        'all_results': results
    }


def calculate_bic(rss: float, n: int, k: int) -> float:
    """
    Calculate Bayesian Information Criterion.

    Parameters
    ----------
    rss : float
        Residual sum of squares
    n : int
        Number of observations
    k : int
        Number of parameters

    Returns
    -------
    float
        BIC value
    """
    return n * np.log(rss / n) + k * np.log(n)
