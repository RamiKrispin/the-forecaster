"""
Utility Functions

Helper functions for data processing and common operations.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


def prepare_time_series(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    sort: bool = True
) -> pd.DataFrame:
    """
    Prepare time series data by sorting and creating time index.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    sort : bool
        Whether to sort by time column

    Returns
    -------
    pd.DataFrame
        Prepared data with time_index column
    """
    df = data.copy()

    if sort:
        df = df.sort_values(by=time_col).reset_index(drop=True)

    df['time_index'] = range(len(df))

    return df


def calculate_residuals(
    actual: np.ndarray,
    fitted: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Calculate residuals and basic statistics.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    fitted : np.ndarray
        Fitted values

    Returns
    -------
    tuple
        (residuals, stats_dict) where stats_dict contains mean, std, etc.
    """
    residuals = actual - fitted

    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals))
    }

    return residuals, stats


def train_test_split_ts(
    data: pd.DataFrame,
    test_size: Union[int, float] = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data (should be sorted by time)
    test_size : int or float
        If float, proportion of data for test set
        If int, number of observations for test set

    Returns
    -------
    tuple
        (train_data, test_data)
    """
    n = len(data)

    if isinstance(test_size, float):
        split_idx = int(n * (1 - test_size))
    else:
        split_idx = n - test_size

    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    return train, test


def generate_candidate_positions(
    n_knots: int,
    min_idx: int,
    max_idx: int,
    min_segment: int,
    grid_resolution: int = 20
) -> List[List[int]]:
    """
    Generate candidate knot positions for grid search.

    Parameters
    ----------
    n_knots : int
        Number of knots
    min_idx : int
        Minimum valid index
    max_idx : int
        Maximum valid index
    min_segment : int
        Minimum segment length
    grid_resolution : int
        Number of candidate positions per knot

    Returns
    -------
    list
        List of candidate knot position combinations
    """
    # TODO: Implement candidate generation logic
    # This will mirror the R function's logic
    candidates = []

    return candidates
