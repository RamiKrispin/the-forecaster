"""
Visualization Module

Plotting and visualization utilities for time series data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def plot_timeseries(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot time series data.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    title : str, optional
        Plot title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data[time_col], data[value_col], linewidth=2)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_piecewise_fit(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    fitted_col: str = "fitted",
    knot_positions: Optional[List] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot time series with piecewise regression fit and knot positions.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with fitted values
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    fitted_col : str
        Name of the fitted values column
    knot_positions : list, optional
        List of knot positions to mark on the plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual values
    ax.plot(data[time_col], data[value_col], 'o', alpha=0.5, label='Actual')

    # Plot fitted line
    ax.plot(data[time_col], data[fitted_col], 'r-', linewidth=2, label='Fitted')

    # Mark knot positions
    if knot_positions:
        for knot in knot_positions:
            ax.axvline(x=knot, color='green', linestyle='--', alpha=0.7)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bic_scores(
    bic_scores: pd.DataFrame,
    optimal_knots: int,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot BIC scores for different numbers of knots.

    Parameters
    ----------
    bic_scores : pd.DataFrame
        DataFrame with columns 'n_knots' and 'bic'
    optimal_knots : int
        The optimal number of knots
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bic_scores['n_knots'], bic_scores['bic'],
            'o-', linewidth=2, markersize=8)

    # Highlight optimal point
    optimal_row = bic_scores[bic_scores['n_knots'] == optimal_knots]
    ax.plot(optimal_row['n_knots'], optimal_row['bic'],
            'ro', markersize=12, label='Optimal')

    ax.set_title('BIC Scores by Number of Knots', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Knots')
    ax.set_ylabel('BIC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
