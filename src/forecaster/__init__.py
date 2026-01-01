"""
Forecaster Module

A collection of supporting functions for time series forecasting articles and notebooks.
This module provides utilities for:
- Piecewise regression and trend detection
- Time series visualization
- Forecasting utilities
- Model evaluation tools
"""

__version__ = "0.1.0"
__author__ = "Rami Krispin"

from .piecewise import piecewise_regression, calculate_bic
from .plotly_viz import plot_bic_scores, plot_knots, plot_piecewise_comparison, create_arc, hex_to_rgba
from .animation import record_piecewise_search, record_piecewise_search_plotly

__all__ = [
    'piecewise_regression',
    'calculate_bic',
    'plot_bic_scores',
    'plot_knots',
    'plot_piecewise_comparison',
    'create_arc',
    'hex_to_rgba',
    'record_piecewise_search',
    'record_piecewise_search_plotly'
]
