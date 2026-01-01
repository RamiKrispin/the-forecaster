"""
Example usage of the forecaster module.

This script demonstrates how to use the piecewise regression function
with both matplotlib and plotly visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecaster import piecewise_regression, plot_bic_scores, plot_knots, plot_piecewise_comparison


def example_simple_piecewise():
    """Example 1: Simple piecewise regression with one breakpoint."""
    print("=" * 60)
    print("Example 1: Simple Piecewise Regression")
    print("=" * 60)

    # Create synthetic data with one clear breakpoint
    np.random.seed(42)
    n = 100

    # First segment: increasing trend
    x1 = np.arange(0, 50)
    y1 = 2 * x1 + 10 + np.random.normal(0, 3, 50)

    # Second segment: decreasing trend
    x2 = np.arange(50, 100)
    y2 = -1 * (x2 - 50) + 110 + np.random.normal(0, 3, 50)

    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.concatenate([y1, y2])
    })

    # Fit piecewise regression
    result = piecewise_regression(
        data=data,
        time_col='date',
        value_col='value',
        max_knots=3,
        min_segment_length=20,
        edge_buffer=0.05,
        grid_resolution=20
    )

    print(f"\nOptimal number of knots: {result['optimal_knots']}")
    print(f"Knot positions (indices): {result['knot_positions']}")
    print(f"Knot dates: {result['knot_dates']}")
    print("\nBIC Scores:")
    print(result['bic_scores'])

    # Create matplotlib plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Time series with fitted values
    ax1.plot(result['data']['date'], result['data']['value'],
             'o', alpha=0.5, label='Actual', markersize=4)
    ax1.plot(result['data']['date'], result['data']['fitted'],
             'r-', linewidth=2, label='Fitted')

    # Mark knot positions
    if result['knot_dates']:
        for knot_date in result['knot_dates']:
            ax1.axvline(x=knot_date, color='green', linestyle='--',
                       alpha=0.7, label='Knot' if knot_date == result['knot_dates'][0] else '')

    ax1.set_title('Piecewise Linear Regression')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: BIC scores
    ax2.plot(result['bic_scores']['n_knots'], result['bic_scores']['bic'],
             'o-', linewidth=2, markersize=8)
    optimal_row = result['bic_scores'][result['bic_scores']['n_knots'] == result['optimal_knots']]
    ax2.plot(optimal_row['n_knots'], optimal_row['bic'],
             'ro', markersize=12, label='Optimal')
    ax2.set_title('BIC Scores by Number of Knots')
    ax2.set_xlabel('Number of Knots')
    ax2.set_ylabel('BIC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create interactive plotly version
    print("\nCreating interactive plotly visualizations...")
    plotly_fig = plot_piecewise_comparison(
        result,
        time_col='date',
        value_col='value',
        title='Piecewise Regression - Interactive'
    )

    return fig, result, plotly_fig


def example_plotly_only():
    """Example 2: Using plotly visualization functions."""
    print("\n" + "=" * 60)
    print("Example 2: Plotly Interactive Visualizations")
    print("=" * 60)

    # Create synthetic data with multiple breakpoints
    np.random.seed(123)
    n = 150

    # Three segments with different slopes
    x1 = np.arange(0, 50)
    y1 = 1.5 * x1 + 20 + np.random.normal(0, 2, 50)

    x2 = np.arange(50, 100)
    y2 = 0.2 * (x2 - 50) + 95 + np.random.normal(0, 2, 50)

    x3 = np.arange(100, 150)
    y3 = -0.8 * (x3 - 100) + 105 + np.random.normal(0, 2, 50)

    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.concatenate([y1, y2, y3])
    })

    # Fit piecewise regression
    result = piecewise_regression(
        data=data,
        time_col='date',
        value_col='value',
        max_knots=4,
        min_segment_length=25,
        edge_buffer=0.05,
        grid_resolution=15
    )

    print(f"\nOptimal number of knots: {result['optimal_knots']}")
    print(f"Knot positions: {result['knot_positions']}")

    # Create individual plotly figures
    print("\nCreating plotly figures...")

    # BIC scores plot
    bic_fig = plot_bic_scores(result)
    print("✓ BIC scores plot created")

    # Knots plot
    knots_fig = plot_knots(
        result,
        time_col='date',
        value_col='value',
        title='Time Series with Detected Breakpoints'
    )
    print("✓ Knots plot created")

    # Combined plot
    combined_fig = plot_piecewise_comparison(
        result,
        time_col='date',
        value_col='value'
    )
    print("✓ Combined comparison plot created")

    print("\nTo view the interactive plots, use:")
    print("  bic_fig.show()")
    print("  knots_fig.show()")
    print("  combined_fig.show()")

    return result, bic_fig, knots_fig, combined_fig


if __name__ == "__main__":
    print("\nForecaster Module - Piecewise Regression Examples\n")

    # Run examples
    print("Running Example 1: Simple piecewise with matplotlib + plotly...")
    fig1, result1, plotly_fig1 = example_simple_piecewise()

    print("\nRunning Example 2: Plotly-only visualizations...")
    result2, bic_fig, knots_fig, combined_fig = example_plotly_only()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    print("\nMatplotlib figures created. Use plt.show() to display them.")
    print("Plotly figures created. Use fig.show() to display them interactively.")

    # Uncomment to display matplotlib plots
    # plt.show()

    # Uncomment to display plotly plots
    # plotly_fig1.show()
    # bic_fig.show()
    # knots_fig.show()
    # combined_fig.show()
