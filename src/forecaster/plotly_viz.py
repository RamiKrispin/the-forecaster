"""
Plotly Visualization Module

Interactive plotting functions for piecewise regression results using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional


def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> str:
    """
    Convert hex color to rgba format.

    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., "#F39EB6" or "F39EB6")
    opacity : float
        Opacity level between 0 and 1 (default=1)

    Returns
    -------
    str
        RGBA color string

    Examples
    --------
    >>> hex_to_rgba("#F39EB6", 0.5)
    'rgba(243, 158, 182, 0.5)'
    """
    # Remove the # if present
    hex_color = hex_color.lstrip('#')

    # Extract RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Validate opacity is between 0 and 1
    if opacity < 0 or opacity > 1:
        raise ValueError("Opacity must be between 0 and 1")

    # Return formatted rgba string
    return f"rgba({r}, {g}, {b}, {opacity})"


def create_arc(x0: float, y0: float, x1: float, y1: float,
               curvature: float = 0.2, length_out: int = 50) -> pd.DataFrame:
    """
    Create arc curve data for plotly annotations.

    Generates bezier curve points for creating curved annotation lines in plotly.

    Parameters
    ----------
    x0, y0 : float
        Starting point coordinates
    x1, y1 : float
        Ending point coordinates
    curvature : float
        Height of the arc as a proportion of distance (default=0.2)
    length_out : int
        Number of points to generate (default=50)

    Returns
    -------
    pd.DataFrame
        DataFrame with 'x' and 'y' columns containing curve points

    Examples
    --------
    >>> arc = create_arc(0, 0, 10, 10, curvature=0.3)
    >>> len(arc)
    50
    """
    # Calculate midpoint
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    # Calculate ranges for scaling
    x_range = abs(x1 - x0)
    y_range = abs(y1 - y0)

    # Handle vertical or horizontal lines
    if x_range == 0:
        x_range = 0.1
    if y_range == 0:
        y_range = abs(my) * 0.1

    # Direction vector
    dx = x1 - x0
    dy = y1 - y0

    # Perpendicular offset: rotate 90 degrees and scale by curvature
    # Apply curvature as a percentage of the respective range
    cx = mx - np.sign(dy) * curvature * x_range
    cy = my + np.sign(dx) * curvature * y_range

    # Generate bezier curve points using quadratic bezier formula
    # B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
    t = np.linspace(0, 1, length_out)
    x = (1 - t)**2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t)**2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1

    # Return data frame ready for plotly
    return pd.DataFrame({'x': x, 'y': y})


def plot_bic_scores(result: Dict, **kwargs) -> go.Figure:
    """
    Plot BIC scores by number of knots using Plotly.

    Creates an interactive plotly visualization of BIC scores across different
    numbers of knots, highlighting the optimal choice.

    Parameters
    ----------
    result : dict
        Output from the piecewise_regression function
    **kwargs : dict
        Additional layout options passed to plotly layout

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure showing BIC scores by number of knots

    Examples
    --------
    >>> from forecaster import piecewise_regression
    >>> result = piecewise_regression(data, time_col='date', value_col='value')
    >>> fig = plot_bic_scores(result)
    >>> fig.show()
    """
    # Extract BIC scores data frame
    bic_data = result['bic_scores']
    optimal_knots = result['optimal_knots']

    # Get the optimal point data
    optimal_point = bic_data[bic_data['n_knots'] == optimal_knots]

    # Create the plot
    fig = go.Figure()

    # Add line with markers for all BIC scores
    fig.add_trace(go.Scatter(
        x=bic_data['n_knots'],
        y=bic_data['bic'],
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        marker=dict(size=8, color='steelblue'),
        name='BIC Score',
        showlegend=False
    ))

    # Add optimal point highlight
    fig.add_trace(go.Scatter(
        x=optimal_point['n_knots'],
        y=optimal_point['bic'],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Optimal',
        showlegend=True
    ))

    # Default layout
    layout_options = dict(
        title=dict(
            text='BIC Scores by Number of Knots<br><sub>Lower BIC = Better Model</sub>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Number of Knots',
            dtick=1
        ),
        yaxis=dict(
            title='BIC'
        ),
        hovermode='closest'
    )

    # Merge with user-provided kwargs
    layout_options.update(kwargs)

    # Add annotation for optimal point
    layout_options['annotations'] = [
        dict(
            x=optimal_point['n_knots'].values[0],
            y=optimal_point['bic'].values[0],
            text='Optimal',
            xanchor='left',
            xshift=10,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            ax=40,
            ay=0,
            font=dict(
                size=12,
                color='red',
                family='Arial Black'
            )
        )
    ]

    fig.update_layout(**layout_options)

    return fig


def plot_knots(result: Dict, time_col: str, value_col: str, **kwargs) -> go.Figure:
    """
    Plot time series with optimal knot positions using Plotly.

    Creates a plotly visualization of the time series data with vertical dashed
    lines indicating the optimal knot positions found by piecewise regression.

    Parameters
    ----------
    result : dict
        Output from the piecewise_regression function
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    **kwargs : dict
        Additional layout options passed to plotly layout

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure showing the time series with knot positions

    Examples
    --------
    >>> from forecaster import piecewise_regression
    >>> result = piecewise_regression(data, time_col='date', value_col='value')
    >>> fig = plot_knots(result, time_col='date', value_col='value')
    >>> fig.show()
    """
    # Extract data
    data = result['data']
    knot_dates = result['knot_dates']
    optimal_knots = result['optimal_knots']

    # Create the plot
    fig = go.Figure()

    # Add actual values as scatter points
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data[value_col],
        mode='markers',
        marker=dict(color='#0072B5', size=6, opacity=0.5),
        name='Actual',
        showlegend=True
    ))

    # Add fitted line
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data['fitted'],
        mode='lines',
        line=dict(color='red', width=2),
        name='Fitted (Piecewise)',
        showlegend=True
    ))

    # Add vertical lines for each knot position
    if knot_dates is not None and len(knot_dates) > 0:
        y_min = data[value_col].min()
        y_max = data[value_col].max()

        for i, knot_date in enumerate(knot_dates):
            # Add vertical line segment
            fig.add_trace(go.Scatter(
                x=[knot_date, knot_date],
                y=[y_min, y_max],
                mode='lines',
                line=dict(color='darkgreen', dash='dash', width=2),
                name='Knots' if i == 0 else None,
                showlegend=True if i == 0 else False,
                legendgroup='knots'
            ))

    # Default layout
    knot_text = 's' if optimal_knots != 1 else ''
    layout_options = dict(
        title=dict(
            text=f'Piecewise Linear Regression<br><sub>Optimal: {optimal_knots} knot{knot_text}</sub>',
            font=dict(size=16)
        ),
        xaxis=dict(title=time_col),
        yaxis=dict(title=value_col),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.2)
    )

    # Merge with user-provided kwargs
    layout_options.update(kwargs)

    fig.update_layout(**layout_options)

    return fig


def plot_piecewise_comparison(
    result: Dict,
    time_col: str,
    value_col: str,
    show_bic: bool = True,
    **kwargs
) -> go.Figure:
    """
    Create side-by-side comparison of BIC scores and fitted model.

    Parameters
    ----------
    result : dict
        Output from the piecewise_regression function
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    show_bic : bool
        Whether to show BIC plot in subplot (default: True)
    **kwargs : dict
        Additional layout options

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with subplots

    Examples
    --------
    >>> from forecaster import piecewise_regression
    >>> result = piecewise_regression(data, time_col='date', value_col='value')
    >>> fig = plot_piecewise_comparison(result, time_col='date', value_col='value')
    >>> fig.show()
    """
    from plotly.subplots import make_subplots

    if show_bic:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('BIC Scores', 'Fitted Model with Knots'),
            horizontal_spacing=0.12
        )

        # Get BIC plot
        bic_fig = plot_bic_scores(result)

        # Add BIC traces to subplot
        for trace in bic_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Update BIC subplot axes
        fig.update_xaxes(title_text='Number of Knots', dtick=1, row=1, col=1)
        fig.update_yaxes(title_text='BIC', row=1, col=1)

        # Add BIC annotations
        for annotation in bic_fig.layout.annotations:
            annotation['xref'] = 'x1'
            annotation['yref'] = 'y1'
            fig.add_annotation(annotation)

    else:
        fig = go.Figure()

    # Get knots plot
    knots_fig = plot_knots(result, time_col=time_col, value_col=value_col)

    # Add knots traces to subplot or main figure
    if show_bic:
        for trace in knots_fig.data:
            fig.add_trace(trace, row=1, col=2)
        fig.update_xaxes(title_text=time_col, row=1, col=2)
        fig.update_yaxes(title_text=value_col, row=1, col=2)
    else:
        for trace in knots_fig.data:
            fig.add_trace(trace)
        fig.update_xaxes(title_text=time_col)
        fig.update_yaxes(title_text=value_col)

    # Update overall layout
    layout_options = dict(
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    layout_options.update(kwargs)

    fig.update_layout(**layout_options)

    return fig
