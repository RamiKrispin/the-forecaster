"""
Animation Module

Functions for creating animated visualizations of the piecewise regression grid search process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, Optional, List
import os
from pathlib import Path
import warnings


def record_piecewise_search(
    data: pd.DataFrame,
    time_col: str = "date",
    value_col: str = "value",
    max_knots: int = 3,
    min_segment_length: int = 30,
    edge_buffer: float = 0.05,
    grid_resolution: int = 20,
    output_dir: str = "grid_search_animation",
    gif_name: str = "grid_search.gif",
    width: int = 10,
    height: int = 6,
    fps: int = 2,
    dpi: int = 100,
    backend: str = "pillow",
    verbose: bool = True
) -> Dict:
    """
    Record piecewise regression grid search process and create animated GIF.

    This function runs the grid search while capturing EVERY tested configuration
    as a frame, then combines all frames into an animated GIF showing the complete
    search process. Configurations are color-coded to show best models as they
    are discovered.

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
    output_dir : str
        Directory to save animation frames and GIF (default: "grid_search_animation")
    gif_name : str
        Name of output GIF file (default: "grid_search.gif")
    width : int
        Figure width in inches (default: 10)
    height : int
        Figure height in inches (default: 6)
    fps : int
        Frames per second for GIF (default: 2)
    dpi : int
        DPI for frames (default: 100)
    backend : str
        Animation backend: 'pillow' (default) or 'imagemagick'
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - regression_result: Full piecewise regression results
        - gif_path: Path to created GIF file
        - frames_dir: Directory containing frame images
        - frames_captured: Number of frames created

    Examples
    --------
    >>> result = record_piecewise_search(
    ...     data=ts,
    ...     time_col='date',
    ...     value_col='value',
    ...     max_knots=3,
    ...     output_dir='animation'
    ... )
    >>> print(f"GIF saved to: {result['gif_path']}")

    Notes
    -----
    - Requires matplotlib and Pillow (or imagemagick)
    - Creates PNG frames in output_dir during processing
    - Frames are combined into GIF at the end
    - Captures ALL grid configurations, not just best results
    - Color coding: dark green (overall best), blue (best for k), light gray (other)
    - Can be memory intensive for large datasets or many frames
    - Reduce grid_resolution to decrease frame count and file size
    """
    # Import here to avoid requiring it if not using animation
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for creating animations. "
            "Install with: pip install Pillow"
        )

    # Import piecewise regression from same package
    from .piecewise import piecewise_regression

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Recording Piecewise Regression Grid Search")
        print("=" * 60)

    # Clean old frames
    old_frames = list(output_path.glob("frame_*.png"))
    if old_frames and verbose:
        print(f"Removing {len(old_frames)} old frames from previous run...")
        for f in old_frames:
            f.unlink()

    # Data analysis
    n = len(data)
    min_idx = int(np.ceil(n * edge_buffer))
    max_idx = int(np.floor(n * (1 - edge_buffer)))
    available_length = max_idx - min_idx + 1

    if verbose:
        print(f"\nData Analysis:")
        print(f"  Total observations: {n}")
        print(f"  Available range for knots: {min_idx} to {max_idx} ({available_length} obs)")
        print(f"  Min segment length: {min_segment_length}")
        print(f"  Max knots: {max_knots}")
        print(f"  Min observations required: {(max_knots + 1) * min_segment_length}")

    if (max_knots + 1) * min_segment_length > available_length:
        recommended = max(3, available_length // (max_knots + 1))
        warnings.warn(
            f"Dataset may be too small! "
            f"Recommended min_segment_length: {recommended} "
            f"or reduce max_knots to: {available_length // min_segment_length - 1}"
        )

    # Storage for frames
    frames_data = []
    frame_counter = [0]  # Use list to allow modification in nested function

    def save_frame(knots, fitted, bic, rss, n_knots, is_best_k, is_overall_best):
        """Save current configuration as a frame."""
        frame_counter[0] += 1

        fig, ax = plt.subplots(figsize=(width, height))

        # Plot actual data
        ax.scatter(data[time_col], data[value_col],
                  alpha=0.5, s=20, color='steelblue', label='Actual')

        # Plot fitted line
        color = 'darkgreen' if is_overall_best else ('blue' if is_best_k else 'lightgray')
        linewidth = 2.5 if is_overall_best else (2.0 if is_best_k else 1.0)
        alpha = 1.0 if (is_overall_best or is_best_k) else 0.5
        ax.plot(data[time_col], fitted, color=color, linewidth=linewidth,
                alpha=alpha, label='Fitted')

        # Add vertical lines for knots
        if len(knots) > 0:
            knot_times = data[time_col].iloc[knots]
            for kt in knot_times:
                ax.axvline(x=kt, color='red', linestyle='--', alpha=0.7, linewidth=0.8)

        # Title and subtitle
        subtitle = "⭐ NEW OVERALL BEST!" if is_overall_best else (
            "✓ Best for this knot count" if is_best_k else ""
        )
        title_color = 'darkgreen' if is_overall_best else ('blue' if is_best_k else 'black')

        ax.set_title(
            f"Testing {n_knots} Knot(s) | Config #{frame_counter[0]}\n{subtitle}",
            fontsize=14, fontweight='bold', color=title_color
        )

        # Labels
        ax.set_xlabel(time_col, fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)

        # BIC annotation
        ax.text(0.02, 0.98, f"BIC: {bic:.2f}\nKnots: {n_knots}\nRSS: {rss:.2f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save frame
        frame_path = output_path / f"frame_{frame_counter[0]:04d}.png"
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        frames_data.append(str(frame_path))

        if verbose and frame_counter[0] % 10 == 0:
            print(f"  Captured {frame_counter[0]} frames...")

    # Run grid search with frame capture for EVERY configuration
    if verbose:
        print(f"\n{'='*60}")
        print("Running Grid Search with Frame Capture...")
        print(f"{'='*60}\n")

    # Replicate the grid search logic from piecewise_regression
    # but capture frames for EVERY configuration
    from sklearn.linear_model import LinearRegression

    # Prepare data
    df = data.copy()
    df['time_index'] = np.arange(len(df))
    n = len(df)
    min_idx = int(np.ceil(n * edge_buffer))
    max_idx = int(np.floor(n * (1 - edge_buffer)))

    def fit_piecewise_local(knots):
        """Fit piecewise model for given knots."""
        knots = sorted(knots)

        # Build design matrix
        X = np.ones((n, 1))
        X = np.column_stack([X, df['time_index'].values])

        for k in knots:
            X = np.column_stack([X, np.maximum(df['time_index'].values - k, 0)])

        y = df[value_col].values

        # Fit model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        n_params = 2 + len(knots)

        return {
            'model': model,
            'rss': rss,
            'n_params': n_params,
            'knots': knots,
            'fitted': y_pred
        }

    def calc_bic_local(rss, n, k):
        """Calculate BIC."""
        return n * np.log(rss / n) + k * np.log(n)

    def generate_candidates_local(n_knots, min_idx, max_idx, min_segment):
        """Generate candidate knot positions."""
        if n_knots == 0:
            return [[]]

        total_min_length = (n_knots + 1) * min_segment
        available_length = max_idx - min_idx + 1

        if total_min_length > available_length:
            return []

        candidates = []

        if n_knots == 1:
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
            segment_length = (max_idx - min_idx) / (n_knots + 1)
            base_positions = np.linspace(
                min_idx + segment_length,
                max_idx - segment_length,
                num=n_knots
            )
            base_positions = np.round(base_positions).astype(int)
            search_window = int(np.round(segment_length * 0.3))

            if n_knots == 3:
                offsets = np.linspace(-search_window, search_window, num=5)
                for offset1 in offsets:
                    for offset2 in offsets:
                        for offset3 in offsets:
                            pos = base_positions + np.array([offset1, offset2, offset3])
                            pos = np.round(pos).astype(int)

                            all_points = np.concatenate([[min_idx], pos, [max_idx]])
                            diffs = np.diff(all_points)

                            if np.all(diffs >= min_segment):
                                candidates.append(pos.tolist())
            else:
                candidates = [base_positions.tolist()]

        return candidates

    # Test different numbers of knots and capture ALL configurations
    results = []
    overall_best_bic = np.inf

    for k in range(max_knots + 1):
        if verbose:
            print(f"\nTesting {k} knot(s)...")

        candidates = generate_candidates_local(k, min_idx, max_idx, min_segment_length)

        if len(candidates) == 0:
            continue

        best_bic_k = np.inf
        best_knots_k = None
        best_model_k = None
        best_rss_k = None
        best_fitted_k = None

        # Test each candidate and save frame
        for i, knots in enumerate(candidates):
            fit = fit_piecewise_local(knots)
            bic = calc_bic_local(fit['rss'], n, fit['n_params'])

            # Check if this is best for current k
            is_best_k = bic < best_bic_k
            if is_best_k:
                best_bic_k = bic
                best_knots_k = knots
                best_model_k = fit
                best_rss_k = fit['rss']
                best_fitted_k = fit['fitted']

            # Check if this is overall best
            is_overall_best = bic < overall_best_bic
            if is_overall_best:
                overall_best_bic = bic

            # Save frame for this configuration
            save_frame(knots, fit['fitted'], bic, fit['rss'], k, is_best_k, is_overall_best)

        results.append({
            'n_knots': k,
            'knots': best_knots_k,
            'bic': best_bic_k,
            'rss': best_rss_k,
            'model': best_model_k,
            'n_candidates': len(candidates)
        })

        if verbose:
            print(f"  Best BIC: {best_bic_k:.2f} | Tested {len(candidates)} configurations")

    # Find optimal
    bic_values = [r['bic'] for r in results]
    optimal_idx = np.argmin(bic_values)
    optimal = results[optimal_idx]

    # Create result structure matching piecewise_regression output
    df['fitted'] = optimal['model']['fitted']

    if optimal['knots'] and len(optimal['knots']) > 0:
        knot_dates = df[time_col].iloc[optimal['knots']].tolist()
    else:
        knot_dates = None

    bic_scores = pd.DataFrame({
        'n_knots': [r['n_knots'] for r in results],
        'bic': bic_values,
        'rss': [r['rss'] for r in results]
    })

    result = {
        'optimal_knots': optimal['n_knots'],
        'knot_positions': optimal['knots'] if optimal['knots'] else [],
        'knot_dates': knot_dates,
        'bic_scores': bic_scores,
        'model': optimal['model'],
        'data': df,
        'all_results': results
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Frames captured: {len(frames_data)}")
        print(f"{'='*60}")

    if len(frames_data) == 0:
        warnings.warn("No frames were captured!")
        return {
            'regression_result': result,
            'gif_path': None,
            'frames_dir': str(output_path),
            'frames_captured': 0
        }

    # Create GIF
    if verbose:
        print(f"\nCreating GIF with Pillow...")

    gif_path = output_path / gif_name

    # Load all frames
    images = []
    for frame_path in frames_data:
        images.append(Image.open(frame_path))

    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    # Close images
    for img in images:
        img.close()

    if verbose:
        print(f"✓ GIF created: {gif_path}")
        print(f"  Size: {gif_path.stat().st_size / 1024:.1f} KB")
        print(f"  Frames: {len(frames_data)}")
        print(f"  FPS: {fps}")

    return {
        'regression_result': result,
        'gif_path': str(gif_path),
        'frames_dir': str(output_path),
        'frames_captured': len(frames_data)
    }


def record_piecewise_search_plotly(
    data: pd.DataFrame,
    time_col: str = "date",
    value_col: str = "value",
    max_knots: int = 3,
    min_segment_length: int = 30,
    edge_buffer: float = 0.05,
    grid_resolution: int = 20,
    output_file: str = "grid_search_animation.html",
    verbose: bool = True,
    dark_mode: bool = False
) -> Dict:
    """
    Create interactive Plotly animation of piecewise regression grid search.

    Alternative to GIF-based animation, creates an interactive HTML file with
    play/pause controls and slider. Captures ALL tested configurations.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    time_col : str
        Name of the time column
    value_col : str
        Name of the value column
    max_knots : int
        Maximum number of knots to test
    min_segment_length : int
        Minimum observations per segment
    edge_buffer : float
        Edge buffer percentage
    grid_resolution : int
        Grid resolution
    output_file : str
        Output HTML filename
    verbose : bool
        Print progress
    dark_mode : bool
        Use dark theme for plots (default: False)

    Returns
    -------
    dict
        Results with path to HTML file

    Notes
    -----
    Creates interactive HTML animation viewable in browser.
    No external dependencies beyond plotly.
    Captures every configuration tested during grid search.
    """
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression

    if verbose:
        print("\n" + "=" * 60)
        print("Creating Interactive Plotly Animation")
        print("=" * 60)

    # Replicate grid search logic to capture all configurations
    df = data.copy()
    df['time_index'] = np.arange(len(df))
    n = len(df)
    min_idx = int(np.ceil(n * edge_buffer))
    max_idx = int(np.floor(n * (1 - edge_buffer)))

    def fit_piecewise_local(knots):
        """Fit piecewise model for given knots."""
        knots = sorted(knots)
        X = np.ones((n, 1))
        X = np.column_stack([X, df['time_index'].values])
        for k in knots:
            X = np.column_stack([X, np.maximum(df['time_index'].values - k, 0)])
        y = df[value_col].values
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        n_params = 2 + len(knots)
        return {
            'model': model,
            'rss': rss,
            'n_params': n_params,
            'knots': knots,
            'fitted': y_pred
        }

    def calc_bic_local(rss, n, k):
        return n * np.log(rss / n) + k * np.log(n)

    def generate_candidates_local(n_knots, min_idx, max_idx, min_segment):
        """Generate candidate knot positions."""
        if n_knots == 0:
            return [[]]
        total_min_length = (n_knots + 1) * min_segment
        available_length = max_idx - min_idx + 1
        if total_min_length > available_length:
            return []
        candidates = []
        if n_knots == 1:
            n_positions = min(grid_resolution, max_idx - min_idx - 2 * min_segment)
            if n_positions > 0:
                positions = np.linspace(min_idx + min_segment, max_idx - min_segment, num=int(n_positions))
                positions = np.round(positions).astype(int)
                candidates = [[int(p)] for p in positions]
        elif n_knots == 2:
            # For 2 knots, create a proper grid search
            available_range = max_idx - min_idx - 2 * min_segment
            n_positions = min(grid_resolution, available_range)
            if n_positions > 0:
                all_positions = np.linspace(min_idx + min_segment, max_idx - min_segment, num=int(n_positions))
                all_positions = np.round(all_positions).astype(int)
                all_positions = np.unique(all_positions)
                for i, pos1 in enumerate(all_positions):
                    for pos2 in all_positions[i+1:]:
                        if pos2 - pos1 >= min_segment:
                            candidates.append([int(pos1), int(pos2)])
        else:
            segment_length = (max_idx - min_idx) / (n_knots + 1)
            base_positions = np.linspace(min_idx + segment_length, max_idx - segment_length, num=n_knots)
            base_positions = np.round(base_positions).astype(int)
            search_window = int(np.round(segment_length * 0.3))
            if n_knots == 3:
                offsets = np.linspace(-search_window, search_window, num=5)
                for offset1 in offsets:
                    for offset2 in offsets:
                        for offset3 in offsets:
                            pos = base_positions + np.array([offset1, offset2, offset3])
                            pos = np.round(pos).astype(int)
                            all_points = np.concatenate([[min_idx], pos, [max_idx]])
                            diffs = np.diff(all_points)
                            if np.all(diffs >= min_segment):
                                candidates.append(pos.tolist())
            else:
                candidates = [base_positions.tolist()]
        return candidates

    # Test all configurations and create frames
    frames = []
    results = []
    overall_best_bic = np.inf
    frame_count = 0

    for k in range(max_knots + 1):
        if verbose:
            print(f"\nTesting {k} knot(s)...")

        candidates = generate_candidates_local(k, min_idx, max_idx, min_segment_length)
        if len(candidates) == 0:
            continue

        best_bic_k = np.inf
        best_knots_k = None
        best_model_k = None
        best_rss_k = None

        for knots in candidates:
            fit = fit_piecewise_local(knots)
            bic = calc_bic_local(fit['rss'], n, fit['n_params'])

            # Track best for this k
            is_best_k = bic < best_bic_k
            if is_best_k:
                best_bic_k = bic
                best_knots_k = knots
                best_model_k = fit
                best_rss_k = fit['rss']

            # Track overall best
            is_overall_best = bic < overall_best_bic
            if is_overall_best:
                overall_best_bic = bic

            # Create frame for this configuration
            # Adjust colors for dark mode
            if dark_mode:
                color = 'lightgreen' if is_overall_best else ('lightblue' if is_best_k else 'lightgray')
                marker_color = 'lightsteelblue'
            else:
                color = 'green' if is_overall_best else ('blue' if is_best_k else 'gray')
                marker_color = 'steelblue'

            width = 3 if is_overall_best else (2 if is_best_k else 1.5)
            opacity = 1.0 if (is_overall_best or is_best_k) else 0.75

            subtitle = "⭐ NEW OVERALL BEST!" if is_overall_best else ("✓ Best for k" if is_best_k else "")

            # Build traces for this frame
            traces = [
                go.Scatter(x=data[time_col], y=data[value_col],
                          mode='markers', name='Actual',
                          marker=dict(color=marker_color, opacity=0.5, size=5)),
                go.Scatter(x=data[time_col], y=fit['fitted'],
                          mode='lines', name='Fitted',
                          line=dict(color=color, width=width),
                          opacity=opacity)
            ]

            # Create shapes for vertical lines at knots
            shapes = []
            if len(knots) > 0:
                # Convert knot indices to actual time values
                knot_times = [data[time_col].iloc[k] for k in knots]

                knot_color = 'lightgreen' if dark_mode else 'darkgreen'
                for knot_time in knot_times:
                    shapes.append(dict(
                        type='line',
                        x0=knot_time,
                        y0=0,
                        x1=knot_time,
                        y1=1,
                        yref='paper',
                        line=dict(color=knot_color, width=2, dash='dash')
                    ))

            frame = go.Frame(
                data=traces,
                name=str(frame_count),
                layout=go.Layout(
                    title=f"Testing {k} knot(s) | Config #{frame_count + 1}<br><sub>{subtitle}</sub><br><sub>BIC: {bic:.2f} | RSS: {fit['rss']:.2f}</sub>",
                    shapes=shapes
                )
            )
            frames.append(frame)
            frame_count += 1

        results.append({
            'n_knots': k,
            'knots': best_knots_k,
            'bic': best_bic_k,
            'rss': best_rss_k,
            'model': best_model_k,
            'n_candidates': len(candidates)
        })

        if verbose:
            print(f"  Best BIC: {best_bic_k:.2f} | Tested {len(candidates)} configurations")

    # Find optimal
    bic_values = [r['bic'] for r in results]
    optimal_idx = np.argmin(bic_values)
    optimal = results[optimal_idx]

    df['fitted'] = optimal['model']['fitted']
    if optimal['knots'] and len(optimal['knots']) > 0:
        knot_dates = df[time_col].iloc[optimal['knots']].tolist()
    else:
        knot_dates = None

    bic_scores = pd.DataFrame({
        'n_knots': [r['n_knots'] for r in results],
        'bic': bic_values,
        'rss': [r['rss'] for r in results]
    })

    result = {
        'optimal_knots': optimal['n_knots'],
        'knot_positions': optimal['knots'] if optimal['knots'] else [],
        'knot_dates': knot_dates,
        'bic_scores': bic_scores,
        'model': optimal['model'],
        'data': df,
        'all_results': results
    }

    # Create figure with dark mode support
    layout_config = {
        'title': "Piecewise Regression Grid Search",
        'xaxis': dict(title=time_col),
        'yaxis': dict(title=value_col),
        'updatemenus': [{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 500}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        'sliders': [{
            'steps': [
                {'args': [[f.name], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                 'label': f.name, 'method': 'animate'}
                for f in frames
            ],
            'active': 0
        }]
    }

    # Apply dark theme if requested
    if dark_mode:
        layout_config.update({
            'template': 'plotly_dark',
            'paper_bgcolor': '#0e1117',
            'plot_bgcolor': '#0e1117'
        })

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(**layout_config),
        frames=frames
    )

    # Save
    output_path = Path(output_file)
    fig.write_html(output_path)

    if verbose:
        print(f"✓ Interactive animation saved: {output_path}")

    return {
        'regression_result': result,
        'html_path': str(output_path),
        'frames_created': len(frames)
    }
