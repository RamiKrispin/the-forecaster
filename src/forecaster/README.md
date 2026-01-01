# Forecaster Module

A Python module providing supporting functions for time series forecasting articles and notebooks in this repository.

## Structure

```
src/forecaster/
├── __init__.py          # Package initialization
├── piecewise.py         # Piecewise regression functions
├── plotly_viz.py        # Interactive Plotly visualizations
├── animation.py         # Grid search animation functions
├── visualization.py     # Matplotlib plotting utilities
└── utils.py             # Helper functions
```

## Installation

From the repository root:

```bash
# Basic installation
pip install -e .

# With animation support (GIF creation)
pip install -e ".[animation]"

# With development tools
pip install -e ".[dev]"
```

## Usage

```python
from forecaster import piecewise_regression, plot_bic_scores, plot_knots

# Fit piecewise regression
result = piecewise_regression(
    data=df,
    time_col='date',
    value_col='value',
    max_knots=3
)

# Create interactive plots
bic_fig = plot_bic_scores(result)
bic_fig.show()

knots_fig = plot_knots(result, time_col='date', value_col='value')
knots_fig.show()
```

## Functions

### Piecewise Regression

**`piecewise_regression(data, time_col, value_col, ...)`**
- Automatic breakpoint detection using grid search and BIC selection
- Returns optimal knots, fitted values, and model statistics

### Interactive Visualizations (Plotly)

**`plot_bic_scores(result)`**
- Interactive plot of BIC scores across different knot counts
- Highlights optimal model with annotation

**`plot_knots(result, time_col, value_col)`**
- Interactive plot of time series with fitted piecewise regression
- Shows actual data, fitted line, and vertical lines at breakpoints

**`plot_piecewise_comparison(result, time_col, value_col)`**
- Side-by-side comparison of BIC scores and fitted model
- Combined interactive visualization

### Helper Functions

**`create_arc(x0, y0, x1, y1, curvature, length_out)`**
- Generate bezier curve points for curved annotation lines
- Used for custom plotly annotations

**`hex_to_rgba(hex_color, opacity)`**
- Convert hex colors to RGBA format for plotly
- Supports custom transparency

### Animation Functions

**`record_piecewise_search(data, time_col, value_col, ...)`**
- Animate the grid search process as a GIF
- **Captures EVERY configuration tested**, not just best results
- Color-codes configurations: dark green (overall best), blue (best for k), light gray (others)
- Highlights best models during search
- Requires Pillow: `pip install -e ".[animation]"`
- Tip: Reduce `grid_resolution` for faster animations

**`record_piecewise_search_plotly(data, time_col, value_col, ...)`**
- Interactive HTML animation alternative
- **Captures ALL configurations** with color coding
- Play/pause controls and slider
- No additional dependencies beyond plotly

## Examples

### Basic Analysis

```python
import pandas as pd
from forecaster import piecewise_regression, plot_bic_scores, plot_knots

# Load data
df = pd.read_csv('data.csv')

# Run piecewise regression
result = piecewise_regression(
    data=df,
    time_col='date',
    value_col='value',
    max_knots=3,
    min_segment_length=30
)

# Visualize results
plot_bic_scores(result).show()
plot_knots(result, 'date', 'value').show()
```

### Create Animation (GIF)

```python
from forecaster import record_piecewise_search

# Create animated GIF of grid search
result = record_piecewise_search(
    data=df,
    time_col='date',
    value_col='value',
    max_knots=3,
    output_dir='animation',
    gif_name='search.gif',
    fps=2
)

print(f"Animation saved to: {result['gif_path']}")
```

### Create Interactive Animation (HTML)

```python
from forecaster import record_piecewise_search_plotly

# Create interactive HTML animation
result = record_piecewise_search_plotly(
    data=df,
    time_col='date',
    value_col='value',
    max_knots=3,
    output_file='search_animation.html'
)

print(f"Interactive animation saved to: {result['html_path']}")
```

## Development

Add new modules as needed for different forecasting techniques and utilities.
