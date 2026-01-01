"""
Piecewise Regression Streamlit App

Interactive web application for exploring piecewise regression with automatic
breakpoint detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
src_path = repo_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from forecaster import (
    piecewise_regression,
    plot_piecewise_comparison,
    record_piecewise_search_plotly
)

# Page configuration
st.set_page_config(
    page_title="Piecewise Regression",
    page_icon="📈",
    layout="wide",
)

# Title and description
st.title("📈 Auto-Detect of Trend Change Points with Piecewise Regression")
st.markdown("""
Use grid search approach to identify the optimal number of knots and their positions. More details are available in the Forecaster Newsletter
""")

# Sidebar for controls
st.sidebar.header("⚙️ Configuration")

# Data loading
st.sidebar.subheader("Data")
data_source = st.sidebar.selectbox(
    "Data Source",
    ["California Natural Gas Consumers", "Upload CSV"]
)

@st.cache_data
def load_data(source):
    """Load and preprocess time series data."""
    if source == "California Natural Gas Consumers":
        path = "https://raw.githubusercontent.com/RamiKrispin/the-forecaster/refs/heads/main/data/ca_natural_gas_consumers.csv"
        ts = pd.read_csv(path)
        ts = ts[['index', 'y']]
        ts = ts.sort_values('index')
        ts = ts[ts['index'] > 1986].reset_index(drop=True)
        time_col = 'index'
        value_col = 'y'
        title = "Number of Natural Gas Consumers in California"
    else:
        # Placeholder for file upload
        ts = None
        time_col = None
        value_col = None
        title = None

    return ts, time_col, value_col, title

# Load data
ts, time_col, value_col, title = load_data(data_source)

if ts is not None:
    # Display data info
    st.sidebar.info(f"**Observations:** {len(ts)}")

    # Piecewise regression parameters
    st.sidebar.subheader("Regression Parameters")

    max_knots = st.sidebar.slider(
        "Maximum Knots",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of breakpoints to test"
    )

    min_segment_length = st.sidebar.slider(
        "Minimum Segment Length",
        min_value=5,
        max_value=30,
        value=8,
        help="Minimum observations required in each segment"
    )

    edge_buffer = st.sidebar.slider(
        "Edge Buffer (%)",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Percentage of data to exclude from edges"
    )

    grid_resolution = st.sidebar.slider(
        "Grid Resolution",
        min_value=5,
        max_value=30,
        value=20,
        help="Number of candidate positions per knot"
    )

    # Run button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

    # Main content area
    st.header("📊 Time Series Data")

    # Show data preview
    with st.expander("View Data", expanded=False):
        st.dataframe(ts.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(ts)} observations")

    # Plot original time series
    st.subheader("Original Time Series")

    import plotly.graph_objects as go

    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(
        x=ts[time_col],
        y=ts[value_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4)
    ))

    fig_original.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Number of Consumers",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig_original, use_container_width=True)

    # Run piecewise regression
    if run_analysis:
        with st.spinner("Running piecewise regression grid search..."):
            try:
                result = piecewise_regression(
                    data=ts,
                    time_col=time_col,
                    value_col=value_col,
                    max_knots=max_knots,
                    min_segment_length=min_segment_length,
                    edge_buffer=edge_buffer,
                    grid_resolution=grid_resolution,
                    verbose=False
                )

                # Store result in session state
                st.session_state['result'] = result
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                result = None

    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state['result']

        st.header("📈 Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Optimal Knots", result['optimal_knots'])

        with col2:
            optimal_bic = result['bic_scores'][result['bic_scores']['n_knots'] == result['optimal_knots']]['bic'].values[0]
            st.metric("BIC Score", f"{optimal_bic:.2f}")

        with col3:
            optimal_rss = result['bic_scores'][result['bic_scores']['n_knots'] == result['optimal_knots']]['rss'].values[0]
            st.metric("RSS", f"{optimal_rss:.2f}")

        with col4:
            configs_tested = sum([r['n_candidates'] for r in result['all_results']])
            st.metric("Configs Tested", configs_tested)

        # Display knot positions
        if result['knot_dates']:
            st.info(f"**Breakpoint Years:** {', '.join(map(str, result['knot_dates']))}")

        # Visualization
        st.subheader("Model Selection and Fit")
        st.markdown("""
        Combined view showing both the model selection process (BIC) and the final fitted model.
        """)

        fig_comparison = plot_piecewise_comparison(result, time_col=time_col, value_col=value_col)
        fig_comparison.update_layout(height=500)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Grid Search Animation
        st.subheader("🎬 Grid Search Animation")

        show_animation = st.checkbox(
            "Generate Interactive Animation",
            value=False,
            help="Creates an animated visualization showing all configurations tested during grid search. May take a moment to generate."
        )

        if show_animation:
            with st.spinner("Generating animation (testing all configurations)..."):
                try:
                    # Create a temporary file for the HTML output
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                        tmp_path = tmp_file.name

                    # Generate the animation
                    animation_result = record_piecewise_search_plotly(
                        data=ts,
                        time_col=time_col,
                        value_col=value_col,
                        max_knots=max_knots,
                        min_segment_length=min_segment_length,
                        edge_buffer=edge_buffer,
                        grid_resolution=grid_resolution,
                        output_file=tmp_path,
                        verbose=False,
                        dark_mode=True
                    )

                    # Read the HTML file
                    with open(tmp_path, 'r') as f:
                        html_content = f.read()

                    # Display the animation
                    st.components.v1.html(html_content, height=600, scrolling=True)

                    st.caption(f"Animation shows {animation_result['frames_created']} configurations tested during grid search")

                    # Clean up temp file
                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Error generating animation: {str(e)}")


        # Model details
        with st.expander("📋 Detailed Results"):
            st.subheader("All Tested Configurations")

            results_df = pd.DataFrame([
                {
                    'Knots': r['n_knots'],
                    'BIC': f"{r['bic']:.2f}",
                    'RSS': f"{r['rss']:.2f}",
                    'Configurations': r['n_candidates']
                }
                for r in result['all_results']
            ])

            st.dataframe(results_df, use_container_width=True)

            st.subheader("Optimal Model Details")
            st.json({
                'n_knots': result['optimal_knots'],
                'knot_positions': result['knot_positions'],
                'knot_dates': result['knot_dates'],
                'bic': float(optimal_bic),
                'rss': float(optimal_rss)
            })

else:
    st.warning("Please select a data source or upload a CSV file.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app demonstrates automatic breakpoint detection using:
- **Piecewise Linear Regression**
- **Grid Search** for optimal knot positions
- **BIC** for model selection

Built with the `forecaster` Python module.
""")
