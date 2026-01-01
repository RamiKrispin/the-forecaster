#!/usr/bin/env python3
"""
Quick launcher for the Piecewise Regression Streamlit App
"""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "Python" / "piecewise_app.py"

    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    print("🚀 Launching Piecewise Regression App...")
    print(f"📍 App location: {app_path}")
    print("🌐 Opening browser at http://localhost:8501\n")

    try:
        subprocess.run(["streamlit", "run", str(app_path)])
    except FileNotFoundError:
        print("\n❌ Error: Streamlit not found!")
        print("\nPlease install streamlit:")
        print("  pip install streamlit")
        print("\nOr install with the forecaster module:")
        print("  pip install -e \".[streamlit]\"")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
