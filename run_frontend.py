#!/usr/bin/env python
# ===============================================
# Solar Radiation Prediction Dashboard Runner
# Run: python run_frontend.py
# Or:  streamlit run frontend/app.py
# ===============================================

import os
import sys
import subprocess


def get_project_python(script_dir):
    """Prefer the local virtual environment when it exists."""
    if os.name == "nt":
        venv_python = os.path.join(script_dir, "solar_env", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(script_dir, "solar_env", "bin", "python")

    return venv_python if os.path.exists(venv_python) else sys.executable


def main():
    """Launch the Streamlit dashboard."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = get_project_python(script_dir)

    # Path to the frontend app
    app_path = os.path.join(script_dir, "frontend", "app.py")

    if not os.path.exists(app_path):
        print(f"Error: Frontend app not found at {app_path}")
        sys.exit(1)

    print("🌤 Starting Solar Radiation Prediction Dashboard...")
    print(f"📂 App location: {app_path}")
    print("-" * 50)

    # Run streamlit
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", app_path], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("\nTry running manually:")
        print(f"  streamlit run {app_path}")


if __name__ == "__main__":
    main()
