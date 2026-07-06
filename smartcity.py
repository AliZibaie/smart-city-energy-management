#!/usr/bin/env python3
"""
SmartCity CLI - Project helper for energy forecasting pipeline.
Usage:
    python smartcity.py setup
    python smartcity.py env
    python smartcity.py db:create
    python smartcity.py notebook <phase>
    python smartcity.py streamlit
    python smartcity.py pipeline   # full automatic workflow
"""

import argparse
import subprocess
import sys
import os
import glob
import platform
import shutil
import time

# ---------- configuration ----------
VENV_DIR = ".venv"
NOTEBOOK_DIR = "notebooks"
REQUIREMENTS_FILE = "requirements.txt"
STREAMLIT_APP = "dashboard/app.py"
ENV_EXAMPLE = ".env.example"
ENV_FILE = ".env"

# default MySQL connection (overridden by .env if present)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "smartcity",
    "port": 3306,
}

# phases that the pipeline will run in order
PIPELINE_PHASES = [1, 2, 4]


def get_venv_python():
    """Return path to Python executable inside the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def run(cmd, **kwargs):
    """Thin wrapper for subprocess.run with default check=True."""
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)


def load_env():
    """Load .env file into environment variables if present."""
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def _get_db_config():
    """Merge DB_CONFIG with values from environment (.env)."""
    config = DB_CONFIG.copy()
    config["host"] = os.environ.get("DB_HOST", config["host"])
    config["user"] = os.environ.get("DB_USER", config["user"])
    config["password"] = os.environ.get("DB_PASSWORD", config["password"])
    config["database"] = os.environ.get("DB_NAME", config["database"])
    config["port"] = int(os.environ.get("DB_PORT", config["port"]))
    return config


def setup():
    """Create virtual environment and install dependencies."""
    if not os.path.isdir(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR} ...")
        run([sys.executable, "-m", "venv", VENV_DIR])

    venv_python = get_venv_python()
    print("Upgrading pip ...")
    run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    if os.path.isfile(REQUIREMENTS_FILE):
        print(f"Installing packages from {REQUIREMENTS_FILE} ...")
        try:
            run([venv_python, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        except subprocess.CalledProcessError:
            print("Requirements failed (long path issue). Installing core packages manually...")
            run([venv_python, "-m", "pip", "install", 
                 "jupyter", "notebook", "pymysql", "pandas", "numpy", 
                 "matplotlib", "scikit-learn", "streamlit", "tensorflow"])
    else:
        print(f"Warning: {REQUIREMENTS_FILE} not found. Installing minimal packages...")
        run([venv_python, "-m", "pip", "install", "jupyter", "notebook", "pymysql"])

    print("Setup complete. Activate manually with:")
    if platform.system() == "Windows":
        print(f"  {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"  source {VENV_DIR}/bin/activate")



def launch_notebook(phase):
    """Open Jupyter notebook (using .venv) for a specific project phase."""
    if not os.path.isdir(VENV_DIR):
        sys.exit("Virtual environment not found. Run `python smartcity.py setup` first.")

    pattern = os.path.join(NOTEBOOK_DIR, f"Phase{phase}_*.ipynb")
    notebooks = glob.glob(pattern)

    venv_python = get_venv_python()
    if notebooks:
        notebook_file = notebooks[0]
        print(f"Opening notebook: {notebook_file}")
        # Jupyter notebook will block until user closes it
        run([venv_python, "-m", "jupyter", "notebook", notebook_file])
    else:
        print(f"No notebook found for Phase {phase} with pattern '{pattern}'.")
        print(f"Opening Jupyter in the {NOTEBOOK_DIR} directory instead.")
        run([venv_python, "-m", "jupyter", "notebook", NOTEBOOK_DIR])


def launch_streamlit():
    """Run the Streamlit dashboard app."""
    if not os.path.isdir(VENV_DIR):
        sys.exit("Virtual environment not found. Run `python smartcity.py setup` first.")

    if not os.path.isfile(STREAMLIT_APP):
        sys.exit(f"Streamlit app not found at {STREAMLIT_APP}. Please update STREAMLIT_APP inside the script.")

    venv_python = get_venv_python()
    run([venv_python, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    print(f"Launching Streamlit: {STREAMLIT_APP}")
    run([venv_python, "-m", "streamlit", "run", STREAMLIT_APP])


def env_copy():
    """Copy .env.example to .env, prompting before overwrite if .env already exists."""
    if not os.path.isfile(ENV_EXAMPLE):
        sys.exit(f"File '{ENV_EXAMPLE}' not found in project root. Nothing to copy.")

    if os.path.isfile(ENV_FILE):
        answer = input(f"'{ENV_FILE}' already exists. Overwrite? (y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    print(f"Copied {ENV_EXAMPLE} → {ENV_FILE}")


def db_create():
    """Create the smartcity database in MySQL if it doesn't exist."""
    load_env()
    config = _get_db_config()

    try:
        import pymysql
        conn = pymysql.connect(
            host=config["host"],
            user=config["user"],
            password=config["password"],
            port=config["port"],
            charset="utf8mb4",
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{config['database']}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        print(f"Database `{config['database']}` is ready (created if missing).")
        cursor.close()
        conn.close()
    except ImportError:
        sys.exit("PyMySQL is required for DB creation. Install it via `pip install pymysql`.")
    except Exception as e:
        sys.exit(f"Failed to create database: {e}")


def pipeline():
    """Run the full project pipeline: setup env, DB, then Phase 1 → 2 → 4 (interactive)."""
    print("===== SmartCity Pipeline =====")

    # Step 0: ensure .env exists
    if not os.path.isfile(ENV_FILE):
        print("[1/4] Creating .env from .env.example ...")
        env_copy()
    else:
        print("[1/4] .env already exists, skipping copy.")

    # Step 1: setup virtual environment
    print("[2/4] Setting up virtual environment ...")
    if not os.path.isdir(VENV_DIR):
        setup()
    else:
        print("Virtual environment already exists. Skipping setup.")

    # Step 2: create database
    load_env()
    print("[3/4] Creating database ...")
    db_create()

    # Step 3: run phases sequentially
    print("[4/4] Opening notebooks in order:")
    for phase in PIPELINE_PHASES:
        print(f"\n>>> Phase {phase} notebook will open. Complete and close the notebook to proceed.")
        input("Press Enter to continue...")
        launch_notebook(phase)

    print("\n===== Pipeline complete =====")
    print("You can now run `python smartcity.py streamlit` to start the dashboard.")


# ---------- CLI entry point ----------
def main():
    parser = argparse.ArgumentParser(
        description="SmartCity CLI – helper for the energy forecasting project"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("setup", help="Create venv and install requirements")
    subparsers.add_parser("env", help="Copy .env.example to .env")
    subparsers.add_parser("db:create", help="Create the smartcity database in MySQL")
    subparsers.add_parser("streamlit", help="Run the Streamlit dashboard")
    subparsers.add_parser("pipeline", help="Full automatic workflow (env → setup → db → Phase1→2→4)")

    nb_parser = subparsers.add_parser("notebook", help="Open a Jupyter notebook for a given phase")
    nb_parser.add_argument("phase", type=int, help="Phase number (1-4)")

    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "env":
        env_copy()
    elif args.command == "db:create":
        db_create()
    elif args.command == "streamlit":
        launch_streamlit()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "notebook":
        launch_notebook(args.phase)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
