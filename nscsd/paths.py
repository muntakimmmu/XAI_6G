"""Location of Sentinel's public repository and its released paper-mode run.

Set SENTINEL_REPO to a clone of https://github.com/AliAlfatemi/sentinel-ddos;
by default it is expected next to this repository (../sentinel-ddos).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL_REPO = os.environ.get("SENTINEL_REPO", os.path.join(ROOT, "..", "sentinel-ddos"))
SENTINEL_RUN = os.path.join(SENTINEL_REPO, "outputs", "run_20260523_200520_paper")
SENTINEL_MODELS = os.path.join(SENTINEL_RUN, "models")
