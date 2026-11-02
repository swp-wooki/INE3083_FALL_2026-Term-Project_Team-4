"""Repository-local input and output locations."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LOWRISK_DATA_DIR", ROOT / "data")).resolve()
OUTPUT = Path(os.environ.get("LOWRISK_OUTPUT_DIR", ROOT / "results/reproduced")).resolve()
PAPER = ROOT / "paper"
