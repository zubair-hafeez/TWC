"""Keep verification imports inside the hidden checks directory."""

from pathlib import Path
import sys

CHECKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKS.parent))
sys.path.insert(0, str(CHECKS))
