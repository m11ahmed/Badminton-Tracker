"""Run the project CLI without requiring an editable package installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from badminton_tracker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())