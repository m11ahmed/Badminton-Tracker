"""Keep all test fixtures and encoded media inside this project."""
from pathlib import Path
import sys
import tempfile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def project_tmp_path():
    staging = PROJECT_ROOT / "working" / "tests"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case_", dir=staging) as directory:
        yield Path(directory)
