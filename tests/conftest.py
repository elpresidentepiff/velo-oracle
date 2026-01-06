"""Configure test environment for Cognitive Mesh."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from mesh.api.server import get_engine


@pytest.fixture(autouse=True)
def reset_decision_engine() -> None:
    """Ensure each test runs against a clean engine instance."""

    engine = get_engine()
    engine.reset()
    try:
        yield
    finally:
        engine.reset()
