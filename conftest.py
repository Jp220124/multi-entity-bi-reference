"""Repository-root conftest.

Adds the repo root to ``sys.path`` so tests can import the ``agents``,
``context``, ``orchestrator``, and ``schemas`` packages without a
prior ``pip install -e .``. Keeps the "clone → pytest" flow one step
instead of two for reviewers skimming the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
