from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point the API at a throwaway SQLite DB and temp dirs *before* any `api` import.
_TMP = Path(tempfile.mkdtemp(prefix="terrashift-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["TERRASHIFT_IMAGERY_ROOT"] = str(_TMP / "imagery")
os.environ["TERRASHIFT_ARTIFACT_DIR"] = str(_TMP / "artifacts")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from terrashift.config import RFTrainConfig  # noqa: E402
from terrashift.data.synthetic import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def synthetic_dir(tmp_root: Path) -> Path:
    out = tmp_root / "imagery" / "synthetic"
    generate_dataset(out, n_per_class=12, size=32, seed=1)
    return out


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def small_rf_cfg() -> RFTrainConfig:
    return RFTrainConfig(n_estimators=25, n_jobs=1)
