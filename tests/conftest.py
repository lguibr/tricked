import multiprocessing as mp
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


