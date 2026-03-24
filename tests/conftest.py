import multiprocessing as mp
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

@pytest.fixture(scope="function", autouse=True)
def mock_buffer_mp_primitives():
    with patch("tricked.training.buffer.mp.Lock"):
        with patch("tricked.training.buffer.mp.Value"):
            yield
