import pytest
from src.tricked import symmetry

def test_symmetry_permutations():
    """Verify D12 symmetry computation completes without dropping topological mapping."""
    # Since the module is entirely loaded on import, D12_PERMUTATIONS is already built
    assert len(symmetry.D12_PERMUTATIONS) == 12
    for p in symmetry.D12_PERMUTATIONS:
        assert len(set(p)) == 96

def test_symmetry_row_cols():
    assert symmetry.get_row_col(0) == (0, 0)
    assert symmetry.is_up(0, 0) is True
    assert symmetry.is_up(0, 1) is False
    assert symmetry.is_up(4, 0) is False
    assert symmetry.is_up(4, 1) is True

def test_symmetry_mirror():
    assert symmetry.mirror_transform(1.0, 1.0) == (-1.0, 1.0)
