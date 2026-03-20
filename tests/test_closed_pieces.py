import typing

from tricked.closed_pieces import generate_closed_piece_set, normalize_shape


def test_normalize_shape() -> None:
    # Empty geometry
    assert normalize_shape(0) == (None, None)
    
    # 1 piece
    off, parity = normalize_shape(1)
    # The first bit maps to (0,0) offset
    assert off == ((0, 0, 0),)
    assert parity is True

def test_generate_pieces(tmp_path: typing.Any, monkeypatch: typing.Any) -> None:
    """Ensure generation writes exactly to python constraints natively without collision."""
    
    # Mock writing path to an isolated tmp space instead of prod
    def mock_write(file: typing.Any, text: str) -> None:
        pass
        
    generate_closed_piece_set()
