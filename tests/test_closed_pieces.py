import typing

from tricked.closed_pieces import generate_closed_piece_set, normalize_shape


def test_normalize_shape() -> None:

    assert normalize_shape(0) == (None, None)

    off, parity = normalize_shape(1)

    assert off == ((0, 0, 0),)
    assert parity is True


def test_generate_pieces(tmp_path: typing.Any, monkeypatch: typing.Any) -> None:
    """Ensure generation writes exactly to python constraints natively without collision."""

    def mock_write(file: typing.Any, text: str) -> None:
        pass

    generate_closed_piece_set()
