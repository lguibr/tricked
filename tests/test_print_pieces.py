import pytest
from src.tricked.print_pieces import print_piece, str_piece_mask

def test_print_pieces_empty():
    assert str_piece_mask(0) == ""

def test_print_pieces_content():
    # Print piece 0
    print_piece(0)
    
    # 1 triangle at 0
    out = str_piece_mask(1)
    assert "^" in out or "v" in out
