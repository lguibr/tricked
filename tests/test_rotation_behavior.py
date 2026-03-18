
from tricked.env.pieces import STANDARD_PIECES
from tricked_web.server import ROTATION_MAP_RIGHT


def test_rotation_triangle_invariants() -> None:
    """Verifies that rotating a piece preserves its exact triangle count."""
    for p_id in ROTATION_MAP_RIGHT:
        # Find exactly how many triangles this piece has
        # Every mask in STANDARD_PIECES[p_id] must have the exact same bit count.
        # Find the first valid mask.
        m_start = next((m for m in STANDARD_PIECES[p_id] if m != 0), 0)
        assert m_start != 0, f"Piece {p_id} has no valid placements."
        
        triangle_count = m_start.bit_count()
        
        rotated_id = ROTATION_MAP_RIGHT[p_id]
        
        m_rot = next((m for m in STANDARD_PIECES[rotated_id] if m != 0), 0)
        assert m_rot != 0, f"Rotated Piece {rotated_id} has no valid placements."
        
        rot_triangle_count = m_rot.bit_count()
        
        assert triangle_count == rot_triangle_count, f"Rotation changed shape! p_id {p_id} ({triangle_count} tris) -> p_id {rotated_id} ({rot_triangle_count} tris)"


def test_rotation_six_cycles() -> None:
    """Verifies that rotating any piece exactly 6 times returns to the identical starting piece ID."""
    for start_id in ROTATION_MAP_RIGHT.keys():
        curr_id = start_id
        for _ in range(6):
            assert curr_id in ROTATION_MAP_RIGHT, f"Piece {curr_id} has no valid right rotation!"
            curr_id = ROTATION_MAP_RIGHT[curr_id]
            
        assert curr_id == start_id, f"Rotating piece {start_id} 6 times resulted in piece {curr_id} instead of itself."
