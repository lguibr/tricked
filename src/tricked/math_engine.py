from tricked.env.pieces import ALL_MASKS, STANDARD_PIECES, TOTAL_TRIANGLES
from tricked.symmetry import generate_board_coordinates


class CoordinateShadowEngine:
    def __init__(self):
        # Coordinates from symmetry model:
        # pts = [(i, x, y, up), ...]
        self.pts = generate_board_coordinates()
        self.idx_to_coord = {}
        self.coord_to_idx = {}

        # Build strict 3-axis coordinates
        # Q axis = -60 deg
        # R axis = +60 deg
        # S axis = horizontal (0 deg)
        # Because we need perfect integer lines to validate line clears.

        # Using the mathematical projection where triangles form continuous lines.
        # A horizontal line (S-axis) shares the same row (Y coordinate essentially).
        # A +60 deg line shares the same Q coordinate.
        # A -60 deg line shares the same R coordinate.

        # Row slices are straightforward: 8 horizontal lines
        self.lines_s = self._extract_lines(axis="y")  # horizontal

        # Q slices (x - y/sqrt(3))
        # R slices (x + y/sqrt(3))
        # Since triangles point up/down, standard barycentric coordinates give integer axes.
        # Let's cleanly derive these lines programmatically from ALL_MASKS.

        self.mask_to_indices = {}
        for i, m in enumerate(ALL_MASKS):
            indices = set()
            for bit in range(TOTAL_TRIANGLES):
                if (m & (1 << bit)) != 0:
                    indices.add(bit)
            self.mask_to_indices[m] = frozenset(indices)

    def _extract_lines(self, axis: str):
        pass

    def get_lines_from_masks(self) -> list[set[int]]:
        return [set(idx) for idx in self.mask_to_indices.values()]

    def apply_move(self, board_mask: int, piece_id: int, piece_idx: int) -> tuple[int, int, int]:
        """
        Pure python mathematical apply move.
        Returns (next_board_mask, score_gained, lines_cleared)
        """
        p_mask = STANDARD_PIECES[piece_id][piece_idx]
        if p_mask == 0 or (board_mask & p_mask) != 0:
            return -1, 0, 0  # Invalid

        next_board = board_mask | p_mask
        score_gained = bin(p_mask).count("1")

        lines_cleared_count = 0
        cleared_mask = 0

        for mask_val, idx_set in self.mask_to_indices.items():
            if (next_board & mask_val) == mask_val:
                cleared_mask |= mask_val
                lines_cleared_count += 1

        if lines_cleared_count > 0:
            next_board &= ~cleared_mask
            score_gained += bin(cleared_mask).count("1") * 2

        return next_board, score_gained, lines_cleared_count


if __name__ == "__main__":
    engine = CoordinateShadowEngine()
    print(f"Shadow Engine active. Loaded {len(engine.get_lines_from_masks())} strict axis lines.")
