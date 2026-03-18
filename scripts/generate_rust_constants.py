import os
import sys

# Ensure the module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tricked.env.pieces import ALL_MASKS, STANDARD_PIECES


def generate_rust_constants():
    num_pieces = len(STANDARD_PIECES)

    out = []
    # Write ALL_MASKS
    out.append(f"pub const ALL_MASKS: [u128; {len(ALL_MASKS)}] = [")
    for m in ALL_MASKS:
        out.append(f"    {m},")
    out.append("];\n\n")

    # Write STANDARD_PIECES
    out.append(f"pub const STANDARD_PIECES: [[u128; 96]; {num_pieces}] = [")
    for piece_idx, piece_masks in enumerate(STANDARD_PIECES):
        out.append("    [")
        for m in piece_masks:
            out.append(f"        {m},")
        out.append("    ],")
    out.append("];\n")

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "tricked_rs", "src", "constants.rs"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(out))

    print(f"Successfully wrote {num_pieces} pieces to {out_path}")


if __name__ == "__main__":
    generate_rust_constants()
