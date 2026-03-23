"""
Master script to generate all piece definitions and constants for both Python and Rust.
Run this script whenever the underlying triangular grid or D12 symmetries change.

Usage:
    python scripts/generate_all.py
"""

import os
import subprocess
import sys

def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("=== Step 1: Generating Python Piece Definitions ===")
    py_script = os.path.join(root_dir, "src", "tricked", "closed_pieces.py")
    subprocess.run([sys.executable, py_script], check=True)

    print("\\n=== Step 2: Generating Rust Constants ===")
    rust_script = os.path.join(root_dir, "scripts", "generate_rust_constants.py")
    subprocess.run([sys.executable, rust_script], check=True)

    print("\\n=== All Code Generation Complete ===")

if __name__ == "__main__":
    main()
