"""Convenience entry point for the score-based DocMIA baseline.

The full attack runner computes this baseline alongside the proposed method so
all methods are scored from the same artifacts.
"""

from __future__ import annotations

import subprocess


if __name__ == "__main__":
    subprocess.run(["python", "method/docmia_attack.py"], check=True)
