"""Compatibility entry point for the current evidence-based manuscript.

The original generator is retained in review/archive/build_thesis_v1.py.
"""
if __package__:
    from .build_revised_thesis import build
else:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from build_revised_thesis import build


if __name__ == "__main__":
    build()
