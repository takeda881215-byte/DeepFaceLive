"""
Convenience launcher for one-click startup of DeepFaceLive.

Usage:
    python launch_deepfacelive.py
"""
from pathlib import Path

import main as dfl_main


def main():
    # Reuse the zero-argument shortcut defined in main.main
    dfl_main.main([])


if __name__ == "__main__":
    main()
