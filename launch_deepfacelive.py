"""
One-click launcher for DeepFaceLive (Windows friendly).
Usage:
  1) Activate venv, then: python launch_deepfacelive.py
  2) Or use a .bat to auto-activate venv and run it
"""

import os
import sys
import multiprocessing


def _project_root() -> str:
    # This script sits in the repo root (same folder as main.py)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    root = _project_root()
    os.chdir(root)

    # Ensure userdata dir is stable (relative paths break easily on Windows)
    userdata_dir = os.path.join(root, "userdata")
    os.makedirs(userdata_dir, exist_ok=True)

    # Equivalent to: python main.py run DeepFaceLive --userdata-dir userdata
    sys.argv = [
        "main.py",
        "run",
        "DeepFaceLive",
        "--userdata-dir",
        userdata_dir,
    ]

    # Import entry ONLY after argv + chdir (important for spawn)
    import main as dfl_main
    dfl_main.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Be explicit: DeepFaceLive uses multiprocessing; spawn is the safe path on Windows.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set by something else; ignore
        pass

    main()
