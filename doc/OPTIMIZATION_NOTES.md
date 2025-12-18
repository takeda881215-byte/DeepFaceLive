# Optimization and dependency notes

This project ships prebuilt Windows binaries, but the source tree can be customized or run directly. The notes below highlight dependency expectations and code hotspots that can benefit from tuning.

## Dependency alignment

- The new `requirements.txt` captures the core runtime used across the ONNX/Torch models and PyQt6 GUI. It mirrors the versions pinned in `build/windows/WindowsBuilder.py` and should be installed before running `main.py`.
- ONNX Runtime comes in multiple variants. Pick the CPU (`onnxruntime`), CUDA (`onnxruntime-gpu`), or DirectML (`onnxruntime-directml`) wheel that matches your hardware, and ensure the chosen variant is uncommented in `requirements.txt` before installation.
- PyTorch/TorchVision wheels should match your CUDA toolkit. If you need GPU acceleration, install from the official index (e.g., `https://download.pytorch.org/whl/cu117`) that matches the versions in `requirements.txt`.

## Performance opportunities

- **Configurable shared heap**: `apps/DeepFaceLive/DeepFaceLiveApp.py` constructs a `BackendWeakHeap` with a fixed 2048 MB budget. Exposing this value through CLI args or settings would let users trade memory for throughput on lower-end or high-resolution setups.
- **Reuse model sessions**: `FaceSwapInsight` and `FaceSwapDFM` instantiate ONNX models during each device/model change (`apps/DeepFaceLive/backend/FaceSwapInsight.py`, `apps/DeepFaceLive/backend/FaceSwapDFM.py`). Keeping a cache of sessions keyed by device/model would reduce re-initialization hitches when users toggle options.
- **Prefer explicit device selection**: ONNX models currently rely on `xlib.onnxruntime.get_available_devices_info()` to populate choices (e.g., `modelhub/onnx/YoloV5Face/YoloV5Face.py`, `modelhub/DFLive/DFMModel.py`). Recording the last successful device in the user settings and skipping CPU when a GPU is available can avoid silent fallbacks.
- **Trainer efficiency**: The FaceAligner trainer (`apps/trainers/FaceAligner/FaceAlignerTrainerApp.py`) always builds models on the selected device. Adding mixed-precision or gradient-accumulation toggles would help users with limited VRAM keep batch sizes reasonable.
