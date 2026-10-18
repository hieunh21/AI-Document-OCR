"""Load order guard for GPU runtimes on Windows.

PaddlePaddle-GPU and PyTorch each ship CUDA/cuDNN DLLs with identical file
names. On Windows the first-loaded copy wins, and loading Paddle first makes
``import torch`` fail ("WinError 127 … torch\\lib\\shm.dll"). PaddleOCR also
imports torch on its own when it is installed. Importing torch first (when it
is installed) and pinning ``nvidia-cudnn-cu12`` to torch's cuDNN version
(see docs/setup_gpu.md) makes both work in one process.

Import this module before anything that imports ``paddle``/``paddleocr``.
"""

from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

if importlib.util.find_spec("torch") is not None:
    import torch  # noqa: F401
