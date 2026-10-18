# OCR environment setup (GPU, Windows)

Verified on 2026-10-18: Windows 11, RTX 3050 Laptop 4 GB, NVIDIA driver 610.62
(CUDA 13.3 capable), Python 3.12.5, `uv` 0.11.

The OCR stack mixes **PaddlePaddle-GPU** (detection, orientation, PP-OCR
recognition) and **PyTorch** (VietOCR recognition) in one process. Both ship CUDA
DLLs with the same file names, so versions and import order matter.

## Install

```bash
# 1. core + dev (see README)
uv pip install --python .venv -e ".[dev]"

# 2. PaddlePaddle GPU (CUDA 12.6 build, bundles its CUDA libs as nvidia-* wheels)
uv pip install --python .venv "paddlepaddle-gpu==3.3.1" \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python .venv "paddleocr>=3.7,<4"

# 3. PyTorch (CUDA 12.6) + VietOCR (inference only)
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv --no-deps vietocr==0.3.13   # its pins (pillow==10.2, imgaug) break the env
uv pip install --python .venv einops gdown pyyaml requests

# 4. Make Paddle use the same cuDNN as torch (torch 2.14 cu126 bundles cuDNN 9.10.2)
uv pip install --python .venv "nvidia-cudnn-cu12==9.10.2.21"
```

Check:

```bash
.venv/Scripts/python -c "import torch, paddle; print(torch.cuda.is_available(), paddle.device.is_compiled_with_cuda())"
```

## Pitfalls met (and fixes)

| Symptom | Cause | Fix |
|---|---|---|
| `ConvertPirAttribute2RuntimeAttribute not support` on CPU | oneDNN bug in Paddle 3.3 on Windows | `enable_mkldnn=False` (default in our providers) |
| ~80 s per image | CPU + oneDNN off + `medium` models | GPU build (≈0.4 s), or `v6_tiny` on CPU (≈2–3 s) |
| `WinError 127 … nvidia\cudnn\bin\cudnn_cnn64_9.dll` | Paddle pulled cuDNN 9.5, torch bundles 9.10 | step 4 above |
| `WinError 127 … torch\lib\shm.dll` | Paddle imported before torch | `import ai.ocr._runtime` first (done in every OCR module) |
| PaddleOCR imports torch by itself | PaddleX optional-dependency probing | same as above — keep the import order guard |

Weights are cached in `~/.paddlex/official_models/` (PaddleOCR) and the system
temp folder (VietOCR `vgg_*.pth`).
