# CUDA + Conda Troubleshooting (setup-conda-environment)

## Target stack (offline/local)
- Environment manager: Conda
- Python: 3.10
- PyTorch stack: torch 2.3.x + torchvision 0.18.x + torchaudio 2.3.x
- CUDA runtime package: pytorch-cuda=12.1 (Conda package, no full CUDA toolkit install required)

## Create / update commands
```powershell
conda env remove -n nmc811-segmentation -y
conda env create -f environment.yml
conda activate nmc811-segmentation
```

## Verification commands
```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import tifffile, cv2, skimage, pandas; print('imports_ok')"
```

## If `torch.cuda.is_available()` is False
1. Validate GPU driver is visible:
   - `nvidia-smi`
2. Ensure environment uses conda-installed CUDA runtime package:
   - `conda list | findstr /I "pytorch pytorch-cuda cudnn"`
3. Reinstall pinned GPU stack in the same env:
   - `conda install pytorch=2.3.* torchvision=0.18.* torchaudio=2.3.* pytorch-cuda=12.1 -c pytorch -c nvidia -y`
4. Confirm Python executable is from activated env:
   - `where python`

## Hardware note
The PRD assumes RTX 4060 8GB. Current host reports a lower-tier GPU and VRAM than target, so throughput and model-size headroom will differ. Keep per-image sequential processing and explicit VRAM cleanup.
