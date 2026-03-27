# Notable findings from plan execution

- Explicit Windows fallback to `C:\Users\adria\miniconda3\Scripts\conda.exe run ...` when `conda` is not available on `PATH`.
- Common failure pattern fixed: exported-stage `int(None)`/`NoneType` coercion in heatmap-manifest path.
- Detection-rate acceptance now supports a provisional baseline pending manual counts (`pending_manual_validation`).
