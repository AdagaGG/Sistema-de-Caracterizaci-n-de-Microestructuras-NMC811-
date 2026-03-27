# Cómo usar el pipeline (NMC811)

## 1) Activar entorno

Si `conda` está en PATH:

```powershell
conda activate nmc811-segmentation
```

Si no está en PATH (Windows):

```powershell
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python --version
```

## 2) Ejecutar el pipeline completo

```powershell
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python -m src.main `
  --input-dir gold_standard `
  --output-dir output `
  --checkpoint-path weights/mobile_sam.pt `
  --device cuda:0 `
  --log-file output\runtime_execution.jsonl
```

## 3) Qué genera

- CSV por imagen con métricas de partículas.
- Visualizaciones heatmap (circularidad).
- Logs de ejecución (`runtime_execution.jsonl`).

## 4) Métricas incluidas

- `area_px`, `area_um2`
- `perimeter_px`
- `circularity` (`4πA/P²`)
- `equivalent_diameter_px`, `equivalent_diameter_um`
- flags de validación (`validation_status`, `rejection_reason`)

## 5) Ejecutar tests de calidad

```powershell
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python -m ruff check src tests
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python -m mypy src tests
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python -m pytest -q
```

## 6) Errores comunes (PRD)

- `ERR_IO_003`: fallo leyendo TIFF 16-bit.
- `ERR_VRAM_001`: memoria GPU insuficiente.
- `ERR_MASK_002`: 0 partículas válidas tras filtros.
- `ERR_METRIC_004`: perímetro 0 en cálculo geométrico.

## 7) Nota de validación científica

El pipeline ya cumple runtime/VRAM y contratos de error.  
La aceptación final de **tasa de detección >=90%** queda pendiente de tus conteos manuales de referencia para:

- `img_RDBS_0050`
- `img_RDBS_0485`
