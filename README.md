# Sistema de Caracterización de Microestructuras (NMC811)

Proyecto de visión computacional aplicado a ciencia de materiales para caracterizar degradación de cátodos NMC811 en micrografías FIB-SEM de 16 bits, con procesamiento 100% local/offline.

## Resumen ejecutivo

Diseñé e implementé un pipeline modular en Python que:

- segmenta partículas de material activo con MobileSAM,
- extrae métricas geométricas por instancia,
- detecta evidencia de daño interno por contraste intrapartícula,
- genera salidas analíticas (`CSV`) y visuales (heatmaps),
- y expone operación tanto por CLI como por dashboard Streamlit.

## Impacto técnico

- Automatización de caracterización sobre lotes de imágenes `.tif` sin pérdida de información.
- Trazabilidad por partícula con identificadores y validación de calidad.
- Mejora de rigor físico mediante enfoque Human-in-the-Loop: detecté y corregí una limitación clave del descriptor clásico de circularidad.

## Stack y restricciones cumplidas

- Python, PyTorch, MobileSAM, OpenCV, tifffile, pandas, Streamlit.
- Ejecución local (sin APIs externas), optimizado para entorno con GPU NVIDIA RTX 4060.
- Compatibilidad con flujo reproducible en entorno conda.

## Arquitectura del sistema

Pipeline orquestado en `src/pipeline/orchestrator_engine.py`:

`raw -> preprocessed -> segmented -> refined -> analyzed -> exported`

Componentes principales:

- `preprocessing`: CLAHE para compensación de brillo y realce de grietas.
- `segmentation`: inferencia por instancia con MobileSAM.
- `postprocessing`: apertura morfológica para separar partículas adyacentes.
- `metrics`: cálculo geométrico + métricas crack-aware.
- `validation`: filtros por área, aspecto, borde y circularidad.
- `visualization/export`: heatmaps y consolidado `analisis_gold_standard.csv`.

## Desafío clave resuelto: “ceguera topológica”

Inicialmente, la circularidad clásica (`C = 4πA/P²`) evaluaba solo la cáscara externa: partículas con fisura interna podían verse “sanas” (verdes) si su contorno exterior era regular.

Para resolverlo, incorporé métricas internas:

- `dark_area_fraction`
- `crack_severity`
- `circularity_effective` (circularidad penalizada por defecto interno)
- `mean_intensity`, `std_intensity`

Con esto, el heatmap refleja mejor el estado estructural real de la partícula, no solo su geometría externa.

## Resultados y calidad de ingeniería

- Suite de pruebas automatizadas en verde (`pytest`).
- Verificación estática (`ruff`, `mypy`).
- Contratos de error explícitos para operación robusta:
  - `ERR_IO_003`
  - `ERR_VRAM_001`
  - `ERR_MASK_002`
  - `ERR_METRIC_004`

Estado actual: pipeline estable para análisis batch y revisión interactiva en UI.

## Ejecución rápida

### CLI

```powershell
& "C:\Users\adria\miniconda3\Scripts\conda.exe" run -n nmc811-segmentation python -m src.main `
  --input-dir gold_standard `
  --output-dir output `
  --checkpoint-path weights/mobile_sam.pt `
  --device cuda:0
```

### UI

```powershell
streamlit run src/ui/app.py --server.port 8501 --server.headless false
```

## Artefactos de salida

- `output/analisis_gold_standard.csv`
- `output/visualizations/*__heatmap_circularity_overlay.png`
- `output/visualizations/*__heatmap_circularity_meta.json`
- logs de runtime

---

Para una guía operativa paso a paso: `docs/COMO_USAR.md`.

---

# NMC811 Microstructure Characterization System (EN)

Computer vision project for materials science to characterize NMC811 cathode degradation from 16-bit FIB-SEM `.tif` images, fully local/offline.

## Executive summary

I designed and implemented a modular Python pipeline that:

- segments active-material particles with MobileSAM,
- extracts per-instance geometric metrics,
- detects internal-damage evidence from intra-particle intensity contrast,
- generates analytical outputs (`CSV`) and visual overlays (heatmaps),
- and supports both CLI and Streamlit dashboard workflows.

## Core architecture

Orchestrated pipeline in `src/pipeline/orchestrator_engine.py`:

`raw -> preprocessed -> segmented -> refined -> analyzed -> exported`

Main modules:

- `preprocessing`: CLAHE for edge brightness compensation and crack enhancement.
- `segmentation`: instance inference with MobileSAM.
- `postprocessing`: morphological opening to separate touching particles.
- `metrics`: geometric + crack-aware descriptors.
- `validation`: area, aspect ratio, edge, and circularity filters.
- `visualization/export`: heatmaps and consolidated `analisis_gold_standard.csv`.

## Key engineering insight: topological blindness

Classical circularity (`C = 4πA/P²`) can rate a particle as healthy if the outer shell is regular, even when severe internal cracks exist.

To address this, I added internal descriptors:

- `dark_area_fraction`
- `crack_severity`
- `circularity_effective` (internally penalized circularity)
- `mean_intensity`, `std_intensity`

Heatmaps now reflect structural integrity more realistically, not only outer geometry.

## Quality gates

- Automated tests (`pytest`)
- Static checks (`ruff`, `mypy`)
- Explicit runtime error contracts:
  - `ERR_IO_003`
  - `ERR_VRAM_001`
  - `ERR_MASK_002`
  - `ERR_METRIC_004`

Current status: stable for both batch analytics and interactive inspection.
