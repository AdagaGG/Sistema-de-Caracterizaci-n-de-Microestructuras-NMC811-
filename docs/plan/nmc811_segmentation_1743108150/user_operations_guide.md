# Guía Operativa de Usuario — Sistema NMC811

Este documento describe cómo preparar el entorno, ejecutar el pipeline, interpretar salidas y recuperar errores operativos en el sistema de caracterización de microestructuras NMC811.

## 1) Requisitos previos

- Sistema operativo: Windows (flujo validado en entorno local con Conda)
- GPU: flujo diseñado para CUDA (`--device cuda:0`), con criterio PRD de VRAM `< 7.5 GB`
- Conda instalado
- Repositorio clonado localmente

### Estructura base esperada

- `gold_standard/` (imágenes `.tif/.tiff`)
- `weights/mobile_sam.pt` (checkpoint)
- `environment.yml`
- `src/main.py`

## 2) Setup del entorno

Desde la raíz del repositorio:

```powershell
conda env remove -n nmc811-segmentation -y
conda env create -f environment.yml
conda activate nmc811-segmentation
```

Validación rápida de dependencias y CUDA:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import tifffile, cv2, skimage, pandas; print('imports_ok')"
```

## 3) Ejecución del pipeline (end-to-end)

Comando de referencia validado:

```powershell
C:\Users\adria\miniconda3\Scripts\conda.exe run -n nmc811-segmentation python -m src.main --input-dir gold_standard --output-dir docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\full_pipeline --checkpoint-path weights/mobile_sam.pt --device cuda:0 --no-progress --log-file docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\full_pipeline\runtime_execution.jsonl
```

Parámetros operativos principales (`src/main.py`):

- `--input-dir` (requerido): carpeta con `.tif/.tiff`
- `--output-dir` (requerido): carpeta de resultados
- `--checkpoint-path` (requerido): checkpoint MobileSAM/SAM2
- `--device` (opcional, default `cpu`): usar `cuda:0` para GPU
- `--log-file` (opcional): bitácora estructurada JSONL
- `--no-progress` (opcional): desactiva barra de progreso
- `--start-at`, `--max-images` (opcionales): reanudación/ventana parcial
- `--continue-on-error` / `--fail-fast` (opcionales): política de tolerancia a fallos

## 4) Qué salidas genera el sistema

Dentro de `--output-dir`:

- `runtime_execution.jsonl`  
  Eventos de ejecución por lote e imagen (`BATCH_STARTED`, `IMAGE_COMPLETED`, `BATCH_COMPLETED`).
- `perf_metrics.json`  
  Métricas de rendimiento global (tiempo total, pico de VRAM, comando ejecutado).
- `stdout.json` / `stderr.txt`  
  Resumen de ejecución y advertencias/runtime.
- `img_RDBS_XXXX/analisis_gold_standard.csv`  
  Métricas por partícula para cada imagen procesada.
- `img_RDBS_XXXX/visualizations/*`  
  Overlay/máscara/leyenda/meta de heatmap de circularidad.

### Evidencia final de validación (referencia)

Raíz de evidencia:

- `docs/plan/nmc811_segmentation_1743108150/evidence/execute-validation-testing_final`

Artefactos clave:

- `full_pipeline/perf_metrics.json`
- `full_pipeline/runtime_execution.jsonl`
- `full_pipeline/stdout.json`
- `full_pipeline/stderr.txt`
- `full_pipeline/vram_samples.csv`
- `error_contract_tests/exit_code.txt`
- `error_contract_tests/pytest_output.txt`
- `error_contract_tests/pytest_junit.xml`
- `acceptance_matrix_status.json`
- `detection_rate_reference_provisional.json`

## 5) Interpretación operacional de resultados

- **Éxito de lote**: `pipeline_exit_code = 0` y `images_failed = 0`.
- **Cumplimiento de desempeño** (evidencia final):
  - Tiempo: `352.652 s` (~`5.88 min`) para 6 imágenes.
  - Pico VRAM: `3.799 GB` (dentro del umbral PRD `< 7.5 GB`).
- **Contratos de error**:
  - Suite `pytest` reporta `4 passed` y `exit_code = 0`.

### Nota de aceptación (detección de partículas)

La aceptación de **detection-rate >90%** está en estado **provisional** hasta contar con conteos manuales de referencia:

- `acceptance_matrix_status.json`: `pending_manual_validation`
- `detection_rate_reference_provisional.json`: conteos automáticos disponibles, conteos manuales `null`

No marcar cierre PRD estricto hasta completar esos conteos manuales y recalcular la tasa de detección.

## 6) Troubleshooting por código de error

Los mensajes canónicos deben coincidir con PRD y catálogo de errores.

### ERR_VRAM_001

**Mensaje canónico**:  
`GPU memory exceeded during segmentation. Reduce image resolution or tile size.`

**Causa típica**:
- OOM en etapa de segmentación (imagen/resolución/carga de GPU).

**Recuperación**:
1. Confirmar uso de GPU y memoria disponible:
   ```powershell
   nvidia-smi
   ```
2. Ejecutar en modo secuencial (ya es comportamiento por defecto del pipeline).
3. Reducir presión de memoria con ventana de ejecución:
   - Usar `--max-images` para lotes parciales y validar estabilidad.
4. Si persiste, ejecutar temporalmente en CPU para aislar problema:
   - `--device cpu`
5. Revisar `vram_samples.csv` y `runtime_execution.jsonl` para identificar el punto exacto de crecimiento.

### ERR_MASK_002

**Mensaje canónico**:  
`No valid particles detected after filtering. Check thresholds or image quality.`

**Causa típica**:
- Filtro de validación deja cero partículas válidas (thresholds estrictos o mala calidad de imagen).

**Recuperación**:
1. Verificar integridad/calidad del `.tif` de entrada.
2. Revisar parámetros de preprocesado/segmentación usados en esa corrida.
3. Confirmar que no se está iniciando en imagen equivocada con `--start-at`.
4. Re-ejecutar imagen aislada con `--max-images 1` para depuración controlada.

### ERR_IO_003

**Mensaje canónico**:  
`Failed to read 16-bit TIFF file. Verify file integrity and tifffile library version.`

**Causa típica**:
- TIFF corrupto/no legible o incompatibilidad de librería en entorno.

**Recuperación**:
1. Verificar archivo de entrada (abrir en visor alternativo o regenerar copia).
2. Confirmar versión de `tifffile` en el entorno:
   ```powershell
   python -c "import tifffile; print(tifffile.__version__)"
   ```
3. Revalidar entorno con `environment.yml` y recrear env si hubo deriva.

### ERR_METRIC_004

**Mensaje canónico**:  
`Circularity calculation failed: perimeter is zero. Check contour extraction.`

**Causa típica**:
- Contorno degenerado (perímetro cero) en etapa de métricas geométricas.

**Recuperación**:
1. Revisar máscaras refinadas de la imagen afectada (`visualizations/*mask*`).
2. Validar que segmentación/morfología no generen regiones degeneradas.
3. Re-ejecutar imagen individual para reproducir y aislar.

## 7) Comandos de verificación final recomendados

### A) Pipeline completo

```powershell
C:\Users\adria\miniconda3\Scripts\conda.exe run -n nmc811-segmentation python -m src.main --input-dir gold_standard --output-dir docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\full_pipeline --checkpoint-path weights/mobile_sam.pt --device cuda:0 --no-progress --log-file docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\full_pipeline\runtime_execution.jsonl
```

### B) Contratos de error

```powershell
C:\Users\adria\miniconda3\Scripts\conda.exe run -n nmc811-segmentation pytest tests/test_error_contracts_cross_module.py -q --junitxml docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\error_contract_tests\pytest_junit.xml
```

### C) Inspección rápida de evidencia

```powershell
Get-Content docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\full_pipeline\perf_metrics.json
Get-Content docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\acceptance_matrix_status.json
Get-Content docs\plan\nmc811_segmentation_1743108150\evidence\execute-validation-testing_final\detection_rate_reference_provisional.json
```

## 8) Checklist operativo para usuario final

- [ ] Entorno `nmc811-segmentation` creado y activado
- [ ] CUDA visible (`torch.cuda.is_available() == True`) para corrida GPU
- [ ] Checkpoint en `weights/mobile_sam.pt`
- [ ] Pipeline ejecutado sin fallos (`pipeline_exit_code = 0`)
- [ ] `images_succeeded = 6` / `images_failed = 0`
- [ ] `peak_vram_gb < 7.5`
- [ ] Error contracts: `4 passed` y `exit_code = 0`
- [ ] Criterio detection-rate marcado como provisional hasta conteos manuales
