# slm-ocr-invoices

Sistema de OCR para facturas en PDF usando un modelo local de visión (LM Studio), con pipeline de optimización automática de prompts mediante GEPA.

## ¿Qué hace?

1. Convierte facturas PDF a imágenes (página a página)
2. Envía las imágenes al modelo local (LM Studio) con un prompt configurable
3. Extrae un JSON estructurado según el esquema Factur-X definido en `schema.json`
4. Opcionalmente, optimiza el prompt automáticamente usando el sistema GEPA

Incluye una interfaz Streamlit (`app.py`) para validación visual y un sistema de evaluación batch (`batch_eval.py`).

## Stack

- **Modelo OCR**: LM Studio (API compatible con OpenAI) — Ministral 3B o superior con soporte de visión
- **Proposer GEPA**: LLM externo vía API OpenAI-compatible (MiniMax M2.7, GPT-4o, etc.)
- **PDF → imagen**: `pdf2image` (requiere `poppler`)
- **Config**: `config.yaml` — modelo, endpoint, DPI, proposer

## Requisitos del sistema

- Python 3.10+
- [poppler](https://poppler.freedesktop.org/) (`apt install poppler-utils` en Linux)
- LM Studio corriendo localmente con un modelo de visión cargado
- Context window del modelo configurado a **≥ 20 000 tokens** en LM Studio
- API key de un LLM externo para el Proposer GEPA (opcional si no se usa optimización)

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copia `config.yaml.example` a `config.yaml` y rellena tus valores:

```bash
cp config.yaml.example config.yaml
```

## Configuración (`config.yaml`)

```yaml
lmstudio:
  base_url: "http://localhost:1234/v1"
  api_key: "lm-studio"
  model: "mistralai/ministral-3-3b"   # nombre exacto en LM Studio

proposer:
  base_url: "https://api.minimax.io/v1"
  api_key: "TU_API_KEY"
  model: "MiniMax-M2.7"

processing:
  dpi: 300
  first_page_only: false
```

> `config.yaml` está en `.gitignore` para no exponer credenciales.

## Uso — Interfaz visual

```bash
source .venv/bin/activate
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador.

## Uso — Procesamiento batch

```bash
python batch_eval.py --input /ruta/a/tus/PDFs --output ./resultados
```

Genera un JSON por factura en `--output` y un `_report.json` con el resumen global.

## Uso — GEPA (optimización de prompt)

### 1. Preparar el dataset golden

```bash
python scripts/prepare_dataset.py --input /ruta/a/tus/PDFs
```

Genera `data/golden/{nombre}/page_01.png` + `predicted.json` por cada factura.  
Edita manualmente cada `gold.json` corrigiendo los errores del modelo.  
Ver [docs/prepare_dataset.md](docs/prepare_dataset.md) para más detalles.

### 2. Evaluar un prompt

```bash
python scripts/evaluate_prompt.py --prompt prompt.txt
```

Muestra el score global y por campo comparando contra los `gold.json`.

### 3. Ejecutar optimización

```bash
python scripts/run_gepa.py \
  --seed prompt.txt \
  --results results/mi-modelo \
  --iterations 20
```

El mejor prompt se guarda en `results/mi-modelo/best_prompt.txt`.

### 4. Validar conexión con el Proposer

```bash
python scripts/test_proposer.py
```

## Resultados obtenidos

| Modelo OCR | Prompt | Score global |
|---|---|---|
| Ministral 3B | Original (v1) | 0.8796 |
| Ministral 3B | Optimizado con GEPA | **0.9441** |

El prompt optimizado se encuentra en `results/ministral-3-3b/best_prompt.txt`  
y en `prompt.txt` (versión activa).

Los mejores prompts por modelo se guardan en `results/{modelo}/best_prompt.txt`.

## Estructura del proyecto

```
├── app.py                     # Interfaz Streamlit
├── batch_eval.py              # Procesamiento batch
├── config.yaml                # Configuración (no versionado)
├── config.yaml.example        # Plantilla de configuración
├── prompt.txt                 # Prompt activo
├── schema.json                # Esquema JSON de salida (Factur-X)
│
├── gepa/                      # Módulo de optimización GEPA
│   ├── adapter.py             # Interfaz OCR ↔ GEPA
│   ├── evaluator.py           # Métricas campo a campo
│   ├── optimizer.py           # Ciclo evolutivo
│   └── proposer.py            # Generador de variantes de prompt
│
├── data/golden/               # Dataset de evaluación
│   └── {nombre_factura}/
│       ├── page_01.png        # Imagen(es) de la factura
│       ├── predicted.json     # Predicción original del modelo
│       └── gold.json          # Ground truth (corregido manualmente)
│
├── results/                   # Salidas de optimización
│   ├── {modelo}/
│   │   ├── best_prompt.txt    # Mejor prompt para ese modelo
│   │   ├── iterations/        # Log de cada iteración
│   │   └── report.json        # Resumen de la optimización
│   └── ...
│
├── scripts/
│   ├── prepare_dataset.py     # Genera el dataset golden
│   ├── evaluate_prompt.py     # Evalúa un prompt concreto
│   ├── run_gepa.py            # Ejecuta la optimización GEPA
│   └── test_proposer.py       # Valida conexión con el Proposer
│
└── docs/
    ├── prepare_dataset.md     # Instrucciones para el dataset golden
    ├── GEPA_DESIGN.md         # Especificación del sistema GEPA
    └── old/                   # Prompts históricos (v1, v2)
```
