# slm-ocr-invoices

MVP para validar el procesamiento automático de facturas en PDF usando un modelo local con OCR via LM Studio.

## ¿Qué hace?

1. Selecciona una carpeta con PDFs desde el navegador (Streamlit)
2. Convierte cada PDF a imágenes (página a página)
3. Envía las imágenes al modelo local (LM Studio / Mistral con visión) con un prompt configurable
4. Recupera una respuesta estructurada definida en el fichero de configuración
5. Muestra el PDF original, la imagen enviada y el JSON resultado — para validar la extracción

## Stack

- **Frontend/Backend**: Python + Streamlit
- **Modelo**: LM Studio (API compatible con OpenAI) — Mistral con OCR/visión
- **PDF → imagen**: `pdf2image` (requiere `poppler`)
- **Config**: `config.yaml` — prompt, esquema de respuesta, endpoint y modelo

## Requisitos del sistema

- Python 3.10+
- [poppler](https://poppler.freedesktop.org/) instalado (`apt install poppler-utils` en Linux)
- LM Studio corriendo localmente con un modelo de visión cargado

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
source .venv/bin/activate
streamlit run app.py
```

Luego abre `http://localhost:8501` en el navegador.
