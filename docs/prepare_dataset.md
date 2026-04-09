# Preparar dataset golden

## Requisitos previos

- LM Studio corriendo en `http://localhost:1234` con un modelo de visión cargado
- Context window del modelo configurado a **≥ 20 000 tokens** en LM Studio
- Facturas PDF en algún directorio accesible (por defecto busca en la raíz del proyecto)

## Ejecución

Desde la raíz del proyecto con el entorno activo:

```bash
source .venv/bin/activate
python scripts/prepare_dataset.py --input /ruta/a/tus/PDFs
```

Esto procesa todos los PDFs, extrae imágenes y obtiene la predicción inicial del modelo.

## Salida

```
data/golden/
  {nombre_factura}/
    page_01.png       ← imagen renderizada (una por página del PDF)
    page_02.png       ← si la factura tiene más páginas
    predicted.json    ← extracción del modelo (no modificar)
    gold.json         ← EDITAR A MANO con los valores correctos
```

## Corrección manual (gold.json)

Abre cada `gold.json` y corrige los errores del modelo:

- **NIFs** — verificar formato: 9 caracteres, empieza con letra si es CIF empresa (B83834747) o con dígito si es NIF persona física (37322251J). Sin prefijos de país (ES, FR, etc.)
- **Descripciones** — verificar que no están truncadas en `line_items[].description`, incluyendo UUIDs completos
- **Seller/Buyer** — en facturas de autónomos comprobar que no están invertidos
- **Campos vacíos** — rellenar cualquier campo que el modelo haya dejado en blanco

> `predicted.json` queda intacto como referencia del estado inicial. Solo edita `gold.json`.

## Opciones útiles

```bash
# Solo extraer imágenes, sin llamar al modelo
python scripts/prepare_dataset.py --skip-model --input /ruta/PDFs

# Reprocesar facturas ya procesadas
python scripts/prepare_dataset.py --force --input /ruta/PDFs

# Directorio de salida alternativo
python scripts/prepare_dataset.py --input /ruta/PDFs --output data/mi-dataset
```

## Verificar el dataset

Tras corregir los `gold.json`, evalúa el prompt actual para confirmar que el dataset es coherente:

```bash
python scripts/evaluate_prompt.py --prompt prompt.txt
```

Un score bajo en la primera evaluación es normal e indica que hay margen de mejora para GEPA.
