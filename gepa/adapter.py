"""
Adapter GEPA: interfaz entre el módulo de optimización y el motor OCR.

Responsabilidades:
  - Cargar imágenes del dataset golden (page_01.png, page_02.png, …)
  - Llamar al modelo con un prompt arbitrario
  - Devolver el JSON predicho
"""

import base64
import io
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image


def load_golden_images(sample_dir: Path) -> list[Image.Image]:
    """Carga todas las páginas PNG de un directorio de muestra, ordenadas."""
    pages = sorted(sample_dir.glob("page_*.png"))
    if not pages:
        raise FileNotFoundError(f"No se encontraron imágenes en {sample_dir}")
    return [Image.open(p).convert("RGB") for p in pages]


def load_golden_dataset(golden_dir: Path) -> list[tuple[str, list[Image.Image], dict]]:
    """
    Carga todo el dataset golden.

    Returns:
        Lista de (nombre, imágenes, gold_dict)
    """
    dataset = []
    for sample_dir in sorted(golden_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        gold_file = sample_dir / "gold.json"
        if not gold_file.exists():
            continue
        images = load_golden_images(sample_dir)
        gold = json.loads(gold_file.read_text(encoding="utf-8"))
        dataset.append((sample_dir.name, images, gold))
    return dataset


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_ocr(
    client: OpenAI,
    model: str,
    schema: dict,
    prompt: str,
    images: list[Image.Image],
) -> dict:
    """
    Llama al modelo OCR con el prompt dado y devuelve el JSON extraído.

    Lanza json.JSONDecodeError o excepciones de OpenAI si algo falla.
    """
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_image_to_base64(img)}"},
        }
        for img in images
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": image_content},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_extraction",
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content)
