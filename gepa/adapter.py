"""
Adapter GEPA: interfaz entre el módulo de optimización y el motor OCR.

Responsabilidades:
  - Cargar imágenes del dataset golden (page_01.png, page_02.png, …)
  - Llamar al modelo con un prompt arbitrario
  - Devolver el JSON predicho
  - Procesar facturas multipágina con contexto acumulado (run_ocr_paged)
"""

import base64
import io
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).parent.parent


def load_golden_images(sample_dir: Path) -> list[Image.Image]:
    """Carga todas las páginas PNG de un directorio de muestra, ordenadas."""
    pages = sorted(sample_dir.glob("page_*.png"))
    if not pages:
        raise FileNotFoundError(f"No se encontraron imágenes en {sample_dir}")
    return [Image.open(p).convert("RGB") for p in pages]


def load_golden_dataset(
    golden_dir: Path,
) -> list[tuple[str, list[Image.Image], dict]]:
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
            "image_url": {
                "url": f"data:image/png;base64,{_image_to_base64(img)}"
            },
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


def run_ocr_paged(
    client: OpenAI,
    model: str,
    schema: dict,
    prompt: str,
    images: list[Image.Image],
    continuation_prompt: str | None = None,
    continuation_schema: dict | None = None,
) -> dict:
    """
    Procesa una factura multipágina enviando cada página por separado.

    Página 1: extracción completa (header + line_items) con el prompt normal.
    Páginas 2+: extracción solo de line_items con contexto de páginas
    anteriores.

    Si solo hay una página, delega directamente en run_ocr().
    """
    if len(images) == 1:
        return run_ocr(client, model, schema, prompt, images)

    if continuation_prompt is None:
        cont_prompt_path = ROOT / "prompt_continuation.txt"
        continuation_prompt = cont_prompt_path.read_text(encoding="utf-8")

    if continuation_schema is None:
        cont_schema_path = ROOT / "schema_continuation.json"
        continuation_schema = json.loads(
            cont_schema_path.read_text(encoding="utf-8")
        )

    # Página 1: extracción completa
    result = run_ocr(client, model, schema, prompt, [images[0]])
    line_items = result.get("line_items") or []

    # Páginas 2+: solo line_items con contexto
    total_pages = len(images)
    for page_idx, page_image in enumerate(images[1:], start=2):
        last_description = line_items[-1]["description"] if line_items else ""
        filled_prompt = continuation_prompt.format(
            page_number=page_idx,
            total_pages=total_pages,
            invoice_number=result.get("invoice_number", ""),
            seller_name=result.get("seller", {}).get("name", ""),
            seller_vat=result.get("seller", {}).get("vat_id", ""),
            extracted_count=len(line_items),
            last_description=last_description,
        )
        page_result = run_ocr(
            client, model, continuation_schema, filled_prompt, [page_image]
        )
        line_items.extend(page_result.get("line_items") or [])

    result["line_items"] = line_items
    return result
