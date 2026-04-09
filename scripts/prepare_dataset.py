"""
Prepara el dataset golden para GEPA.

Para cada PDF:
  - Convierte a imágenes PNG (una por página)
  - Llama al modelo y guarda la predicción inicial
  - Crea un gold.json vacío listo para corregir a mano

Estructura de salida:
  data/golden/
    {factura_stem}/
      page_01.png
      page_02.png   (si hay más páginas)
      predicted.json
      gold.json     (copia de predicted.json → corrígelo a mano)

Uso:
    python scripts/prepare_dataset.py --input /ruta/facturas
    python scripts/prepare_dataset.py --input /ruta/facturas --skip-model
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_extraction_config(config: dict) -> tuple[str, dict]:
    extraction = config.get("extraction", {})
    prompt = (ROOT / extraction.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")
    schema = json.loads((ROOT / extraction.get("schema_file", "schema.json")).read_text(encoding="utf-8"))
    return prompt, schema


def pdf_to_images(pdf_bytes: bytes, dpi: int) -> list[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_model(client: OpenAI, config: dict, prompt: str, schema: dict, images: list[Image.Image]) -> dict:
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"},
        }
        for img in images
    ]
    response = client.chat.completions.create(
        model=config["lmstudio"]["model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": image_content},
        ],
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


def process_pdf(
    pdf_file: Path,
    output_dir: Path,
    client: OpenAI | None,
    config: dict,
    prompt: str,
    schema: dict,
    dpi: int,
    skip_model: bool,
    force: bool,
) -> str:
    """Procesa un PDF. Devuelve 'ok', 'skip' o 'error'."""
    sample_dir = output_dir / pdf_file.stem
    predicted_file = sample_dir / "predicted.json"
    gold_file = sample_dir / "gold.json"

    if predicted_file.exists() and not force:
        return "skip"

    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convertir a imágenes y guardar
    pdf_bytes = pdf_file.read_bytes()
    images = pdf_to_images(pdf_bytes, dpi=dpi)

    for i, img in enumerate(images, 1):
        img_path = sample_dir / f"page_{i:02d}.png"
        img.save(img_path, format="PNG")

    if skip_model:
        return "ok"

    # 2. Llamar al modelo
    result = call_model(client, config, prompt, schema, images)

    # 3. Guardar predicted.json
    predicted_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Copiar a gold.json si no existe (para editar a mano)
    if not gold_file.exists():
        gold_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara dataset golden para GEPA")
    parser.add_argument("--input", required=True, help="Directorio con PDFs")
    parser.add_argument("--output", default=str(ROOT / "data" / "golden"), help="Directorio de salida")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Fichero de configuración")
    parser.add_argument("--skip-model", action="store_true", help="Solo extraer imágenes, no llamar al modelo")
    parser.add_argument("--force", action="store_true", help="Reprocesar aunque ya exista predicted.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    prompt, schema = load_extraction_config(config)
    dpi = config["processing"].get("dpi", 200)
    output_path = Path(args.output)

    client = None
    if not args.skip_model:
        client = OpenAI(
            base_url=config["lmstudio"]["base_url"],
            api_key=config["lmstudio"].get("api_key", "lm-studio"),
        )

    pdfs = sorted(Path(args.input).glob("*.pdf")) + sorted(Path(args.input).glob("*.PDF"))
    if not pdfs:
        print(f"No se encontraron PDFs en {args.input}")
        sys.exit(1)

    print(f"Procesando {len(pdfs)} facturas → {output_path}\n")

    ok = skipped = errors = 0
    for i, pdf_file in enumerate(pdfs, 1):
        label = f"[{i:02d}/{len(pdfs)}] {pdf_file.name}"
        print(f"{label} ...", end=" ", flush=True)
        start = time.time()
        try:
            status = process_pdf(
                pdf_file, output_path, client, config, prompt, schema, dpi,
                args.skip_model, args.force,
            )
            elapsed = time.time() - start
            if status == "skip":
                print(f"SKIP (ya existe)")
                skipped += 1
            else:
                print(f"OK ({elapsed:.1f}s)")
                ok += 1
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            errors += 1

    print(f"\n{'─'*50}")
    print(f"OK: {ok}  Saltados: {skipped}  Errores: {errors}")
    print(f"Dataset en: {output_path}")
    if not args.skip_model:
        print("\nPróximo paso: revisa y corrige los gold.json en cada subdirectorio.")


if __name__ == "__main__":
    main()
