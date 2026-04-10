"""
Batch evaluation script: procesa todos los PDFs de un directorio,
llama al modelo y guarda los resultados en JSON individuales
y un resumen consolidado.

Uso:
    python batch_eval.py --input /ruta/facturas --output ./resultados
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gepa import adapter  # noqa: E402


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_extraction_config(config: dict) -> tuple[str, dict]:
    extraction = config.get("extraction", {})
    prompt = Path(extraction.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")
    schema = json.loads(Path(extraction.get("schema_file", "schema.json")).read_text(encoding="utf-8"))
    return prompt, schema


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200, first_only: bool = False) -> list[Image.Image]:
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    return images[:1] if first_only else images



def evaluate_result(filename: str, result: dict) -> dict:
    """Evalúa la calidad de la extracción y devuelve un informe de campos."""
    issues = []

    seller = result.get("seller", {})
    if not seller.get("name"):
        issues.append("seller.name vacío")
    if not seller.get("vat_id"):
        issues.append("seller.vat_id vacío")
    if not seller.get("address"):
        issues.append("seller.address ausente")

    buyer = result.get("buyer")
    if not buyer:
        issues.append("buyer ausente")
    elif not buyer.get("name"):
        issues.append("buyer.name vacío")

    if not result.get("invoice_number"):
        issues.append("invoice_number vacío")
    if not result.get("invoice_date"):
        issues.append("invoice_date vacío")

    line_items = result.get("line_items", [])
    if not line_items:
        issues.append("line_items vacío")
    else:
        for i, item in enumerate(line_items):
            if not item.get("description"):
                issues.append(f"line_items[{i}].description vacío")
            if item.get("price_includes_vat") is None:
                issues.append(f"line_items[{i}].price_includes_vat ausente")

    return {
        "file": filename,
        "seller_name": seller.get("name", ""),
        "seller_vat": seller.get("vat_id", ""),
        "buyer_name": (buyer or {}).get("name", ""),
        "invoice_number": result.get("invoice_number", ""),
        "invoice_date": result.get("invoice_date", ""),
        "amount": result.get("amount"),
        "vat_amount": result.get("vat_amount"),
        "line_items_count": len(line_items),
        "issues": issues,
        "quality": "OK" if not issues else f"WARN ({len(issues)} campos)",
    }


def process_batch(input_dir: str, output_dir: str, config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    prompt, schema = load_extraction_config(config)
    dpi = config["processing"].get("dpi", 200)
    first_only = config["processing"].get("first_page_only", False)

    model = config["lmstudio"]["model"]
    client = OpenAI(
        base_url=config["lmstudio"]["base_url"],
        api_key=config["lmstudio"].get("api_key", "lm-studio"),
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en {input_dir}")
        sys.exit(1)

    print(f"Procesando {len(pdfs)} facturas → {output_dir}\n")

    summary = []
    errors = []

    for i, pdf_file in enumerate(pdfs, 1):
        print(f"[{i:02d}/{len(pdfs)}] {pdf_file.name} ...", end=" ", flush=True)
        start = time.time()

        try:
            pdf_bytes = pdf_file.read_bytes()
            images = pdf_to_images(pdf_bytes, dpi=dpi, first_only=first_only)
            result = adapter.run_ocr_paged(client, model, schema, prompt, images)

            # Guardar JSON individual
            out_file = output_path / f"{pdf_file.stem}.json"
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            evaluation = evaluate_result(pdf_file.name, result)
            summary.append(evaluation)

            elapsed = time.time() - start
            print(f"{evaluation['quality']} ({elapsed:.1f}s)")
            if evaluation["issues"]:
                for issue in evaluation["issues"]:
                    print(f"         ⚠ {issue}")

        except json.JSONDecodeError as e:
            elapsed = time.time() - start
            print(f"ERROR JSON ({elapsed:.1f}s): {e}")
            errors.append({"file": pdf_file.name, "error": f"JSON inválido: {e}"})

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            errors.append({"file": pdf_file.name, "error": str(e)})

    # Guardar resumen consolidado
    report = {
        "total": len(pdfs),
        "ok": sum(1 for s in summary if s["quality"] == "OK"),
        "warnings": sum(1 for s in summary if s["quality"].startswith("WARN")),
        "errors": len(errors),
        "results": summary,
        "errors_detail": errors,
    }

    report_file = output_path / "_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'─'*50}")
    print(f"Total: {report['total']}  OK: {report['ok']}  Warnings: {report['warnings']}  Errores: {report['errors']}")
    print(f"Informe guardado en: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR de facturas con LM Studio")
    parser.add_argument("--input", required=True, help="Directorio con PDFs")
    parser.add_argument("--output", default="./resultados", help="Directorio de salida para JSONs")
    parser.add_argument("--config", default="config.yaml", help="Fichero de configuración")
    args = parser.parse_args()

    process_batch(args.input, args.output, args.config)
