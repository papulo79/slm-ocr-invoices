"""
Evalúa un prompt específico contra el dataset golden.

Uso:
    python scripts/evaluate_prompt.py
    python scripts/evaluate_prompt.py --prompt results/best_prompt.txt
    python scripts/evaluate_prompt.py --json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gepa import adapter, evaluator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa un prompt contra el dataset golden"
    )
    parser.add_argument(
        "--prompt",
        default=str(ROOT / "prompt.txt"),
        help="Fichero de prompt a evaluar",
    )
    parser.add_argument(
        "--golden",
        default=str(ROOT / "data" / "golden"),
        help="Directorio golden",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Configuración",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en JSON",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    schema_file = ROOT / config["extraction"]["schema_file"]
    schema = json.loads(schema_file.read_text(encoding="utf-8"))

    client = OpenAI(
        base_url=config["lmstudio"]["base_url"],
        api_key=config["lmstudio"].get("api_key", "lm-studio"),
    )
    model = config["lmstudio"]["model"]

    golden_dir = Path(args.golden)
    print(f"Cargando dataset desde {golden_dir} ...")
    dataset = adapter.load_golden_dataset(golden_dir)
    if not dataset:
        print(f"ERROR: No se encontraron muestras en {golden_dir}")
        sys.exit(1)
    print(f"  {len(dataset)} muestras\n")

    ocr_results = []
    for i, (name, images, gold) in enumerate(dataset, 1):
        print(f"[{i:02d}/{len(dataset)}] {name} ...", end=" ", flush=True)
        try:
            pred = adapter.run_ocr(client, model, schema, prompt, images)
            print("OK")
        except Exception as exc:
            print(f"ERROR: {exc}")
            pred = {}
        ocr_results.append((name, pred, gold))

    result = evaluator.evaluate_dataset(ocr_results)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n{'═'*55}")
    print(f"SCORE GLOBAL: {result['mean_total']:.4f}")
    print(f"{'─'*55}")
    print("Scores por métrica:")
    for metric, score in result["mean_scores"].items():
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  {metric:<22} {bar} {score:.3f}")

    print(f"\n{'─'*55}")
    print("Por muestra:")
    for sample in result["per_sample"]:
        issues = sample["issues"]
        suffix = f"  {len(issues)} issues" if issues else ""
        print(f"  {sample['sample']:<35} {sample['total_score']:.4f}{suffix}")
        for issue in issues:
            subtype = f"/{issue['subtype']}" if "subtype" in issue else ""
            print(
                f"       [{issue['type']}{subtype}] {issue['field']}: "
                f"'{issue.get('actual', '')}'"
                f" → '{issue.get('expected', '')}'"
            )


if __name__ == "__main__":
    main()
