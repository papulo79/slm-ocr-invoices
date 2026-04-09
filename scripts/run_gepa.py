"""
Ejecuta la optimización GEPA completa.

Uso:
    python scripts/run_gepa.py
    python scripts/run_gepa.py --iterations 20 --stagnation 5
    python scripts/run_gepa.py --seed results/best_prompt.txt
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gepa.optimizer import GEPAConfig, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimización GEPA del prompt OCR"
    )
    parser.add_argument(
        "--seed",
        default=str(ROOT / "prompt.txt"),
        help="Prompt semilla",
    )
    parser.add_argument(
        "--golden",
        default=str(ROOT / "data" / "golden"),
        help="Dataset golden",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Configuración",
    )
    parser.add_argument(
        "--results",
        default=str(ROOT / "results"),
        help="Directorio de resultados",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Máximo de iteraciones",
    )
    parser.add_argument(
        "--stagnation",
        type=int,
        default=10,
        help="Iteraciones sin mejora para parar",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_prompt = Path(args.seed).read_text(encoding="utf-8")
    schema_file = ROOT / config["extraction"]["schema_file"]
    schema = json.loads(schema_file.read_text(encoding="utf-8"))

    client = OpenAI(
        base_url=config["lmstudio"]["base_url"],
        api_key=config["lmstudio"].get("api_key", "lm-studio"),
    )
    model = config["lmstudio"]["model"]

    gepa_config = GEPAConfig(
        max_iterations=args.iterations,
        stagnation_limit=args.stagnation,
        results_dir=Path(args.results),
        golden_dir=Path(args.golden),
    )

    print("GEPA — Optimización de prompt OCR")
    print(f"  Modelo:      {model}")
    print(
        f"  Iteraciones: máx {gepa_config.max_iterations},"
        f" estancamiento {gepa_config.stagnation_limit}"
    )
    print(f"  Dataset:     {gepa_config.golden_dir}")
    print(f"  Resultados:  {gepa_config.results_dir}")
    print(f"  Semilla:     {args.seed}\n")

    result = run(
        client=client,
        model=model,
        schema=schema,
        seed_prompt=seed_prompt,
        config=gepa_config,
        proposer_config=config.get("proposer", {}),
        verbose=True,
    )

    print(f"\nMejor score final: {result.best_score:.4f}")
    best_path = gepa_config.results_dir / "best_prompt.txt"
    print(f"Prompt óptimo en:  {best_path}")


if __name__ == "__main__":
    main()
