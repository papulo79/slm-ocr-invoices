"""
Optimizer GEPA: orquesta el ciclo de optimización evolutiva del prompt.

Ciclo por iteración:
  1. Evaluar prompt actual contra el dataset golden
  2. Si mejora respecto al mejor conocido → aceptar como nuevo mejor
  3. Llamar al Proposer para generar una variante
  4. Repetir hasta max_iterations o estancamiento
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from gepa import adapter, evaluator, proposer


@dataclass
class GEPAConfig:
    max_iterations: int = 50
    stagnation_limit: int = 10
    results_dir: Path = Path("results")
    golden_dir: Path = Path("data/golden")


@dataclass
class IterationLog:
    iteration: int
    prompt_preview: str
    mean_total: float
    mean_scores: dict
    issues_count: int
    accepted: bool
    elapsed: float


@dataclass
class GEPAResult:
    best_prompt: str
    best_score: float
    iterations: list[IterationLog] = field(default_factory=list)
    total_elapsed: float = 0.0


def _collect_all_issues(eval_result: dict) -> list[dict]:
    issues = []
    for sample in eval_result["per_sample"]:
        issues.extend(sample["issues"])
    return issues


def _save_iteration(
    results_dir: Path,
    log: IterationLog,
    prompt: str,
    eval_result: dict,
) -> None:
    iters_dir = results_dir / "iterations"
    iters_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "iteration": log.iteration,
        "accepted": log.accepted,
        "mean_total": log.mean_total,
        "mean_scores": log.mean_scores,
        "issues_count": log.issues_count,
        "elapsed": log.elapsed,
        "prompt": prompt,
        "per_sample": eval_result["per_sample"],
    }
    out = iters_dir / f"iter_{log.iteration:03d}.json"
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    client: OpenAI,
    model: str,
    schema: dict,
    seed_prompt: str,
    config: GEPAConfig,
    proposer_config: dict,
    verbose: bool = True,
) -> GEPAResult:
    """
    Ejecuta el ciclo GEPA completo.

    Args:
        client:          cliente OpenAI/LM Studio (para OCR)
        model:           nombre del modelo OCR
        schema:          JSON schema para structured output
        seed_prompt:     prompt inicial (semilla)
        config:          configuración GEPA
        proposer_config: sección 'proposer' del config.yaml
        verbose:         imprimir progreso por consola

    Returns:
        GEPAResult con el mejor prompt encontrado y el histórico
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    p_client, p_model = proposer.make_client({"proposer": proposer_config})

    if verbose:
        print(f"Cargando dataset golden desde {config.golden_dir} ...")
    dataset = adapter.load_golden_dataset(config.golden_dir)
    if not dataset:
        raise RuntimeError(f"Dataset vacío en {config.golden_dir}")
    if verbose:
        print(f"  {len(dataset)} muestras cargadas.\n")

    current_prompt = seed_prompt
    best_prompt = seed_prompt
    best_score = -1.0
    stagnation = 0
    result = GEPAResult(best_prompt=seed_prompt, best_score=0.0)
    t_start = time.time()

    for iteration in range(1, config.max_iterations + 1):
        if verbose:
            sep = "─" * 38
            print(f"── Iteración {iteration:03d}/{config.max_iterations} {sep}")

        t_iter = time.time()

        # 1. Evaluar prompt actual
        ocr_results: list[tuple[str, dict, dict]] = []
        failed_names: list[str] = []
        for name, images, gold in dataset:
            try:
                pred = adapter.run_ocr_paged(
                    client, model, schema, current_prompt, images
                )
            except Exception as exc:
                if verbose:
                    print(f"   ERROR OCR en '{name}': {exc}")
                pred = {}
                failed_names.append(name)
            ocr_results.append((name, pred, gold))

        eval_result = evaluator.evaluate_dataset(ocr_results)
        score = eval_result["mean_total"]
        # Solo issues de muestras con OCR exitoso — evita feedback loop
        all_issues = [
            issue
            for sample in eval_result["per_sample"]
            if sample["sample"] not in failed_names
            for issue in sample["issues"]
        ]
        elapsed = time.time() - t_iter

        accepted = score > best_score
        log = IterationLog(
            iteration=iteration,
            prompt_preview=current_prompt[:200],
            mean_total=score,
            mean_scores=eval_result["mean_scores"],
            issues_count=len(all_issues),
            accepted=accepted,
            elapsed=elapsed,
        )
        result.iterations.append(log)

        if verbose:
            status = "ACEPTADO" if accepted else "rechazado"
            print(
                f"   Score: {score:.4f}  "
                f"(mejor: {best_score:.4f})  {status}"
            )
            print(
                f"   Issues: {len(all_issues)}"
                f"  |  Tiempo: {elapsed:.1f}s"
            )

        _save_iteration(config.results_dir, log, current_prompt, eval_result)

        if accepted:
            best_score = score
            best_prompt = current_prompt
            stagnation = 0
            best_path = config.results_dir / "best_prompt.txt"
            best_path.write_text(best_prompt, encoding="utf-8")
        else:
            stagnation += 1
            if verbose:
                print(
                    f"   Estancamiento: "
                    f"{stagnation}/{config.stagnation_limit}"
                )

        if best_score >= 1.0:
            if verbose:
                print("\nScore perfecto alcanzado. Deteniendo.")
            break

        if stagnation >= config.stagnation_limit:
            if verbose:
                print(
                    f"\nEstancamiento tras {stagnation} iteraciones"
                    " sin mejora. Deteniendo."
                )
            break

        # 2. Proponer nueva variante
        if verbose:
            print("   Generando variante con Proposer ...")
        try:
            candidate = proposer.propose(
                p_client, p_model, current_prompt, all_issues
            )
            # Rechazar propuestas >50% más largas que la semilla — evita
            # que el prompt crezca hasta superar el context window del OCR
            max_len = int(len(seed_prompt) * 1.5)
            if len(candidate) > max_len:
                if verbose:
                    print(
                        f"   Propuesta descartada: "
                        f"{len(candidate)} chars > límite {max_len}. "
                        "Usando mejor prompt conocido."
                    )
                current_prompt = best_prompt
            else:
                current_prompt = candidate
        except Exception as exc:
            if verbose:
                print(
                    f"   ERROR en Proposer: {exc}. "
                    "Usando mejor prompt conocido."
                )
            current_prompt = best_prompt

    result.best_prompt = best_prompt
    result.best_score = best_score
    result.total_elapsed = time.time() - t_start

    _save_report(config.results_dir, result, len(dataset))

    if verbose:
        print(f"\n{'═'*55}")
        print(f"GEPA completado en {result.total_elapsed:.1f}s")
        print(f"Mejor score: {best_score:.4f}")
        best_path = config.results_dir / "best_prompt.txt"
        print(f"Prompt guardado en: {best_path}")

    return result


def _save_report(
    results_dir: Path, result: GEPAResult, dataset_size: int
) -> None:
    report = {
        "best_score": result.best_score,
        "total_elapsed": round(result.total_elapsed, 1),
        "dataset_size": dataset_size,
        "iterations_run": len(result.iterations),
        "iterations": [
            {
                "iteration": item.iteration,
                "mean_total": item.mean_total,
                "issues_count": item.issues_count,
                "accepted": item.accepted,
                "elapsed": round(item.elapsed, 1),
            }
            for item in result.iterations
        ],
    }
    (results_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
