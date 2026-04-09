"""
Proposer GEPA: genera variantes mejoradas del prompt basándose
en los errores detectados.

Usa un LLM externo (MiniMax o similar, vía API OpenAI-compatible)
con un meta-prompt que describe los errores encontrados y pide
una modificación específica al prompt de extracción.
"""

import re
from collections import Counter

from openai import OpenAI

# Meta-prompt para el proposer
_META_PROMPT = """\
Eres un experto en prompt engineering para modelos de visión (OCR).
Tu tarea es mejorar un prompt de extracción de datos de facturas
españolas.

Se ha evaluado el prompt actual contra un dataset de facturas reales
y se han detectado los siguientes errores:

{error_summary}

PROMPT ACTUAL:
---
{current_prompt}
---

INSTRUCCIONES:
1. Analiza los errores listados arriba.
2. Identifica qué instrucciones del prompt podrían causar esos errores
   o qué instrucciones faltan.
3. Genera una versión MEJORADA del prompt que corrija específicamente
   esos errores.
4. Mantén todas las instrucciones que funcionan bien. Solo modifica
   o añade lo necesario.
5. El prompt resultante debe ser claro, conciso y directo.

RESTRICCIONES CRÍTICAS:
- El prompt resultante NO debe ser más largo que el prompt actual.
  Preferiblemente igual de largo o más corto.
- Responde ÚNICAMENTE con el nuevo prompt completo, sin explicaciones
  ni comentarios.
- No añadas texto antes ni después del prompt.
- No uses markdown ni bloques de código.
"""


def _build_error_summary(issues: list[dict]) -> str:
    """Resumen legible de los errores del dataset para el meta-prompt."""
    if not issues:
        return "No se detectaron errores significativos."

    type_counts: Counter = Counter()
    examples: dict[str, list[str]] = {}

    for issue in issues:
        issue_type = issue.get("subtype") or issue.get("type", "unknown")
        type_counts[issue_type] += 1
        if issue_type not in examples:
            examples[issue_type] = []
        if len(examples[issue_type]) < 3:
            field = issue.get("field", "")
            expected = issue.get("expected", "")
            actual = issue.get("actual", "")
            examples[issue_type].append(
                f'  • Campo "{field}": '
                f'esperado="{expected}" → obtenido="{actual}"'
            )

    lines = []
    for issue_type, count in type_counts.most_common():
        lines.append(f"\n[{issue_type.upper()}] — {count} ocurrencia(s):")
        lines.extend(examples.get(issue_type, []))

    return "\n".join(lines)


def make_client(config: dict) -> tuple["OpenAI", str]:
    """Construye el cliente del Proposer a partir de config.yaml."""
    cfg = config.get("proposer", {})
    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )
    return client, cfg["model"]


def propose(
    client: OpenAI,
    model: str,
    current_prompt: str,
    all_issues: list[dict],
) -> str:
    """
    Genera una variante mejorada del prompt.

    Args:
        client:         cliente OpenAI-compatible (MiniMax, etc.)
        model:          nombre del modelo
        current_prompt: prompt actual a mejorar
        all_issues:     issues agregados de todas las muestras

    Returns:
        Nuevo prompt propuesto como string
    """
    error_summary = _build_error_summary(all_issues)

    meta = _META_PROMPT.format(
        error_summary=error_summary,
        current_prompt=current_prompt,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": meta}],
    )

    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("Proposer devolvió respuesta vacía o sin contenido")
    raw = response.choices[0].message.content.strip()
    # MiniMax M2.7 incluye bloques <think>...</think> — eliminarlos
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not raw:
        raise ValueError("Proposer devolvió contenido vacío tras limpiar <think>"
                         )
    return raw
