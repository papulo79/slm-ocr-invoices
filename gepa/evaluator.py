"""
Evaluador GEPA: compara JSON predicho vs ground truth y produce métricas.

Métricas:
  - Por campo: seller_vat, seller_name, buyer_vat, buyer_name,
               invoice_number, invoice_date, amount, vat_amount,
               line_descriptions, line_prices
  - Score global ponderado (0–1)

Issues detectados:
  - letter_vs_number  → NIF que debería empezar por letra empieza por dígito
  - truncation        → descripción predicha es prefijo de la real
  - empty_field       → campo requerido vacío en la predicción
"""

import re
from difflib import SequenceMatcher
from typing import Any

# Pesos de cada métrica (suman 1.0)
WEIGHTS: dict[str, float] = {
    "seller_vat":        0.20,
    "seller_name":       0.08,
    "buyer_vat":         0.10,
    "buyer_name":        0.07,
    "invoice_number":    0.08,
    "invoice_date":      0.08,
    "amount":            0.10,
    "vat_amount":        0.07,
    "line_descriptions": 0.15,
    "line_prices":       0.07,
}

# Letras válidas que inician un NIF/CIF español
_NIF_LETTER_STARTS = set("ABCDEFGHJKLMNPQRSUVW")


def _strip_country_prefix(vat: str) -> str:
    """Elimina prefijos de país como ES, FR, DE del inicio del VAT."""
    if vat and len(vat) > 2 and vat[:2].isalpha() and vat[2:3].isalnum():
        return vat[2:].upper()
    return vat.upper() if vat else ""


def _similarity(a: str, b: str) -> float:
    """Ratio de similitud textual entre dos strings (0–1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _normalize_date(date_str: str) -> str:
    """Intenta normalizar fechas a YYYY-MM-DD."""
    if not date_str:
        return ""
    # Ya en formato correcto
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str
    # DD/MM/YYYY o DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", date_str)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return date_str


def _amount_score(pred: Any, gold: Any, tolerance: float = 0.01) -> float:
    try:
        return 1.0 if abs(float(pred) - float(gold)) <= tolerance else 0.0
    except (TypeError, ValueError):
        return 1.0 if pred == gold else 0.0


def _line_descriptions_score(pred_items: list, gold_items: list) -> float:
    """Score medio de similitud de descripciones alineadas por posición."""
    if not gold_items:
        return 1.0 if not pred_items else 0.0
    if not pred_items:
        return 0.0
    scores = []
    for i, gold_item in enumerate(gold_items):
        gold_desc = gold_item.get("description", "")
        if i < len(pred_items):
            pred_desc = pred_items[i].get("description", "")
        else:
            pred_desc = ""
        scores.append(_similarity(pred_desc, gold_desc))
    return sum(scores) / len(scores)


def _line_prices_score(pred_items: list, gold_items: list) -> float:
    """Score medio de exactitud de precios unitarios."""
    if not gold_items:
        return 1.0 if not pred_items else 0.0
    if not pred_items:
        return 0.0
    scores = []
    for i, gold_item in enumerate(gold_items):
        gold_price = gold_item.get("unit_price", 0)
        pred_price = pred_items[i].get("unit_price", 0) if i < len(pred_items) else None
        scores.append(_amount_score(pred_price, gold_price))
    return sum(scores) / len(scores)


def _detect_issues(pred: dict, gold: dict) -> list[dict]:
    issues = []

    def _check_vat(pred_vat: str, gold_vat: str, field: str) -> None:
        if not gold_vat:
            return
        p = _strip_country_prefix(pred_vat or "")
        g = _strip_country_prefix(gold_vat)
        if not p:
            issues.append({"field": field, "type": "empty_field",
                           "expected": gold_vat, "actual": pred_vat or "", "severity": "high"})
            return
        # B vs 8: gold empieza por letra válida, pred empieza por dígito
        if g and g[0] in _NIF_LETTER_STARTS and p and p[0].isdigit():
            issues.append({"field": field, "type": "ocr_confusion",
                           "subtype": "letter_vs_number",
                           "expected": gold_vat, "actual": pred_vat, "severity": "high"})
        elif p != g:
            issues.append({"field": field, "type": "mismatch",
                           "expected": gold_vat, "actual": pred_vat, "severity": "medium"})

    _check_vat(
        (pred.get("seller") or {}).get("vat_id", ""),
        (gold.get("seller") or {}).get("vat_id", ""),
        "seller.vat_id",
    )
    _check_vat(
        (pred.get("buyer") or {}).get("vat_id", ""),
        (gold.get("buyer") or {}).get("vat_id", ""),
        "buyer.vat_id",
    )

    # Campos obligatorios vacíos
    required_fields = [
        ("invoice_number", "medium"),
        ("invoice_date", "medium"),
        ("amount", "medium"),
    ]
    for field, severity in required_fields:
        gold_val = gold.get(field)
        pred_val = pred.get(field)
        if gold_val is not None and gold_val != "" and (pred_val is None or pred_val == ""):
            issues.append({"field": field, "type": "empty_field",
                           "expected": str(gold_val), "actual": str(pred_val or ""), "severity": severity})

    # Truncamiento en line_items
    gold_items = gold.get("line_items", [])
    pred_items = pred.get("line_items", [])
    for i, gold_item in enumerate(gold_items):
        gold_desc = gold_item.get("description", "")
        if i >= len(pred_items):
            issues.append({"field": f"line_items[{i}].description", "type": "missing_item",
                           "expected": gold_desc, "actual": "", "severity": "high"})
            continue
        pred_desc = pred_items[i].get("description", "")
        if gold_desc and pred_desc and len(pred_desc) < len(gold_desc) * 0.75:
            if gold_desc.startswith(pred_desc) or _similarity(pred_desc, gold_desc[:len(pred_desc)]) > 0.9:
                issues.append({"field": f"line_items[{i}].description", "type": "truncation",
                               "expected": gold_desc, "actual": pred_desc, "severity": "medium"})

    return issues


def evaluate(pred: dict, gold: dict, sample_name: str = "") -> dict:
    """
    Evalúa una predicción contra el ground truth.

    Returns:
        dict con 'sample', 'scores' (por métrica), 'total_score' y 'issues'
    """
    pred_seller = pred.get("seller") or {}
    gold_seller = gold.get("seller") or {}
    pred_buyer = pred.get("buyer") or {}
    gold_buyer = gold.get("buyer") or {}

    # --- scores por campo ---
    seller_vat_pred = _strip_country_prefix(pred_seller.get("vat_id", ""))
    seller_vat_gold = _strip_country_prefix(gold_seller.get("vat_id", ""))
    buyer_vat_pred = _strip_country_prefix(pred_buyer.get("vat_id", ""))
    buyer_vat_gold = _strip_country_prefix(gold_buyer.get("vat_id", ""))

    scores: dict[str, float] = {
        "seller_vat": 1.0 if seller_vat_pred == seller_vat_gold else 0.0,
        "seller_name": _similarity(pred_seller.get("name", ""), gold_seller.get("name", "")),
        "buyer_vat": 1.0 if buyer_vat_pred == buyer_vat_gold else 0.0,
        "buyer_name": _similarity(pred_buyer.get("name", ""), gold_buyer.get("name", "")),
        "invoice_number": _similarity(
            pred.get("invoice_number", ""), gold.get("invoice_number", "")
        ),
        "invoice_date": 1.0 if (
            _normalize_date(pred.get("invoice_date", "")) ==
            _normalize_date(gold.get("invoice_date", ""))
        ) else 0.0,
        "amount": _amount_score(pred.get("amount"), gold.get("amount")),
        "vat_amount": _amount_score(pred.get("vat_amount"), gold.get("vat_amount")),
        "line_descriptions": _line_descriptions_score(
            pred.get("line_items", []), gold.get("line_items", [])
        ),
        "line_prices": _line_prices_score(
            pred.get("line_items", []), gold.get("line_items", [])
        ),
    }

    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    issues = _detect_issues(pred, gold)

    return {
        "sample": sample_name,
        "scores": scores,
        "total_score": round(total, 4),
        "issues": issues,
    }


def evaluate_dataset(results: list[tuple[str, dict, dict]]) -> dict:
    """
    Evalúa una lista de (nombre, predicción, gold).

    Returns:
        dict con 'per_sample', 'mean_scores' y 'mean_total'
    """
    per_sample = [evaluate(pred, gold, name) for name, pred, gold in results]
    all_scores: dict[str, list[float]] = {k: [] for k in WEIGHTS}
    totals: list[float] = []

    for r in per_sample:
        for k in WEIGHTS:
            all_scores[k].append(r["scores"][k])
        totals.append(r["total_score"])

    mean_scores = {k: round(sum(v) / len(v), 4) for k, v in all_scores.items()}
    mean_total = round(sum(totals) / len(totals), 4) if totals else 0.0

    return {
        "per_sample": per_sample,
        "mean_scores": mean_scores,
        "mean_total": mean_total,
    }
