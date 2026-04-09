"""
Test rápido de conexión con el proposer configurado en config.yaml.
Envía un mensaje mínimo y muestra la respuesta.

Uso:
    python scripts/test_proposer.py
"""

import sys
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).parent.parent


def main():
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: No se encuentra {config_path}")
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text())
    cfg = config.get("proposer", {})

    if not cfg:
        print("ERROR: No hay sección 'proposer' en config.yaml")
        sys.exit(1)

    print(f"Proposer:  {cfg.get('base_url')}")
    print(f"Modelo:    {cfg.get('model')}")
    print("Enviando mensaje de prueba...")

    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": "Responde solo con: OK"}],
    )

    print(f"Choices:   {len(response.choices)}")
    if response.choices:
        msg = response.choices[0].message
        print(f"Role:      {msg.role}")
        print(f"Content:   {repr(msg.content)}")
        if hasattr(msg, 'reasoning') and msg.reasoning:
            print(f"Reasoning: {repr(msg.reasoning[:80])}...")
        finish = response.choices[0].finish_reason
        print(f"Finish:    {finish}")

    content = response.choices[0].message.content if response.choices else None
    if content:
        print(f"\nRespuesta: {content.strip()}")
        print("Conexión OK")
    else:
        print("\nERROR: Respuesta vacía")
        sys.exit(1)


if __name__ == "__main__":
    main()
