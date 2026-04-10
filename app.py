import json
import sys
from pathlib import Path

import streamlit as st
import yaml
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gepa import adapter  # noqa: E402

st.set_page_config(page_title="OCR Facturas", layout="wide")


@st.cache_data
def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_extraction_config(config: dict) -> tuple[str, dict]:
    extraction = config.get("extraction", {})
    prompt_file = extraction.get("prompt_file", "prompt.txt")
    schema_file = extraction.get("schema_file", "schema.json")
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    schema = json.loads(Path(schema_file).read_text(encoding="utf-8"))
    return prompt, schema


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def pdf_to_base64(pdf_bytes: bytes) -> str:
    import base64
    return base64.b64encode(pdf_bytes).decode()


def render_pdf_embed(pdf_bytes: bytes, key: str) -> None:
    b64 = pdf_to_base64(pdf_bytes)
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="520px" style="border:none;"></iframe>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label="Descargar PDF",
        data=pdf_bytes,
        file_name="factura.pdf",
        mime="application/pdf",
        key=key,
    )


def process_file(client: OpenAI, config: dict, name: str, pdf_bytes: bytes) -> None:
    dpi = config["processing"].get("dpi", 200)
    first_only = config["processing"].get("first_page_only", False)

    with st.spinner("Convirtiendo PDF a imágenes..."):
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        if first_only:
            images = images[:1]

    col_pdf, col_img, col_json = st.columns([2, 2, 3])

    with col_pdf:
        st.markdown("**PDF**")
        render_pdf_embed(pdf_bytes, key=f"dl_{name}")

    with col_img:
        st.markdown("**Imágenes enviadas al modelo**")
        for image in images:
            st.image(image, width="stretch")

    with col_json:
        st.markdown("**Respuesta del modelo**")
        with st.spinner("Consultando modelo..."):
            try:
                prompt, schema = load_extraction_config(config)
                model = config["lmstudio"]["model"]
                result = adapter.run_ocr_paged(client, model, schema, prompt, images)
                st.json(result)
            except json.JSONDecodeError as e:
                st.error(f"El modelo no devolvió JSON válido: {e}")
            except Exception as e:
                st.error(f"Error llamando al modelo: {e}")


def main() -> None:
    config = load_config()

    client = OpenAI(
        base_url=config["lmstudio"]["base_url"],
        api_key=config["lmstudio"].get("api_key", "lm-studio"),
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuración")
        st.markdown(f"**Endpoint:** `{config['lmstudio']['base_url']}`")
        st.markdown(f"**Modelo:** `{config['lmstudio']['model']}`")
        st.markdown(f"**DPI:** `{config['processing'].get('dpi', 200)}`")
        first_only = config["processing"].get("first_page_only", False)
        st.markdown(f"**Solo primera página:** `{first_only}`")
        extraction = config.get("extraction", {})
        st.markdown(f"**Prompt:** `{extraction.get('prompt_file', 'prompt.txt')}`")
        st.markdown(f"**Schema:** `{extraction.get('schema_file', 'schema.json')}`")


    # ── Main ─────────────────────────────────────────────────────────────────
    st.title("OCR Facturas")
    st.caption("Sube PDFs para extraer datos con el modelo local.")

    uploaded_files = st.file_uploader(
        "Selecciona facturas (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Sube uno o más PDFs para comenzar.")
        return

    if st.button("Procesar facturas", type="primary"):
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                process_file(client, config, uploaded_file.name, uploaded_file.read())


if __name__ == "__main__":
    main()
