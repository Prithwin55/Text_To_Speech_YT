"""Gradio UI for the Text-to-Speech Story Pipeline."""

import os

import gradio as gr

from tts_engine import SUPPORTED_LANGUAGES, generate_speech

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _preview_text(txt_file) -> str:
    if txt_file is None:
        return ""
    try:
        with open(txt_file, encoding="utf-8") as f:
            text = f.read().strip()
        words = len(text.split())
        chars = len(text)
        # Rough estimate: ~220 chars per chunk, ~2 s/chunk on GPU, ~8 s on CPU
        from tts_engine import MAX_CHUNK_CHARS, split_into_chunks
        chunks = split_into_chunks(text)
        return (
            f"**{words:,} words · {chars:,} chars · {len(chunks)} TTS chunks**\n\n"
            f"*Estimated time: {len(chunks) * 2}–{len(chunks) * 8} s depending on hardware.*"
        )
    except Exception as e:
        return f"Could not preview file: {e}"


def run_pipeline(
    txt_file,
    language_name: str,
    ref_audio: str | None,
    fmt: str,
    progress=gr.Progress(track_tqdm=True),
):
    if txt_file is None:
        raise gr.Error("Please upload a .txt story script.")

    with open(txt_file, encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise gr.Error("The uploaded text file is empty.")

    lang = SUPPORTED_LANGUAGES[language_name]

    out_path = generate_speech(
        text=text,
        language=lang,
        reference_audio=ref_audio,
        output_format=fmt,
        output_dir=OUTPUT_DIR,
        progress_callback=progress,
    )
    return out_path, out_path


# ──────────────────────────────────────────────────────────────────────────────
# UI layout
# ──────────────────────────────────────────────────────────────────────────────

css = """
.generate-btn { font-size: 1.1rem !important; padding: 14px 24px !important; }
.tip { color: #888; font-size: 0.85rem; margin-top: -8px; }
"""

with gr.Blocks(title="TTS Story Pipeline") as demo:

    gr.Markdown(
        "# Text-to-Speech Story Pipeline\n"
        "Convert your story script to natural-sounding speech.  \n"
        "Supports **Hindi** and 16 other languages · **Voice cloning** via reference audio."
    )

    with gr.Row(equal_height=False):

        # ── Left column: inputs ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            txt_file = gr.File(
                label="Story Script (.txt)",
                file_types=[".txt"],
            )
            text_info = gr.Markdown(elem_classes="tip")
            txt_file.change(_preview_text, txt_file, text_info)

            language = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value="English",
                label="Language",
            )

            with gr.Accordion("Voice Cloning (optional)", open=False):
                gr.Markdown(
                    "Upload or record **6–30 seconds** of clear speech in the target language.  \n"
                    "Leave empty to use the built-in default voice.",
                    elem_classes="tip",
                )
                ref_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

            fmt = gr.Radio(["MP3", "WAV"], value="MP3", label="Output Format")

            generate_btn = gr.Button(
                "Generate Speech", variant="primary", elem_classes="generate-btn"
            )

        # ── Right column: outputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            out_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
            out_file = gr.File(label="Download")

    # ── Event wiring ───────────────────────────────────────────────────────────
    generate_btn.click(
        fn=run_pipeline,
        inputs=[txt_file, language, ref_audio, fmt],
        outputs=[out_audio, out_file],
        show_progress="full",
    )

    gr.Markdown(
        "---\n"
        "*Powered by [Coqui XTTS-v2](https://github.com/coqui-ai/TTS).  "
        "First run downloads the model (~1.8 GB).*",
        elem_classes="tip",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=css,
    )
