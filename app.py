"""Gradio UI for the Text-to-Speech Story Pipeline."""

import os

import gradio as gr

from tts_engine import SUPPORTED_LANGUAGES, generate_speech

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ──────────────────────────────────────────────────────────────────────────────
# Reference texts — phonetically varied passages to read aloud for cloning
# ──────────────────────────────────────────────────────────────────────────────

_REF_TEXTS: dict[str, str] = {
    "en": (
        "The morning light spread slowly over the hills, turning the sky from deep "
        "purple to a warm golden orange. A young woman walked along the narrow path, "
        "listening to the birds singing in the tall trees beside her. She smiled gently "
        "and took a long, slow breath of the fresh morning air."
    ),
    "hi": (
        "सुबह की रोशनी धीरे-धीरे पहाड़ियों पर फैल गई और आसमान गहरे नीले से सुनहरे "
        "नारंगी रंग में बदल गया। एक युवती पतली पगडंडी पर चलती रही, ऊंचे पेड़ों में "
        "चिड़ियों का मधुर गीत सुनती रही। उसने धीरे से मुस्कुराया और ताज़ी हवा की "
        "एक लंबी, गहरी सांस ली।"
    ),
    "es": (
        "La luz de la mañana se extendió lentamente sobre las colinas, tiñendo el cielo "
        "de un cálido naranja dorado. Una joven caminaba por el sendero estrecho, "
        "escuchando el canto de los pájaros en los altos árboles a su lado. Sonrió "
        "suavemente y respiró profundo el aire fresco de la mañana."
    ),
    "fr": (
        "La lumière du matin se répandit lentement sur les collines, teintant le ciel "
        "d'un orange doré et chaud. Une jeune femme marchait sur l'étroit sentier, "
        "écoutant le chant des oiseaux dans les grands arbres à ses côtés. Elle sourit "
        "doucement et prit une longue et profonde inspiration d'air frais."
    ),
    "de": (
        "Das Morgenlicht breitete sich langsam über die Hügel aus und tauchte den Himmel "
        "in ein warmes Goldorange. Eine junge Frau ging den schmalen Pfad entlang und "
        "lauschte dem Gesang der Vögel in den hohen Bäumen neben ihr. Sie lächelte sanft "
        "und atmete tief die frische Morgenluft ein."
    ),
    "it": (
        "La luce del mattino si diffuse lentamente sulle colline, tingendo il cielo di un "
        "caldo arancio dorato. Una giovane donna camminava lungo il sentiero stretto, "
        "ascoltando il canto degli uccelli negli alti alberi accanto a lei. Sorrise "
        "dolcemente e fece un lungo, lento respiro d'aria fresca."
    ),
}
# Fallback: use English text for languages without a specific passage
_DEFAULT_REF_TEXT = _REF_TEXTS["en"]


def _ref_text_for(language_name: str) -> str:
    lang_code = SUPPORTED_LANGUAGES.get(language_name, "en")
    text = _REF_TEXTS.get(lang_code, _DEFAULT_REF_TEXT)
    return (
        f"**Read this aloud** (6–30 s) to clone your voice in {language_name}:\n\n"
        f"> {text}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preview_text(txt_file) -> str:
    if txt_file is None:
        return ""
    try:
        with open(txt_file, encoding="utf-8") as f:
            text = f.read().strip()
        words = len(text.split())
        chars = len(text)
        from tts_engine import split_into_chunks
        chunks = split_into_chunks(text)
        return (
            f"**{words:,} words · {chars:,} chars · {len(chunks)} TTS chunks**\n\n"
            f"*Estimated time on GPU: ~{len(chunks)}–{len(chunks) * 2} s.*"
        )
    except Exception as e:
        return f"Could not preview: {e}"


def _toggle_gender(ref_audio_val):
    """Show the gender picker only when no reference audio is provided."""
    return gr.update(visible=ref_audio_val is None)


def run_pipeline(
    txt_file,
    language_name: str,
    ref_audio: str | None,
    gender: str,
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
        gender=gender,
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
.tip  { color: #888; font-size: 0.85rem; margin-top: -8px; }
.ref-text blockquote { border-left: 3px solid #aaa; padding-left: 10px; color: #555; }
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

            txt_file = gr.File(label="Story Script (.txt)", file_types=[".txt"])
            text_info = gr.Markdown(elem_classes="tip")
            txt_file.change(_preview_text, txt_file, text_info)

            language = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value="English",
                label="Language",
            )

            gr.Markdown("### Voice")

            # ── Reference text prompt ──────────────────────────────────────────
            ref_text_md = gr.Markdown(
                value=_ref_text_for("English"),
                elem_classes="ref-text",
            )
            language.change(_ref_text_for, language, ref_text_md)

            # ── Microphone / upload (mic first so it's prominent) ──────────────
            ref_audio = gr.Audio(
                label="Record or upload reference audio for voice cloning (optional)",
                type="filepath",
                sources=["microphone", "upload"],
            )

            # ── Gender picker — only visible when no reference audio ───────────
            with gr.Group(visible=True) as gender_group:
                gr.Markdown(
                    "*No reference audio — choose a built-in voice:*",
                    elem_classes="tip",
                )
                gender = gr.Radio(
                    ["Female", "Male"],
                    value="Female",
                    label="Voice Gender",
                )

            ref_audio.change(_toggle_gender, ref_audio, gender_group)

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
        inputs=[txt_file, language, ref_audio, gender, fmt],
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
