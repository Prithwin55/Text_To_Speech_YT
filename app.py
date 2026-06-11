"""Gradio UI for the Text-to-Speech Story Pipeline."""

import os

import gradio as gr
from pydub import AudioSegment

from tts_engine import SUPPORTED_LANGUAGES, generate_speech, mix_with_bgm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MIN_REF_WARN_SEC  = 3.0   # warn if reference audio shorter than this
MIN_REF_ERR_SEC   = 0.5   # hard error if reference audio shorter than this
MIN_SPEECH_SEC    = 0.5   # hard error if trimmed speech shorter than this
MIN_BGM_SEC       = 0.5   # hard error if trimmed BGM shorter than this

# ──────────────────────────────────────────────────────────────────────────────
# Reference texts
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
        "escuchando el canto de los pájaros. Sonrió suavemente y respiró profundo."
    ),
    "fr": (
        "La lumière du matin se répandit lentement sur les collines, teintant le ciel "
        "d'un orange doré et chaud. Une jeune femme marchait sur l'étroit sentier, "
        "écoutant le chant des oiseaux. Elle sourit doucement et prit une profonde inspiration."
    ),
    "de": (
        "Das Morgenlicht breitete sich langsam über die Hügel aus. Eine junge Frau ging "
        "den schmalen Pfad entlang und lauschte dem Gesang der Vögel in den hohen Bäumen. "
        "Sie lächelte sanft und atmete tief die frische Morgenluft ein."
    ),
    "it": (
        "La luce del mattino si diffuse lentamente sulle colline. Una giovane donna camminava "
        "lungo il sentiero stretto, ascoltando il canto degli uccelli. Sorrise dolcemente "
        "e fece un lungo, lento respiro d'aria fresca."
    ),
}
_DEFAULT_REF_TEXT = _REF_TEXTS["en"]


def _ref_text_for(lang_name: str) -> str:
    code = SUPPORTED_LANGUAGES.get(lang_name, "en")
    return _REF_TEXTS.get(code, _DEFAULT_REF_TEXT)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dur_label(path) -> str:
    if not path:
        return ""
    try:
        dur = len(AudioSegment.from_file(path)) / 1000.0
        m, s = divmod(int(dur), 60)
        return f"Duration: **{m}:{s:02d}**" if m else f"Duration: **{dur:.1f} s**"
    except Exception:
        return ""


def _preview_text(txt_file) -> str:
    if txt_file is None:
        return ""
    try:
        with open(txt_file, encoding="utf-8") as f:
            text = f.read().strip()
        from tts_engine import split_into_chunks
        chunks = split_into_chunks(text)
        words = len(text.split())
        return (
            f"**{words:,} words · {len(text):,} chars · {len(chunks)} TTS chunks**  \n"
            f"Estimated GPU time: ~{len(chunks)}–{len(chunks)*2} s"
        )
    except Exception as e:
        return f"Preview error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Validation handlers  (called from audio .change events)
# ──────────────────────────────────────────────────────────────────────────────

def _on_ref_change(path):
    """Fires when reference audio is uploaded, recorded, or trimmed via waveform."""
    if not path:
        return "", gr.update(visible=True)

    try:
        dur = len(AudioSegment.from_file(path)) / 1000.0
    except Exception as e:
        raise gr.Error(f"Cannot read reference audio: {e}")

    if dur < MIN_REF_ERR_SEC:
        raise gr.Error(
            f"Reference audio is only {dur:.2f}s — too short to use. "
            f"Please record at least {MIN_REF_ERR_SEC}s."
        )
    if dur < MIN_REF_WARN_SEC:
        gr.Warning(
            f"Reference audio is {dur:.1f}s. "
            f"For best voice cloning results use 6–30 s of clear speech."
        )

    return _dur_label(path), gr.update(visible=False)


def _on_speech_change(path):
    """Fires when generated speech changes or is trimmed via waveform."""
    if not path:
        return None, None, ""

    try:
        dur = len(AudioSegment.from_file(path)) / 1000.0
    except Exception as e:
        raise gr.Error(f"Cannot read speech audio: {e}")

    if dur < MIN_SPEECH_SEC:
        raise gr.Error(
            f"Speech is only {dur:.2f}s after trimming — too short. "
            f"Select a region of at least {MIN_SPEECH_SEC}s."
        )

    return path, path, _dur_label(path)   # state, out_file, label


def _on_bgm_change(path):
    """Fires when BGM is uploaded or trimmed via waveform."""
    if not path:
        return ""

    try:
        dur = len(AudioSegment.from_file(path)) / 1000.0
    except Exception as e:
        raise gr.Error(f"Cannot read BGM file: {e}")

    if dur < MIN_BGM_SEC:
        raise gr.Error(
            f"BGM clip is only {dur:.2f}s — too short to loop. "
            f"Please select at least {MIN_BGM_SEC}s."
        )

    return _dur_label(path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline handlers
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    txt_file, language_name, ref_audio, gender, target_dur, fmt,
    progress=gr.Progress(track_tqdm=True),
):
    if txt_file is None:
        raise gr.Error("Please upload a .txt story script.")
    with open(txt_file, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise gr.Error("The uploaded text file is empty.")

    lang     = SUPPORTED_LANGUAGES[language_name]
    dur_sec  = float(target_dur) if target_dur and float(target_dur) > 0 else None

    out_path = generate_speech(
        text=text, language=lang, reference_audio=ref_audio,
        gender=gender, output_format=fmt, output_dir=OUTPUT_DIR,
        target_duration=dur_sec, progress_callback=progress,
    )
    # out_audio, out_file, speech_path_state, dur_label
    return out_path, out_path, out_path, _dur_label(out_path)


def run_mix(speech_path, bgm_path, bgm_vol):
    if not speech_path:
        raise gr.Error("Generate speech first before mixing.")
    if bgm_path is None:
        raise gr.Error("Upload a background music file.")
    out = mix_with_bgm(speech_path, bgm_path, bgm_vol, OUTPUT_DIR)
    return out, out


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

css = """
/* ── Fix waveform: no horizontal scroll ───────────────────────────────────── */
/* WaveSurfer sets overflow:hidden auto on its <wave> element inline;
   we override to prevent any scrolling — the waveform scales to fit instead. */
wave {
    overflow: hidden !important;
}
wave > div {
    overflow: hidden !important;
}
/* Catch-all for the gradio waveform wrapper */
.component-wrapper,
[data-testid="waveform-player"],
[data-testid="microphone-waveform"],
[data-testid="recording-waveform"] {
    overflow: hidden !important;
}

/* ── Typography ────────────────────────────────────────────────────────────── */
/* Make all label/span text fully visible (Soft theme can wash these out) */
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap span,
.gradio-container .prose span,
.gradio-container p {
    opacity: 1 !important;
}

/* Duration badge */
.dur-lbl p { font-weight: 600; font-size: 0.88rem; margin: 2px 0 6px 0; }

/* Reference text box */
.ref-box textarea {
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

/* Generate button */
.gen-btn { font-size: 1.05rem !important; padding: 12px 22px !important; }
"""

WAVE_REF    = gr.WaveformOptions(trim_region_color="#7eb2d4", waveform_color="#4a90d9")
WAVE_SPEECH = gr.WaveformOptions(trim_region_color="#82c982", waveform_color="#3a9e3a")
WAVE_BGM    = gr.WaveformOptions(trim_region_color="#f0c080", waveform_color="#d4860a")

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="TTS Story Pipeline") as demo:

    speech_path_state = gr.State(None)

    gr.Markdown(
        "# Text-to-Speech Story Pipeline\n"
        "Hindi & 16 other languages · Voice cloning · Duration control · BGM mixing"
    )

    with gr.Row(equal_height=False):

        # ── LEFT ──────────────────────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("## Input")
            txt_file  = gr.File(label="Story Script (.txt)", file_types=[".txt"])
            text_info = gr.Markdown()
            txt_file.change(_preview_text, txt_file, text_info)

            language  = gr.Dropdown(
                list(SUPPORTED_LANGUAGES.keys()), value="English", label="Language"
            )

            # ── Voice ─────────────────────────────────────────────────────────
            gr.Markdown("## Voice")
            gr.Markdown(
                "Record or upload **6–30 s** of clear speech to clone your voice.  \n"
                "Read the passage below for best results:"
            )
            ref_text_box = gr.Textbox(
                value=_ref_text_for("English"),
                label="Suggested passage to read aloud",
                lines=4,
                interactive=False,
                elem_classes="ref-box",
            )
            language.change(_ref_text_for, language, ref_text_box)

            ref_audio = gr.Audio(
                label="Reference Audio  — drag the two handles on the waveform, then press ✂ to trim",
                type="filepath",
                sources=["microphone", "upload"],
                waveform_options=WAVE_REF,
            )
            ref_dur_lbl = gr.Markdown(elem_classes="dur-lbl")

            with gr.Group(visible=True) as gender_group:
                gr.Markdown("No reference audio — choose a built-in voice:")
                gender = gr.Radio(["Female", "Male"], value="Female", label="Voice Gender")

            # Validation + gender toggle on every ref_audio change (upload / record / trim)
            ref_audio.change(
                _on_ref_change, ref_audio, [ref_dur_lbl, gender_group]
            )

            # ── Output settings ───────────────────────────────────────────────
            gr.Markdown("## Output Settings")
            target_dur = gr.Slider(
                0, 300, value=0, step=5,
                label="Target Duration — seconds  (0 = natural length)",
                info="Speech is time-stretched to fit. 0 = no adjustment.",
            )
            fmt = gr.Radio(["MP3", "WAV"], value="MP3", label="Format")
            generate_btn = gr.Button(
                "Generate Speech", variant="primary", elem_classes="gen-btn"
            )

        # ── RIGHT ─────────────────────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("## Generated Speech")
            gr.Markdown(
                "Drag the **two handles** on the waveform to select a region, "
                "then press **✂** to trim."
            )
            out_audio = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=True,          # must be True to enable waveform trimming
                waveform_options=WAVE_SPEECH,
            )
            speech_dur_lbl = gr.Markdown(elem_classes="dur-lbl")
            out_file = gr.File(label="Download Speech")

            # After trim (or new generation): validate + propagate state
            out_audio.change(
                _on_speech_change, out_audio,
                [speech_path_state, out_file, speech_dur_lbl]
            )

            # ── BGM ───────────────────────────────────────────────────────────
            with gr.Accordion("Background Music Mixing", open=False):
                gr.Markdown(
                    "Upload a track — it will be **looped** to the speech duration and "
                    "blended underneath.  \n"
                    "Drag the **two handles** on the waveform, then press **✂** to select "
                    "the loop region."
                )
                bgm_file = gr.Audio(
                    label="Background Music",
                    type="filepath",
                    sources=["upload"],
                    waveform_options=WAVE_BGM,
                )
                bgm_dur_lbl = gr.Markdown(elem_classes="dur-lbl")
                bgm_file.change(_on_bgm_change, bgm_file, bgm_dur_lbl)

                bgm_vol = gr.Slider(
                    -35, -5, value=-18, step=1,
                    label="BGM Volume (dB relative to speech)",
                    info="-18 dB = subtle background · -5 dB = prominent music",
                )
                mix_btn = gr.Button("Mix BGM with Speech", variant="secondary")

                gr.Markdown("---")
                mixed_audio = gr.Audio(label="Mixed Output", type="filepath", interactive=False)
                mixed_file  = gr.File(label="Download Mixed")

    # ── Event wiring ──────────────────────────────────────────────────────────

    generate_btn.click(
        fn=run_pipeline,
        inputs=[txt_file, language, ref_audio, gender, target_dur, fmt],
        outputs=[out_audio, out_file, speech_path_state, speech_dur_lbl],
        show_progress="full",
    )

    mix_btn.click(
        fn=run_mix,
        inputs=[speech_path_state, bgm_file, bgm_vol],
        outputs=[mixed_audio, mixed_file],
        show_progress="minimal",
    )

    gr.Markdown(
        "---\n*Powered by Coqui XTTS-v2. First launch downloads the model (~1.8 GB).*"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=css,
    )
