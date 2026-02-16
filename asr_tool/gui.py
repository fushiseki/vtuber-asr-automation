from __future__ import annotations

import gradio as gr
from pathlib import Path

from asr_tool.core import ASRCore, transcribe_to_files


asr = ASRCore(model_name="base", compute_type="float16")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "transcripts"


def transcribe_audio(file_path: str, language: str):
    try:
        outputs = transcribe_to_files(
            asr=asr,
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            language=language,
            formats=("srt",),
            add_timestamp_suffix=True,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}", None

    srt_file = next((p for p in outputs if p.suffix == ".srt"), None)
    return f"Transcription complete. File saved to {srt_file}", str(srt_file) if srt_file else None


def build_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown("## 🎙️ Audio/Video Transcriber with Faster-Whisper")

        with gr.Row():
            audio_input = gr.Audio(label="Upload MP3/MP4", type="filepath")
            language_input = gr.Dropdown(
                label="Language",
                choices=["auto", "en", "es", "ja", "pt", "fr", "de"],
                value="auto",
            )

        transcribe_btn = gr.Button("Transcribe")
        output_msg = gr.Textbox(label="Status", interactive=False)
        download_file = gr.File(label="Download Transcript")

        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language_input],
            outputs=[output_msg, download_file],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
