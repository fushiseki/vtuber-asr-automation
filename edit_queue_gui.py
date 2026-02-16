from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr

from validation_model import Queue, QueueEntry, TimestampSegment

ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queue.json"


def load_queue() -> Queue:
    if not QUEUE_PATH.exists():
        return Queue(entries=[])
    raw = QUEUE_PATH.read_text(encoding="utf-8").strip()
    return Queue(entries=[]) if not raw else Queue.model_validate_json(raw)


def save_queue(queue: Queue) -> None:
    QUEUE_PATH.write_text(queue.model_dump_json(indent=2), encoding="utf-8")


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    sec = int(seconds)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def waiting_entries(queue: Queue) -> list[QueueEntry]:
    return [e for e in queue.entries if e.status == "waiting_timestamps"]


def dropdown_choices() -> list[tuple[str, str]]:
    return [
        (f"{e.video_id} - {e.title or 'Untitled'}", e.video_id)
        for e in waiting_entries(load_queue())
    ]


def parse_segments(raw_text: str) -> list[TimestampSegment]:
    segments: list[TimestampSegment] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("-", maxsplit=1)]
        if len(parts) != 2:
            raise ValueError(f"Invalid line format: {line}")
        start, end = parts
        end_value = None if end.lower() == "end" or end == "" else end
        segments.append(TimestampSegment(start=start, end=end_value))
    return segments


def load_video(video_id: str):
    if not video_id:
        return "", "", "", ""
    entry = load_queue().get_by_id(video_id)
    if not entry:
        return "", "", "", "Video not found in queue."

    current = "\n".join(
        f"{s.start} - {s.end if s.end else 'end'}" for s in entry.segments
    )
    info = f"**Title:** {entry.title or 'Untitled'}\n\n**Duration:** {format_duration(entry.video_duration)}"
    link = f"[Open YouTube video]({entry.url})"
    return info, link, current, ""


def save_timestamps(video_id: str, text_value: str) -> str:
    try:
        queue = load_queue()
        entry = queue.get_by_id(video_id)
        if not entry:
            return "Error: selected video not found."
        if entry.status != "waiting_timestamps":
            return "Error: selected video is no longer waiting for timestamps."

        entry.segments = parse_segments(text_value)
        entry.status = "download_finished"
        Queue.model_validate(queue.model_dump())
        save_queue(queue)
        return f"Saved {len(entry.segments)} segment(s). Status changed to download_finished."
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def refresh_dropdown():
    choices = dropdown_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def build_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown("## Edit queue timestamps (waiting_timestamps only)")
        video = gr.Dropdown(label="Video", choices=[], value=None)
        info = gr.Markdown()
        link = gr.Markdown()
        segments = gr.Textbox(
            label="Segments",
            lines=10,
            placeholder=(
                "One per line:\n00:10:30 - 01:45:20\n01:50:00 - end\n"
                "Supports HH:MM:SS or MM:SS"
            ),
        )
        save_btn = gr.Button("Save")
        status = gr.Textbox(label="Status", interactive=False)

        app.load(fn=refresh_dropdown, outputs=[video])
        app.load(fn=lambda v: load_video(v), inputs=[video], outputs=[info, link, segments, status])
        video.change(fn=load_video, inputs=[video], outputs=[info, link, segments, status])
        save_btn.click(fn=save_timestamps, inputs=[video, segments], outputs=[status]).then(
            fn=refresh_dropdown, outputs=[video]
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=7888, share=False)
