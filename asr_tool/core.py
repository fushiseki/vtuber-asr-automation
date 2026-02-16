from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class ASRCore:
    """Reusable faster-whisper transcription logic (no UI code)."""

    def __init__(self, model_name: str = "base", compute_type: str = "float16") -> None:
        self.model = WhisperModel(model_name, compute_type=compute_type)

    def transcribe_segments(self, file_path: str | Path, language: str = "auto") -> List[TranscriptSegment]:
        audio_path = Path(file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Invalid audio file: {audio_path}")

        segments, _info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            language=None if language == "auto" else language,
            vad_filter=True,
        )

        return [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments
        ]


def format_seconds_to_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(segments: Sequence[TranscriptSegment], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}\n")
    return path


def write_srt(
    segments: Sequence[TranscriptSegment],
    output_path: str | Path,
    header_lines: Optional[Sequence[str]] = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header_lines:
            for line in header_lines:
                f.write(f"{line}\n")
            f.write("\n")

        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{format_seconds_to_srt(seg.start)} --> {format_seconds_to_srt(seg.end)}\n"
            )
            f.write(f"{seg.text}\n\n")
    return path


def transcribe_to_files(
    asr: ASRCore,
    file_path: str | Path,
    output_dir: str | Path,
    language: str = "auto",
    formats: Iterable[str] = ("txt",),
    base_name: Optional[str] = None,
    add_timestamp_suffix: bool = False,
    srt_header_lines: Optional[Sequence[str]] = None,
) -> List[Path]:
    src = Path(file_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    segment_data = asr.transcribe_segments(src, language=language)

    stem = base_name or src.stem
    if add_timestamp_suffix:
        stem = f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    saved: List[Path] = []
    normalized = {fmt.lower().strip(".") for fmt in formats}
    if "txt" in normalized:
        saved.append(write_txt(segment_data, output_root / f"{stem}.txt"))
    if "srt" in normalized:
        saved.append(write_srt(segment_data, output_root / f"{stem}.srt", header_lines=srt_header_lines))

    if not saved:
        raise ValueError("No valid transcript formats requested. Use txt and/or srt.")

    return saved
