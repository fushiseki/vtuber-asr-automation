from __future__ import annotations

import argparse
from pathlib import Path

from asr_tool.core import ASRCore, transcribe_to_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless faster-whisper transcription tool")
    parser.add_argument("--input", required=True, help="Input audio/video file or directory")
    parser.add_argument("--output-dir", required=True, help="Directory for transcript output")
    parser.add_argument("--language", default="auto", choices=["auto", "ja", "en"], help="Language")
    parser.add_argument(
        "--formats",
        default="txt,srt",
        help="Comma-separated output formats (txt,srt)",
    )
    return parser.parse_args()


def run_batch(input_path: Path, output_dir: Path, language: str, formats: list[str]) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    asr = ASRCore(model_name="base", compute_type="float16")

    targets: list[Path]
    if input_path.is_dir():
        targets = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".mp4"}
            ]
        )
    else:
        targets = [input_path]

    for target in targets:
        transcribe_to_files(
            asr=asr,
            file_path=target,
            output_dir=output_dir,
            language=language,
            formats=formats,
            base_name=target.stem,
            add_timestamp_suffix=False,
        )

    return 0


def main() -> int:
    args = parse_args()
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    return run_batch(Path(args.input), Path(args.output_dir), args.language, formats)


if __name__ == "__main__":
    raise SystemExit(main())
