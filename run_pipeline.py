from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from asr_tool.core import ASRCore, transcribe_to_files
from validation_model import Queue, QueueEntry, TimestampSegment

ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queue.json"
LOCK_PATH = ROOT / "pipeline.lock"
RAW_DIR = ROOT / "downloads" / "raw"
SEGMENTS_DIR = ROOT / "downloads" / "segments"
TRANSCRIPTS_DIR = ROOT / "output" / "transcripts"
TRASH_DIR = ROOT / "downloads" / "trash"

HOLODEX_URL = "https://holodex.net/api/v2/videos"
DEFAULT_CHANNEL_ID = "UC9p_lqQ0FEDz327Vgf5JwqA"
DEFAULT_STREAM_LIMIT = 5
DEFAULT_ASR_LANGUAGE = "en"
DEFAULT_ASR_OUTPUT_FORMAT = "srt"
DEFAULT_COOKIES_BROWSER = "firefox"
SUPPORTED_COOKIES_BROWSERS = {
    "brave",
    "chrome",
    "chromium",
    "edge",
    "firefox",
    "opera",
    "safari",
    "vivaldi",
    "whale",
}


class PipelineConfig:
    def __init__(
        self,
        holodex_api_key: str,
        channel_id: str,
        stream_limit: int,
        asr_language: str,
        asr_formats: tuple[str, ...],
        cookies_from_browser: str,
    ) -> None:
        self.holodex_api_key = holodex_api_key
        self.channel_id = channel_id
        self.stream_limit = stream_limit
        self.asr_language = asr_language
        self.asr_formats = asr_formats
        self.cookies_from_browser = cookies_from_browser


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def load_config() -> PipelineConfig:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("HOLODEX_API_KEY")
    if not api_key:
        raise RuntimeError("Missing HOLODEX_API_KEY in .env")

    channel_id = os.getenv("HOLODEX_CHANNEL_ID", DEFAULT_CHANNEL_ID).strip()
    if not channel_id:
        raise RuntimeError("HOLODEX_CHANNEL_ID is empty in .env")

    stream_limit_raw = os.getenv("HOLODEX_STREAM_LIMIT", str(DEFAULT_STREAM_LIMIT)).strip()
    try:
        stream_limit = int(stream_limit_raw)
    except ValueError as exc:
        raise RuntimeError("HOLODEX_STREAM_LIMIT must be an integer") from exc

    if stream_limit <= 0:
        raise RuntimeError("HOLODEX_STREAM_LIMIT must be greater than 0")

    asr_language = os.getenv("ASR_LANGUAGE", DEFAULT_ASR_LANGUAGE).strip().lower()
    if asr_language not in {"en", "ja", "auto"}:
        raise RuntimeError("ASR_LANGUAGE must be one of: en, ja, auto")

    asr_output_format = os.getenv("ASR_OUTPUT_FORMAT", DEFAULT_ASR_OUTPUT_FORMAT).strip().lower()
    if asr_output_format == "both":
        asr_formats = ("srt", "txt")
    elif asr_output_format in {"srt", "txt"}:
        asr_formats = (asr_output_format,)
    else:
        raise RuntimeError("ASR_OUTPUT_FORMAT must be one of: srt, txt, both")

    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", DEFAULT_COOKIES_BROWSER).strip().lower()
    if not cookies_from_browser:
        raise RuntimeError("YTDLP_COOKIES_FROM_BROWSER is empty in .env")
    if cookies_from_browser not in SUPPORTED_COOKIES_BROWSERS:
        supported = ", ".join(sorted(SUPPORTED_COOKIES_BROWSERS))
        raise RuntimeError(
            f"YTDLP_COOKIES_FROM_BROWSER must be one of: {supported}"
        )

    return PipelineConfig(
        holodex_api_key=api_key,
        channel_id=channel_id,
        stream_limit=stream_limit,
        asr_language=asr_language,
        asr_formats=asr_formats,
        cookies_from_browser=cookies_from_browser,
    )


def ensure_dirs() -> None:
    for path in [RAW_DIR, SEGMENTS_DIR, TRANSCRIPTS_DIR, TRASH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_queue() -> Queue:
    if not QUEUE_PATH.exists():
        return Queue(entries=[])

    raw_text = QUEUE_PATH.read_text(encoding="utf-8").strip()
    if not raw_text:
        return Queue(entries=[])

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"queue.json is not valid JSON. Fix or clear {QUEUE_PATH} and retry."
        ) from exc

    return Queue.model_validate(data)


def save_queue(queue: Queue) -> None:
    QUEUE_PATH.write_text(queue.model_dump_json(indent=2), encoding="utf-8")


def update_entry(entry: QueueEntry, *, status: str | None = None, error: str | None = None) -> None:
    if status:
        entry.status = status  # type: ignore[assignment]
    entry.last_updated = datetime.now()
    entry.last_error = error


def fetch_latest_streams(config: PipelineConfig) -> list[dict[str, Any]]:
    headers = {"X-APIKEY": config.holodex_api_key}
    params = {
        "channel_id": config.channel_id,
        "type": "stream",
        "limit": config.stream_limit,
    }
    log(
        f"Fetching latest streams from Holodex (channel={config.channel_id}, limit={config.stream_limit})"
    )
    response = requests.get(HOLODEX_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    videos = response.json()
    log(f"Holodex returned {len(videos)} stream(s)")
    return videos


def upsert_new_streams(queue: Queue, videos: list[dict[str, Any]]) -> None:
    added = 0
    updated = 0
    for v in videos:
        video_id = v["id"]
        existing = queue.get_by_id(video_id)
        if existing:
            incoming_title = v.get("title")
            incoming_channel = (v.get("channel") or {}).get("name")
            incoming_date = v.get("available_at")

            changed = False
            if incoming_title and incoming_title != existing.title:
                existing.title = incoming_title
                changed = True
            if incoming_channel and incoming_channel != existing.channel:
                existing.channel = incoming_channel
                changed = True
            if incoming_date and incoming_date != existing.upload_date:
                existing.upload_date = incoming_date
                changed = True

            if changed:
                existing.last_updated = datetime.now()
                updated += 1
            continue

        queue.entries.append(
            QueueEntry(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                status="pending_download",
                title=v.get("title"),
                channel=(v.get("channel") or {}).get("name"),
                upload_date=v.get("available_at"),
            )
        )
        added += 1
    log(f"Queue upsert complete: {added} new stream(s) added, {updated} existing stream(s) refreshed")


def run_cmd(cmd: list[str], *, stream_output: bool = False) -> str:
    if stream_output:
        result = subprocess.run(cmd, check=True, text=True)
        return "" if result.stdout is None else result.stdout

    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def run_yt_dlp_download(url: str, out_template: str, cookies_browser: str) -> None:
    """Download with yt-dlp using a few safe fallback combinations."""
    attempts = [
        ["yt-dlp", "--cookies-from-browser", cookies_browser, "-o", out_template, url],
        [
            "yt-dlp",
            "--cookies-from-browser",
            cookies_browser,
            "-f",
            "bv*+ba/b",
            "-o",
            out_template,
            url,
        ],
        ["yt-dlp", "-o", out_template, url],
        ["yt-dlp", "-f", "bv*+ba/b", "-o", out_template, url],
    ]

    last_error: Exception | None = None
    for index, cmd in enumerate(attempts, start=1):
        try:
            log(f"yt-dlp download attempt {index}/{len(attempts)}")
            run_cmd(cmd, stream_output=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"yt-dlp attempt {index} failed: {exc}")

    if last_error:
        raise RuntimeError(f"yt-dlp failed after {len(attempts)} attempts: {last_error}")
    raise RuntimeError("yt-dlp failed without a captured error")


def download_video(entry: QueueEntry, config: PipelineConfig) -> None:
    out_template = str(RAW_DIR / f"{entry.video_id}.%(ext)s")
    log(f"[{entry.video_id}] Downloading with yt-dlp: {entry.url}")
    run_yt_dlp_download(entry.url, out_template, config.cookies_from_browser)

    candidates = sorted(RAW_DIR.glob(f"{entry.video_id}.*"))
    if not candidates:
        raise RuntimeError(f"yt-dlp completed but no file found for {entry.video_id}")

    raw_file = candidates[0]
    entry.raw_file = str(raw_file)
    log(f"[{entry.video_id}] Download complete: {raw_file}")

    try:
        info = fetch_video_metadata(entry.url, config.cookies_from_browser, prefer_cookies=used_cookies)
        entry.title = info.get("title") or entry.title
        entry.channel = info.get("channel") or entry.channel
        entry.upload_date = info.get("upload_date") or entry.upload_date
    except Exception as exc:  # noqa: BLE001
        log(f"[{entry.video_id}] Metadata fetch failed; continuing with existing metadata: {exc}")


def probe_duration_seconds(raw_file: Path) -> float:
    output = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(raw_file),
        ]
    )
    return float(output)


def to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def sec_to_timestamp(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def ts_for_filename(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    return "-".join(p.zfill(2) for p in parts)


def validate_and_normalize_segments(entry: QueueEntry) -> list[tuple[int, int, str, str]]:
    if not entry.raw_file or not entry.video_duration:
        raise RuntimeError("Cannot validate timestamps without raw file and duration")
    if not entry.segments:
        raise RuntimeError("No timestamp segments found")

    validated_segments: list[TimestampSegment] = []
    for seg in entry.segments:
        if isinstance(seg, TimestampSegment):
            validated_segments.append(seg)
        else:
            validated_segments.append(TimestampSegment.model_validate(seg))

    normalized: list[tuple[int, int, str, str]] = []
    for seg in validated_segments:
        start_sec = to_seconds(seg.start)
        end_sec = to_seconds(seg.end) if seg.end else int(entry.video_duration)
        if start_sec >= int(entry.video_duration):
            raise ValueError(f"Segment start out of range: {seg.start}")
        if end_sec > int(entry.video_duration):
            raise ValueError(f"Segment end out of range: {seg.end}")
        if start_sec >= end_sec:
            raise ValueError(f"Invalid segment bounds: {seg.start} - {seg.end}")
        normalized.append((start_sec, end_sec, sec_to_timestamp(start_sec), sec_to_timestamp(end_sec)))

    normalized.sort(key=lambda x: x[0])
    for i in range(1, len(normalized)):
        prev = normalized[i - 1]
        curr = normalized[i]
        if curr[0] < prev[1]:
            raise ValueError("Timestamp segments overlap")

    return normalized


def trim_segments(entry: QueueEntry, normalized: list[tuple[int, int, str, str]]) -> list[str]:
    if not entry.raw_file:
        raise RuntimeError("Missing raw file path")
    raw_file = Path(entry.raw_file)
    outputs: list[str] = []

    for start_sec, end_sec, start_ts, end_ts in normalized:
        mp3_name = f"{entry.video_id}_{ts_for_filename(start_ts)}__{ts_for_filename(end_ts)}.mp3"
        out_path = SEGMENTS_DIR / mp3_name
        if not out_path.exists():
            log(f"[{entry.video_id}] Trimming segment {start_ts} -> {end_ts}")
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw_file),
                    "-ss",
                    str(start_sec),
                    "-to",
                    str(end_sec),
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    str(out_path),
                ],
                stream_output=True,
            )
        outputs.append(str(out_path))

    return outputs




def format_upload_date(upload_date: str | None) -> str:
    if not upload_date:
        return "unknown"

    if len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    if "T" in upload_date:
        return upload_date.split("T", maxsplit=1)[0]

    return upload_date


def build_srt_metadata_lines(entry: QueueEntry) -> list[str]:
    title = entry.title or entry.video_id
    upload_date = format_upload_date(entry.upload_date)
    return [f"# Title: {title}", f"# Date: {upload_date}"]


def write_metadata(entry: QueueEntry) -> Path:
    out_path = TRANSCRIPTS_DIR / f"{entry.video_id}_metadata.json"
    payload = {
        "video_id": entry.video_id,
        "url": entry.url,
        "title": entry.title,
        "channel": entry.channel,
        "upload_date": entry.upload_date,
        "video_duration": entry.video_duration,
        "segments": [s.model_dump() for s in entry.segments],
        "generated_at": now_iso(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def run_asr(entry: QueueEntry, asr: ASRCore, config: PipelineConfig) -> list[str]:
    transcript_files: list[str] = []
    for segment in entry.mp3_segments:
        segment_path = Path(segment)
        log(f"[{entry.video_id}] Running ASR on {segment_path.name}")
        outputs = transcribe_to_files(
            asr=asr,
            file_path=segment_path,
            output_dir=TRANSCRIPTS_DIR,
            language=config.asr_language,
            formats=config.asr_formats,
            base_name=segment_path.stem,
            srt_header_lines=build_srt_metadata_lines(entry),
        )
        transcript_files.extend(str(p) for p in outputs)

    metadata_path = write_metadata(entry)
    transcript_files.append(str(metadata_path))
    return transcript_files


def verify_transcripts(entry: QueueEntry) -> bool:
    return bool(entry.transcript_files) and all(Path(p).exists() for p in entry.transcript_files)


def cleanup_files(entry: QueueEntry) -> None:
    for seg in entry.mp3_segments:
        seg_path = Path(seg)
        if seg_path.exists():
            seg_path.unlink()

    if entry.raw_file:
        raw = Path(entry.raw_file)
        if raw.exists():
            dest = TRASH_DIR / raw.name
            if dest.exists():
                dest = TRASH_DIR / f"{raw.stem}_{int(datetime.now().timestamp())}{raw.suffix}"
            shutil.move(str(raw), str(dest))
            entry.raw_file = str(dest)
            log(f"[{entry.video_id}] Raw moved to trash: {dest}")


def process_entry(entry: QueueEntry, asr: ASRCore, config: PipelineConfig) -> None:
    log(f"[{entry.video_id}] Processing entry with status={entry.status}")

    if entry.status == "cleaned":
        log(f"[{entry.video_id}] Already cleaned; skipping")
        return

    if entry.status == "waiting_timestamps" and entry.segments:
        log(f"[{entry.video_id}] Found manual timestamps; continuing pipeline")
        update_entry(entry, status="download_finished")

    if entry.status == "pending_download":
        download_video(entry, config)
        update_entry(entry, status="download_finished")
        log(f"[{entry.video_id}] Status -> download_finished")

    if entry.status == "download_finished":
        if not entry.raw_file:
            raise RuntimeError("download_finished but raw_file is missing")
        if entry.video_duration is None:
            entry.video_duration = probe_duration_seconds(Path(entry.raw_file))
            log(f"[{entry.video_id}] Duration detected: {entry.video_duration:.2f}s")

        if not entry.segments:
            update_entry(entry, status="waiting_timestamps")
            log(f"[{entry.video_id}] Waiting for manual timestamps")
            return

        normalized = validate_and_normalize_segments(entry)
        entry.mp3_segments = trim_segments(entry, normalized)
        update_entry(entry, status="trim_finished")
        log(f"[{entry.video_id}] Status -> trim_finished ({len(entry.mp3_segments)} segment(s))")

    if entry.status == "trim_finished":
        if not entry.mp3_segments:
            raise RuntimeError("trim_finished but no segments were recorded")
        entry.transcript_files = run_asr(entry, asr, config)
        update_entry(entry, status="asr_finished")
        log(f"[{entry.video_id}] Status -> asr_finished")

    if entry.status == "asr_finished":
        if not verify_transcripts(entry):
            raise RuntimeError("Transcript verification failed")
        cleanup_files(entry)
        update_entry(entry, status="cleaned")
        log(f"[{entry.video_id}] Status -> cleaned")


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        raise RuntimeError("pipeline.lock exists; another run may be in progress")
    LOCK_PATH.write_text(now_iso(), encoding="utf-8")


def release_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def main() -> int:
    config = load_config()
    log(
        "Pipeline starting "
        f"(channel={config.channel_id}, limit={config.stream_limit}, "
        f"asr_language={config.asr_language}, asr_formats={','.join(config.asr_formats)}, "
        f"cookies_from_browser={config.cookies_from_browser})"
    )
    ensure_dirs()
    acquire_lock()
    try:
        queue = load_queue()
        latest = fetch_latest_streams(config)
        upsert_new_streams(queue, latest)

        asr = ASRCore(model_name="base", compute_type="float16")
        for entry in queue.entries:
            try:
                process_entry(entry, asr, config)
            except Exception as exc:  # noqa: BLE001
                entry.retry_count += 1
                update_entry(entry, status="error", error=str(exc))
                log(f"[{entry.video_id}] ERROR: {exc}")

        save_queue(queue)
        log("Pipeline finished")
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
