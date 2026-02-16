# vtuber-asr-automation

Automation pipeline for downloading, trimming, and transcribing recent Koseki Bijou streams. You can use another channel ID and choose another chuuba instead.

This project has two modes:
- **Automated pipeline** (`run_pipeline.py`) for daily processing.
- **Manual tools** (`gui_whisper.py`, `edit_queue_gui.py`) for transcription and timestamp editing.

---

## 1) What `run_pipeline.py` does

`run_pipeline.py` is an idempotent state-machine pipeline driven by `queue.json`.

### High-level flow
1. Fetch latest streams from Holodex.
2. Upsert queue entries (new entries become `pending_download`; existing metadata can be refreshed).
3. Download pending videos with yt-dlp (with fallback strategies).
4. Probe duration with ffprobe.
5. If no timestamp segments exist, mark as `waiting_timestamps` and stop for that entry.
6. Once segments exist, validate timestamps using Pydantic models (`validation_model.py`).
7. Trim source MP4 into MP3 segments with ffmpeg.
8. Run ASR on each segment via shared `asr_tool/core.py` logic.
9. Write transcript files into `output/transcripts/`.
10. Verify transcript outputs exist.
11. Cleanup: delete temporary MP3 segments and move MP4 to trash.

### Queue statuses
- `pending_download`
- `download_finished`
- `waiting_timestamps`
- `trim_finished`
- `asr_finished`
- `cleaned`
- `error`

### Safety / idempotency
- Uses a lock file: `pipeline.lock`.
- Skips completed states safely.
- Records failures in `last_error` and continues with other entries.
- Never hard-deletes source MP4 before transcript verification.

### Run it
```bash
python run_pipeline.py
```

For daily automation, run it via OS scheduler (e.g., Windows Task Scheduler at 4 AM).

---

## 2) Manual ASR GUI (`gui_whisper.py`)

Use this when you want one-off manual transcription outside the full pipeline.

### Run
```bash
python gui_whisper.py
```

### Behavior
- Uses faster-whisper (`base` model) through `asr_tool/gui.py`.
- Accepts uploaded MP3/MP4 in the Gradio UI.
- Outputs **`.srt` by default**.
- Saves transcripts to:
  - `output/transcripts/`

---

## 3) Timestamp editor GUI (`edit_queue_gui.py`)

Use this for the manual step when pipeline entries are waiting for timestamps.

### Run
```bash
python edit_queue_gui.py
```

### UI workflow
1. Select a queue item from dropdown (`{video_id} - {title}`) where status is `waiting_timestamps`.
2. Paste segments, one per line:
   - `HH:MM:SS - HH:MM:SS`
   - `HH:MM:SS - end`
3. Click **Save**.

### What Save does
- Parses lines into `TimestampSegment` objects.
- Validates through Pydantic queue model logic.
- If valid:
  - updates the selected entry in `queue.json`
  - changes status to `download_finished`
- If invalid:
  - shows validation error in status box
  - does not save

---

## 4) Where files go

Paths are relative to project root.

- Raw downloads (MP4): `downloads/raw/`
- Temporary trimmed audio segments (MP3): `downloads/segments/`
- Final transcripts (SRT/TXT + metadata): `output/transcripts/`
- Soft-deleted source MP4s: `downloads/trash/`
- Pipeline lock file: `pipeline.lock`
- Queue/state file: `queue.json`
- Env file (local secrets): `.env`

---

## 5) Project structure (important files)

- `run_pipeline.py` - end-to-end automation state machine
- `validation_model.py` - Pydantic models for queue + timestamp validation
- `asr_tool/core.py` - shared ASR/transcript writing logic (no UI)
- `asr_tool/gui.py` - Gradio manual transcription UI logic
- `asr_tool/cli.py` - headless CLI entry for ASR batch/single runs
- `gui_whisper.py` - thin launcher for manual ASR GUI
- `edit_queue_gui.py` - GUI for editing timestamps in waiting queue items
- `ASR_workflow.md` - design notes / workflow rationale

---

## 6) Environment & dependencies

### Install
```bash
pip install -r requirements.txt
```

### Required tools on PATH
- `yt-dlp`
- `ffmpeg`
- `ffprobe`

### `.env`
Use `env.example` as a template. Key values include:
- `HOLODEX_API_KEY`
- optional overrides like channel id, stream limit, ASR output config, browser cookie source.

---

## 7) Typical daily usage

1. Runs `python run_pipeline.py`, essentially with a scheduler. 
2. New streams download and then pause at `waiting_timestamps`.
3. You run `python edit_queue_gui.py` and fill segment timestamps.
4. Next scheduled run continues trim -> ASR -> export -> cleanup.

