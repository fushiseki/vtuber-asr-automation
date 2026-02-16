from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from datetime import datetime
from pathlib import Path
import re

class TimestampSegment(BaseModel):
    """Single segment with start/end timestamps"""
    start: str = Field(..., description="HH:MM:SS format")
    end: Optional[str] = Field(None, description="HH:MM:SS or null for 'until end'")
    
    @field_validator('start', 'end')
    @classmethod
    def validate_timestamp_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Accept HH:MM:SS or MM:SS
        if not re.match(r'^(\d{1,2}:)?\d{1,2}:\d{2}$', v):
            raise ValueError(f"Invalid timestamp format: {v}. Use HH:MM:SS or MM:SS")
        return v
    
    @model_validator(mode='after')
    def validate_segment(self):
        """Ensure start < end if both present"""
        if self.start and self.end:
            start_sec = self._to_seconds(self.start)
            end_sec = self._to_seconds(self.end)
            if start_sec >= end_sec:
                raise ValueError(f"Start {self.start} must be before end {self.end}")
        return self
    
    @staticmethod
    def _to_seconds(ts: str) -> int:
        parts = list(map(int, ts.split(':')))
        if len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        else:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    
    def to_seconds(self) -> tuple[int, Optional[int]]:
        """Convert to (start_sec, end_sec) for ffmpeg"""
        start_sec = self._to_seconds(self.start)
        end_sec = self._to_seconds(self.end) if self.end else None
        return start_sec, end_sec


class QueueEntry(BaseModel):
    """Single video in the processing queue"""
    
    # Identity
    video_id: str
    url: str
    
    # State machine
    status: Literal[
        "pending_download",
        "download_finished", 
        "waiting_timestamps",
        "trim_finished",
        "asr_finished",
        "cleaned",
        "error"
    ]
    
    # File paths (use strings, convert to Path when needed)
    raw_file: Optional[str] = None  # full mp4 path
    mp3_segments: List[str] = Field(default_factory=list)
    transcript_files: List[str] = Field(default_factory=list)
    
    # Timestamps (user-provided)
    segments: List[TimestampSegment] = Field(default_factory=list)
    video_duration: Optional[float] = None  # seconds, from ffprobe
    
    # Metadata
    title: Optional[str] = None
    channel: Optional[str] = None
    upload_date: Optional[str] = None  # YYYYMMDD from yt-dlp
    
    # Tracking
    discovered_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    last_error: Optional[str] = None
    retry_count: int = 0
    
    @model_validator(mode='after')
    def validate_segments_against_duration(self):
        """Check segments don't exceed video duration"""
        if self.video_duration and self.segments:
            for i, seg in enumerate(self.segments):
                start_sec, end_sec = seg.to_seconds()
                if start_sec >= self.video_duration:
                    raise ValueError(
                        f"Segment {i}: start {seg.start} exceeds video duration "
                        f"{self._format_duration(self.video_duration)}"
                    )
                if end_sec and end_sec > self.video_duration:
                    raise ValueError(
                        f"Segment {i}: end {seg.end} exceeds video duration "
                        f"{self._format_duration(self.video_duration)}"
                    )
        return self
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


class Queue(BaseModel):
    """The entire queue file"""
    entries: List[QueueEntry] = Field(default_factory=list)
    
    def get_by_id(self, video_id: str) -> Optional[QueueEntry]:
        """Find entry by video_id"""
        return next((e for e in self.entries if e.video_id == video_id), None)
    
    def needs_timestamps(self) -> List[QueueEntry]:
        """Return entries waiting for manual timestamps"""
        return [e for e in self.entries if e.status == "waiting_timestamps"]