"""Video helpers: read metadata and create a video saver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2


@dataclass
class VideoMetadata:
    """Basic information about a video stream."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def size(self) -> Tuple[int, int]:
        """Return frame size as (width, height)."""
        return self.width, self.height


def open_video_capture(path: str | Path) -> tuple[cv2.VideoCapture, VideoMetadata]:
    """Open a video file and gather metadata.

    Args:
        path: Path to the input video.

    Returns:
        Pair of open VideoCapture and its metadata.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the video cannot be opened or has no dimensions.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video dimensions for: {video_path}")

    metadata = VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)
    return capture, metadata


def create_video_writer(output_path: str | Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    """Create a VideoWriter for saving annotated frames.

    Args:
        output_path: Path to the output video file.
        metadata: Metadata of the input video.

    Returns:
        Configured and opened VideoWriter.

    Raises:
        ValueError: If the writer cannot be created or opened.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = metadata.fps if metadata.fps > 0 else 30.0
    writer = cv2.VideoWriter(str(output), fourcc, fps, metadata.size)

    if not writer.isOpened():
        raise ValueError(f"Failed to create VideoWriter for: {output}")

    return writer
