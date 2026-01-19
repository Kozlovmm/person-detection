"""Script for detecting people in videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .detector import Detector
from .draw import draw_detections
from .video_io import create_video_writer, open_video_capture

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class VideoMetrics:
    """Per-video metrics summary."""

    input_path: str
    output_path: str
    frames: int
    total_detections: int
    avg_detections_per_frame: float
    processing_fps: float
    duration_seconds: float
    model: str
    conf: float
    imgsz: int | Tuple[int, int] | None
    device: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: Optional[int | Tuple[int, int]] = None,
    device: str = "auto",
) -> VideoMetrics:
    """Run detection on a video and save annotated frames."""
    capture, metadata = open_video_capture(input_path)
    effective_imgsz: int | Tuple[int, int] | None
    if imgsz is None:
        # Default to the input video size to avoid resizing.
        effective_imgsz = (metadata.height, metadata.width)
    else:
        effective_imgsz = imgsz

    if effective_imgsz is not None:
        effective_imgsz = _adjust_imgsz_to_stride(effective_imgsz)

    effective_device = _normalize_device(device)
    detector = Detector(model_path=model_path, conf_threshold=conf, imgsz=effective_imgsz, device=effective_device)
    writer = create_video_writer(output_path, metadata)

    frame_index = 0
    total_detections = 0
    start = time.perf_counter()
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            detections = detector.detect(frame)
            total_detections += len(detections)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)
            frame_index += 1
            _print_progress(frame_index, metadata.frame_count)
    finally:
        capture.release()
        writer.release()
    duration = time.perf_counter() - start
    processing_fps = frame_index / duration if duration > 0 else 0.0
    avg_det = total_detections / frame_index if frame_index else 0.0
    print(
        f"\nDone: {output_path} "
        f"(frames: {frame_index}, fps: {processing_fps:.2f}, detections: {total_detections})"
    )

    return VideoMetrics(
        input_path=str(input_path),
        output_path=str(output_path),
        frames=frame_index,
        total_detections=total_detections,
        avg_detections_per_frame=avg_det,
        processing_fps=processing_fps,
        duration_seconds=duration,
        model=model_path,
        conf=conf,
        imgsz=effective_imgsz,
        device=effective_device,
    )


def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_path: str,
    conf: float,
    imgsz: Optional[int | Tuple[int, int]] = None,
    device: str = "auto",
) -> list[VideoMetrics]:
    """Process every video file in input_dir and save results to output_dir."""
    videos = _list_videos(input_dir)
    if not videos:
        raise FileNotFoundError(
            f"No videos with extensions {sorted(VIDEO_EXTENSIONS)} found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    collected: list[VideoMetrics] = []
    for video_path in videos:
        target_path = output_dir / video_path.name
        print(f"\nProcessing {video_path} -> {target_path}")
        metrics = process_video(
            input_path=video_path,
            output_path=target_path,
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )
        collected.append(metrics)
    return collected


def _list_videos(directory: Path) -> list[Path]:
    """Return a list of video files in the top level of a directory."""
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name,
    )


def _print_progress(current: int, total: int) -> None:
    """Print a simple progress bar to the console."""
    if total <= 0:
        print(f"Frames processed: {current}", end="\r")
        return
    bar_length = 30
    ratio = min(current / total, 1.0)
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    print(f"[{bar}] {percent:5.1f}% ({current}/{total})", end="\r")


def _adjust_imgsz_to_stride(imgsz: int | Tuple[int, int], stride: int = 32) -> int | Tuple[int, int]:
    """Snap imgsz to a stride multiple to avoid YOLO warnings."""
    if isinstance(imgsz, int):
        return int(math.ceil(imgsz / stride) * stride)
    height, width = imgsz
    return (
        int(math.ceil(height / stride) * stride),
        int(math.ceil(width / stride) * stride),
    )


def write_metrics_files(metrics: list[VideoMetrics], output_dir: Path) -> None:
    """Save metrics to JSON and CSV in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    data = _round_float_fields([asdict(m) for m in metrics], ndigits=2)
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    if data:
        headers = list(data[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)


def _round_float_fields(data: list[dict], ndigits: int) -> list[dict]:
    """Round float values in metrics dictionaries."""
    rounded: list[dict] = []
    for item in data:
        new_item = {}
        for key, value in item.items():
            if isinstance(value, float):
                new_item[key] = round(value, ndigits)
            else:
                new_item[key] = value
        rounded.append(new_item)
    return rounded


def _normalize_device(device: str) -> str:
    """Choose a safe device string based on availability."""
    if device.lower() != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """Entry point when running from a terminal."""
    args = parse_args()
    input_path = Path("assets")
    output_path = Path("outputs")

    if input_path.is_dir():
        output_dir = output_path if output_path.suffix == "" else output_path.parent
        metrics = process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        write_metrics_files(metrics, output_dir)
    else:
        final_output = output_path
        if output_path.is_dir() or output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            final_output = output_path / input_path.name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = process_video(
            input_path=input_path,
            output_path=final_output,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        write_metrics_files([metrics], final_output.parent)


if __name__ == "__main__":
    main()
