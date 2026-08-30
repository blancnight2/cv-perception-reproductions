"""Benchmark the steady-state YOLO + ByteTrack video pipeline.

The reported E2E latency starts immediately before ``VideoCapture.read`` and
ends after the detection boxes and tracking IDs have been rendered. It includes
video decoding, preprocessing, TensorRT inference, NMS, ByteTrack association,
and rendering. It deliberately excludes optional video-file encoding because
disk and codec throughput are not part of an online perception pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best_fp16_trt112.engine"
DEFAULT_SOURCE = PROJECT_ROOT / "runs" / "detect" / "data" / "demo1.mp4"


def summarize_latency_ms(latencies_ms: list[float]) -> dict[str, float | int]:
    """Return percentile statistics for one non-empty sequence of frame latencies."""
    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")

    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    return {
        "count": int(values.size),
        "min_ms": float(np.min(values)),
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "fps_from_mean": float(1000.0 / mean_ms),
    }


def synchronize_cuda() -> None:
    """Wait for queued CUDA work so wall-clock timings include GPU execution."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_trackers(model: Any) -> None:
    """Clear tracker state while retaining the already-warmed TensorRT predictor."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        tracker.reset()


def track_count(results: list[Any]) -> int:
    """Return the number of rendered detections that were assigned track IDs."""
    boxes = results[0].boxes if results else None
    if boxes is None or boxes.id is None:
        return 0
    return int(len(boxes.id))


def output_directory(value: str | None) -> Path:
    """Use a timestamped benchmark directory unless the caller selects one."""
    if value:
        return Path(value).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "detect" / "benchmarks" / f"e2e_fp16_bytetrack_{timestamp}"


def write_outputs(
    destination: Path,
    records: list[dict[str, float | int]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist per-frame timing records and their machine-readable summary."""
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "per_frame_latency.csv"
    json_path = destination / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return csv_path, json_path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmup and benchmark passes, returning a report suitable for JSON output."""
    import cv2
    from ultralytics import YOLO

    engine_path = Path(args.model).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Video source not found: {source_path}")

    warmup_capture = cv2.VideoCapture(str(source_path))
    ok, warmup_frame = warmup_capture.read()
    warmup_capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode the first frame: {source_path}")

    print(f"Loading engine: {engine_path}")
    warmup_model = YOLO(str(engine_path), task="detect")
    print(f"Warming up {args.warmup} frames (excluded from all metrics) ...")
    for _ in range(args.warmup):
        warmup_model.track(
            warmup_frame,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
    synchronize_cuda()

    # Keep the warm TensorRT predictor, but begin benchmarked track IDs at frame zero.
    reset_trackers(warmup_model)
    model = warmup_model
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    destination = output_directory(args.output_dir)
    video_writer = None
    output_video_path = destination / "tracked.mp4"
    records: list[dict[str, float | int]] = []
    frame_index = 0

    try:
        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break

            e2e_start = time.perf_counter()
            decode_start = e2e_start
            ok, frame = capture.read()
            decode_end = time.perf_counter()
            if not ok:
                break

            synchronize_cuda()
            track_start = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            synchronize_cuda()
            track_end = time.perf_counter()

            render_start = track_end
            rendered = results[0].plot()
            synchronize_cuda()
            render_end = time.perf_counter()

            write_ms = math.nan
            if args.save:
                if video_writer is None:
                    destination.mkdir(parents=True, exist_ok=True)
                    height, width = rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, source_fps, (width, height))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output_video_path}")
                write_start = time.perf_counter()
                video_writer.write(rendered)
                write_ms = (time.perf_counter() - write_start) * 1000.0

            records.append(
                {
                    "frame_index": frame_index,
                    "decode_ms": (decode_end - decode_start) * 1000.0,
                    "detect_nms_track_ms": (track_end - track_start) * 1000.0,
                    "render_ms": (render_end - render_start) * 1000.0,
                    "e2e_ms": (render_end - e2e_start) * 1000.0,
                    "write_ms_excluded": write_ms,
                    "tracked_detections": track_count(results),
                }
            )
            frame_index += 1
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    if not records:
        raise RuntimeError("No frames were benchmarked")

    e2e = summarize_latency_ms([float(record["e2e_ms"]) for record in records])
    decode = summarize_latency_ms([float(record["decode_ms"]) for record in records])
    detect_track = summarize_latency_ms([float(record["detect_nms_track_ms"]) for record in records])
    render = summarize_latency_ms([float(record["render_ms"]) for record in records])
    summary: dict[str, Any] = {
        "benchmark": "YOLO TensorRT + ByteTrack steady-state video pipeline",
        "latency_definition": (
            "VideoCapture.read through YOLO preprocessing, TensorRT inference, NMS, "
            "ByteTrack association, and Results.plot rendering; excludes optional video encoding."
        ),
        "model": str(engine_path),
        "source": str(source_path),
        "tracker": args.tracker,
        "input_size": args.imgsz,
        "warmup_frames_excluded": args.warmup,
        "benchmarked_frames": len(records),
        "save_video": bool(args.save),
        "output_video": str(output_video_path) if args.save else None,
        "end_to_end": e2e,
        "decode": decode,
        "detect_nms_track": detect_track,
        "render": render,
    }
    csv_path, json_path = write_outputs(destination, records, summary)
    summary["per_frame_csv"] = str(csv_path)
    summary["summary_json"] = str(json_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for a reproducible steady-state video benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_ENGINE), help="TensorRT .engine path")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input video path")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML or built-in name")
    parser.add_argument("--imgsz", type=int, default=640, help="Static engine input size")
    parser.add_argument("--device", default="0", help="CUDA device passed to Ultralytics")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Detection NMS IoU threshold")
    parser.add_argument("--warmup", type=int, default=100, help="Excluded warmup frame count")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means benchmark the whole video")
    parser.add_argument("--save", action="store_true", help="Also save tracked.mp4; encoding is timed separately")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV, JSON, and optional video")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print the three interview-relevant headline metrics."""
    args = parse_args(argv)
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")

    summary = run_benchmark(args)
    e2e = summary["end_to_end"]
    assert isinstance(e2e, dict)
    print("\n=== End-to-end latency (decode -> render, video encoding excluded) ===")
    print(f"Frames: {e2e['count']}")
    print(f"mean: {e2e['mean_ms']:.3f} ms | p50: {e2e['p50_ms']:.3f} ms | p95: {e2e['p95_ms']:.3f} ms")
    print(f"p99: {e2e['p99_ms']:.3f} ms | FPS (1 / mean): {e2e['fps_from_mean']:.2f}")
    print(f"CSV: {summary['per_frame_csv']}")
    print(f"JSON: {summary['summary_json']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
