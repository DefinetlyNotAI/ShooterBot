"""Simple benchmarking tool for inference performance."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def positive_int(value: str) -> int:
    result = int(value)

    if result <= 0:
        raise argparse.ArgumentTypeError("frames must be > 0")

    return result


def print_progress(
        current: int,
        total: int,
        ms: float,
        width: int = 30,
) -> None:
    if total <= 0:
        return

    progress: float = current / total
    filled: int = int(width * progress)

    bar = "█" * filled + "░" * (width - filled)
    fps: float = 1000.0 / ms if ms > 0 else 0.0

    print(
        f"\r[{bar}] {progress * 100.0:6.2f}% "
        f"{current}/{total} | {ms:.1f}ms | {fps:.1f} FPS",
        end="",
        flush=True,
    )


def benchmark_frame(dm: DetectorManager, frame: Any) -> float:
    """
    Run one inference pass with CUDA synchronization.
    Ensures GPU timing is accurate.
    """
    try:
        # noinspection PyUnusedImports,PyPackageRequirements
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = dm.predict(frame)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

    except ImportError:
        start = time.perf_counter()

        _ = dm.predict(frame)

        end = time.perf_counter()

    return float((end - start) * 1000.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )

    parser.add_argument(
        "--frames",
        type=positive_int,
        default=200,
        help="Frames count to benchmark",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Benchmark a generated frame instead of opening a camera.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    cfg = load_config(str(config_path))

    print("Loading models...")

    load_start = time.perf_counter()

    dm = DetectorManager(cfg)

    load_time = time.perf_counter() - load_start

    print(f"\nModel loading time: {load_time:.2f}s")

    print("Device:", dm.device)

    print("Detectors:")

    for name, meta in dm.detectors.items():
        obj = meta.get("obj")

        name_str = name if isinstance(name, str) else "unknown"

        raw_type = meta.get("type")
        detector_type = (
            raw_type if isinstance(raw_type, str) else "unknown"
        )

        raw_priority = meta.get("priority")
        priority_str = (
            str(raw_priority)
            if isinstance(raw_priority, (int, float, str))
            else "unknown"
        )

        raw_model_name = (
            getattr(obj, "model_name", None)
            if obj is not None
            else None
        )
        model_name = (
            raw_model_name
            if isinstance(raw_model_name, str)
            else "unknown"
        )

        print(
            f"  {name_str}: "
            f"type={detector_type} "
            f"priority={priority_str} "
            f"model={model_name}"
        )

    cap = None
    synthetic_frame = None
    if args.synthetic:
        synthetic_frame = np.full(
            (int(cfg.camera.height), int(cfg.camera.width), 3),
            120,
            dtype=np.uint8,
        )
        print("Using synthetic frame")
    else:
        cap = cv2.VideoCapture(cfg.camera.sources[0])
        if not cap.isOpened():
            cap.release()
            cap = None
            synthetic_frame = np.full(
                (int(cfg.camera.height), int(cfg.camera.width), 3),
                120,
                dtype=np.uint8,
            )
            print("Camera unavailable; using synthetic frame")

    times: list[float] = []

    # -------------------------
    # CUDA warmup
    # -------------------------

    try:
        import torch

        if torch.cuda.is_available():

            print("\nWarming up CUDA...")

            for i in range(1, 11):

                if cap is None:
                    frame = synthetic_frame
                    ret = frame is not None
                else:
                    ret, frame = cap.read()

                if not ret or frame is None:
                    raise RuntimeError("Failed to capture warmup frame")

                ms = benchmark_frame(dm, frame)

                print_progress(i, 10, ms)

            print()

        else:
            print("\nCUDA unavailable")

    except Exception as e:
        print(f"\nWarmup skipped: {e}")

    print("\nWarmup complete")

    # -------------------------
    # Benchmark
    # -------------------------

    print("\nStarting benchmark...")
    print("Press Ctrl+C to stop\n")

    frame_count: int = 0

    try:

        while frame_count < args.frames:

            if cap is None:
                frame = synthetic_frame
                ret = frame is not None
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                break

            ms = benchmark_frame(dm, frame)

            times.append(ms)

            frame_count += 1

            print_progress(frame_count, args.frames, ms)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted")
    finally:
        if cap is not None:
            cap.release()

    print()

    if not times:
        print("No samples collected")
        return

    print(f"\nFrames: {len(times)}")

    print(f"Avg inference ms: {statistics.mean(times):.2f}")

    print(f"Min inference ms: {min(times):.2f}")

    print(f"Max inference ms: {max(times):.2f}")

    print(f"P50: {statistics.median(times):.2f}")

    if len(times) >= 20:

        p95 = statistics.quantiles(times, n=20)[18]

        print(f"P95: {p95:.2f}")

    else:

        print("P95: N/A")

    avg_fps = 1000.0 / statistics.mean(times)

    print(f"Approx FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
