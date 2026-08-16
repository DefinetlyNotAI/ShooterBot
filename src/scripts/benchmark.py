"""Simple benchmarking tool for inference performance."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("frames must be > 0")
    return value


def print_progress(current: int, total: int, ms: float, width: int = 30):
    if total <= 0:
        return

    progress = current / total
    filled = int(width * progress)

    bar = "█" * filled + "░" * (width - filled)
    fps = 1000 / ms if ms > 0 else 0

    print(
        f"\r[{bar}] {progress * 100:6.2f}% "
        f"{current}/{total} | {ms:.1f}ms | {fps:.1f} FPS",
        end="",
        flush=True,
    )


def benchmark_frame(dm, frame):
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

    return (end - start) * 1000.0


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

        print(
            f"  {name}: "
            f"type={meta.get('type')} "
            f"priority={meta.get('priority')} "
            f"model={getattr(obj, 'model_name', 'unknown')}"
        )

    cap = cv2.VideoCapture(cfg.camera.sources[0])

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    times = []

    # -------------------------
    # CUDA warmup
    # -------------------------

    try:
        import torch

        if torch.cuda.is_available():

            print("\nWarming up CUDA...")

            for i in range(1, 11):

                ret, frame = cap.read()

                if not ret:
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

    frame_count = 0

    try:

        while frame_count < args.frames:

            ret, frame = cap.read()

            if not ret:
                break

            ms = benchmark_frame(dm, frame)

            times.append(ms)

            frame_count += 1

            print_progress(frame_count, args.frames, ms)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted")
    finally:
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
