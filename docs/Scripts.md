# Scripts Console

The Scripts Console is an interactive launcher for the diagnostic and utility
programs in `src/scripts/`.

Run it from the project root after activating the virtual environment:

```powershell
python -m src.cli.scripts
```

The console uses the same terminal style as the installer. It checks each
script before it can be selected and disables it when a required package,
configuration file, model file, support file, or CUDA capability is missing.
Choose **Health Check** to review all script availability without loading a
model, opening a camera, or creating output.

## Available scripts

| Console action | Module | Purpose |
| --- | --- | --- |
| Runtime Report | `src.scripts.runtime_report` | Reports Python, package, config, and model readiness without loading models. |
| Inspect Detectors | `src.scripts.check_detectors` | Loads configured detectors and prints metadata. |
| Print Loaded Models | `src.scripts.print_models` | Lists configured models after loading them. |
| Run Simulation Inference | `src.scripts.sim_infer` | Runs the configured detector stack against a synthetic frame. |
| Save Simulation Output | `src.scripts.run_sim_save` | Saves an annotated synthetic image. |
| Save Annotated Frame | `src.scripts.save_annotated_frame` | Saves an annotated simulation image using `configs/default.yaml`. |
| Test One Frame | `src.scripts.test_frame` | Uses one webcam frame when available, otherwise a synthetic frame. |
| Camera Inference Smoke Test | `src.scripts.test_inference` | Runs installed models against one webcam frame; uses a synthetic frame if no webcam is available. |
| Benchmark Configured Inference | `src.scripts.benchmark` | Measures configured detector throughput. Use `--synthetic` when running directly without a camera. |
| CUDA YOLO Benchmark | `src.scripts.test_yolo` | Measures YOLO throughput with CUDA and a webcam. Disabled without CUDA or `models/yolov8n.pt`. |

Scripts can also be run directly as modules, for example:

```powershell
python -m src.scripts.runtime_report
python -m src.scripts.benchmark --synthetic --frames 100
```

## Output and model policy

Scripts always resolve project paths from their own source location; they do
not write beside the script or into the caller's current working directory.

- Images and other script output are written under the root `files/` directory.
- Logs are written under the root `logs/` directory.
- Models are read from and downloaded only to the root `models/` directory.
- Inference cache data remains under the root `.cache/` directory.

Those generated directories are ignored by Git. Do not put models under
`src/scripts/models/`; that path is intentionally unsupported.

## Logging

The runtime installs the project formatter before importing heavyweight
libraries. After `configs/default.yaml` loads, logging is reconfigured using
its level, file, color, and verbose settings. Native diagnostics written to
stderr by libraries such as TensorFlow Lite are routed through the
`realtime_cv.library` logger so they use the same console and file format.

Set `NIRT_CAPTURE_NATIVE_STDERR=0` before starting the application only if a
native debugging tool requires direct ownership of stderr.
