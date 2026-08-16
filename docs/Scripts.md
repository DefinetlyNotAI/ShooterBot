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
| Validate Configuration | `src.scripts.check_config` | Parses the active YAML configuration without opening models or cameras. |
| Check External Data Sources | `src.scripts.check_data_sources` | Checks source reachability plus HTTPS and redirect trust boundaries, and reports SHA-256 pin coverage for model downloads. |
| Check Package Updates | `src.scripts.check_package_updates` | Reports available project dependency updates without installing anything. |
| Runtime Report | `src.scripts.runtime_report` | Reports Python, package, config, and model readiness without loading models. |
| Inspect Detectors | `src.scripts.check_detectors` | Loads configured detectors and prints metadata. |
| Print Loaded Models | `src.scripts.print_models` | Lists configured models after loading them. |
| Run Simulation Inference | `src.scripts.sim_infer` | Runs the configured detector stack against a synthetic frame. |
| Save Annotated Simulation | `src.scripts.run_sim_save` | Saves an annotated synthetic image. |
| Save Annotated Debug Frame | `src.scripts.save_annotated_frame` | Saves an annotated simulation image using `configs/default.yaml`. |
| Run One-Frame Detector Test | `src.scripts.test_frame` | Uses one webcam frame when available, otherwise a synthetic frame. |
| Camera Inference Smoke Test | `src.scripts.test_inference` | Runs installed models against one webcam frame; uses a synthetic frame if no webcam is available. |
| Benchmark Configured Inference | `src.scripts.benchmark` | Measures configured detector throughput. Use `--synthetic` when running directly without a camera. |
| CUDA YOLO Benchmark | `src.scripts.test_yolo` | Benchmarks every local `models/*.pt` file with CUDA and a webcam. Disabled without CUDA or local models. |
| Generate GitHub Support Report | `src.scripts.support_report` | Writes a sensitive diagnostic JSON report under `logs/`; review and redact it before upload. |

**Run All Ready Scripts** runs each available diagnostic in sequence, skips
disabled entries, and shows a final success/failure summary. It always skips
the support report because that report is intended to be generated manually
when filing an issue.

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
`nirt_shooterbot.library` logger so they use the same console and file format.

Set `NIRT_CAPTURE_NATIVE_STDERR=0` before starting the application only if a
native debugging tool requires direct ownership of stderr.

## Issue reports

Use **Generate GitHub Support Report** before opening an issue when practical.
It gathers system, package, GPU, repository, configuration, and recent-log
context, so it may contain sensitive information. Inspect and redact the file
before attaching it. GitHub issue forms and the triage workflow label bugs by
core runtime, installer/configuration, or scripts, and post a compact summary.
