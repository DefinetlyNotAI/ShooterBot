# Runtime Reference

This document describes runtime behavior and operational choices.

## Configuration workflow

The application requires `configs/default.yaml`. If it is missing, startup stops with an installer-first message and does not attempt to create a partial configuration. Run `python -m src.cli.installer` and choose Install before starting the application.

Runtime options are configured in YAML. The main controls are:

- `tracking.track_only` selects the classes that receive persistent tracks.
- `inference.ignored_classes` removes detector classes before tracking.
- Inference intervals control how frequently individual detectors run.
- `serial.simulation` selects simulated or hardware serial behavior.
- `serial.advanced_datapackets` enables verbose detection metadata.

The shipped tracking defaults include `face`, `sports ball`, and `cell phone`. Multiple detections share the same tracker and target queue, so IDs remain independent while the queue rotates between targets.

## Camera modes

The camera layer supports:

- A physical camera index such as `0`.
- A video file path.
- A synthetic simulation feed.

When a physical camera cannot be opened, the configured simulation fallback can keep the UI and control loop available.

## Serial modes

Serial is disabled by default. Hardware operation requires the configured port, baud rate, and non-simulation mode. During development, simulation mode allows UI and control-loop validation without hardware.

The default telemetry packet is intentionally small:

```json
{"x":0.52,"y":0.41}
```

The target queue cycles shot targets by default. Set `tracking.cycle_remember: false` to permanently exclude a target after it is shot.

## Diagnostics

Use the logging settings in `configs/default.yaml` to control level, color, verbose source context, and the runtime log file. Runtime logs are stored under `logs/` by default. The runtime setup check reports missing required packages, configured model files, and enabled optional features that cannot load.

For standalone checks, run `python -m src.cli.scripts`. Its capability-aware
menu disables scripts that cannot run with the current packages, configuration,
models, or CUDA support. Script-generated images are stored under `files/` in
the project root; models must stay under the root `models/` directory.
