# NIRT ShooterBot

---

Production-ready, modular real-time computer vision framework for robotics and automation.

## Requirements

- Python 3.11+
- CUDA-enabled GPU recommended (optional for performance)

## Setup and start

1. Create a virtual environment:

    ```bash
    python -m venv .venv
    ```

2. Activate it and run the installer:

     ```bash
     .\.venv\Scripts\Activate.ps1
     python -m src.cli.installer
     ```
   
   The installer opens an action menu for Health Check, Install, Clean Install, Basic Config, and Advanced Config. It detects NVIDIA hardware and offers CUDA-enabled PyTorch. Select Health Check to inspect the current setup without changing anything.

3. Edit `configs/default.yaml` as needed.

4. Start the application:

    ```bash
    python -m src --config configs\default.yaml
    ```

## Features

- YOLO (Ultralytics) primary detector with automatic model download
- Optional face detector (Custom YOLO model)
- Tracking with persistent IDs, velocity, and trajectories
- Generic Arduino serial communication layer (simulation mode)
- Modular architecture: camera, inference, tracking, visualization, config, logging, serial

See [docs/installer.md](docs/Installer.md) for setup, dependency, model, and troubleshooting details.
Use [docs/Scripts.md](docs/Scripts.md) for the diagnostic Scripts Console and utility-script reference.

### License

MIT License
