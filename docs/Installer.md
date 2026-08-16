# Installer and Setup Guide

All environment setup is handled by the standard-library-only `src/cli/installer.py` script. No requirements files are used. Run it without flags; the first screen selects the requested operation.

## Default setup

Use Python 3.11 or newer in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m src.cli.installer
```

The installer refuses to modify system Python, checks that the active virtual environment is writable, creates the default configuration when it is missing, installs only missing packages, detects NVIDIA hardware, and offers only missing optional packages.

The installer does not use command-line mode flags. It asks what to do interactively and provides safe defaults for click-through installs. Detailed diagnostics are always written to `logs/installer.log`.

## Installer actions

The opening menu provides:

- `Health Check` - validates the existing setup without changing anything. It is disabled until a configuration has been created.
- `Install` - creates or preserves configuration and installs missing required, GPU, model, and optional components.
- `Wipe and Install` - performs a Clean Install. It asks for confirmation, removes generated `configs/`, `models/`, `logs/`, `.cache/`, and `files/`, then asks separately whether project-managed pip distributions should be removed.
- `Modify Config (Basic)` - edits the important camera, device, tracking, feature, and simulation settings.
- `Modify Config (Advanced)` - edits all supported scalar settings, including options not shown in Basic mode.

Disabled actions are shown in gray with the reason they are unavailable.

To run diagnostics without installing, repairing, downloading, or modifying project files, select `Health Check` from the opening menu. It exits with code `0` when all checks pass and code `2` when issues are found.

## Configuration

When `configs/default.yaml` does not exist, the installer asks for the camera, resolution, FPS, inference device, face memory, emotion tracking, hand tracking, and simulation fallback. The generated defaults enable face memory, emotion tracking, and hand tracking, use `0.55` confidence for normal detections and `0.35` for faces, and disable simulation fallback. Pressing Enter accepts each default.

When `configs/default.yaml` already exists, Install asks whether it should be modified. Choosing No keeps every existing value and uses the configuration as-is. Choosing Yes opens the Basic settings prompts. Existing configuration files are otherwise preserved. Use Modify Config (Advanced) for the complete setting list.

## Required packages

The installer checks and installs:

- Ultralytics for YOLO inference.
- PyTorch 2.5 or newer for the supported inference/Transformers stack.
- OpenCV with GUI support.
- NumPy, PyYAML, psutil, and pyserial.

Each package is checked by installed distribution name and minimum version before any download begins. Satisfied packages are reported as already installed and are skipped.

## Models

The installer includes a model selection section. It shows a short description and whether each file is already present. Existing models are not downloaded again. Press Enter to accept the defaults, or select model numbers manually.

The catalog includes:

- `models/yolov8n.pt` for lightweight inference and fallback.
- `models/yolo26n-face.pt` for face detection.
- `models/yolov8m.pt` as an optional higher-quality model.
- `models/yolo26n.pt` as an optional lightweight general detector.

The installer downloads publicly available catalog entries using temporary files, verifies that the download is non-empty, and atomically moves completed files into `models/`. `yolo26n.pt` is available from the Ultralytics assets release and can be downloaded by selecting it. Missing local model files are reported by the Health Check and runtime setup check.

If multiple face models are installed, the installer requires one face model to be selected. If multiple general models are installed, it requires their priority order. Choices can be written back to the configuration; otherwise an existing configuration order remains authoritative.

The Models section displays general detection models first and face detection models second, with color-coded recommendation, installation status, and grouping. If the configuration already contains model order or a face-model choice, the installer shows those values and asks whether they should be edited. Choosing No preserves them unchanged.

## Health Check and Repair

Before declaring installation complete, the installer validates configuration syntax, required package versions, PyTorch and CUDA state, configured model files, and enabled optional feature backends. If a check fails, it enters a Repair section, attempts safe package/model repairs, and repeats the complete Health Check. Unresolved issues stop completion with specific recovery guidance.

The installer always writes a complete diagnostic transcript to `logs/installer.log`, including prompts, installer progress, pip commands, pip output, uninstall output, and verification results.

Package installation progress uses a live activity bar while pip resolves or writes files, reports bytes staged whenever available, and switches to a completed byte/artifact total only after staging finishes. This avoids presenting pip's non-terminal `0%` metadata status as a false download percentage.

Emotion tracking installs a compatible `setuptools<81` because the current FER backend imports `pkg_resources`. Hand tracking uses MediaPipe `0.10.21`, which provides the legacy `mediapipe.solutions` API required by this project.

When a package is detected as incompatible or corrupted, repair first stages the replacement artifacts, then removes only the affected distribution(s) with pip, installs the staged replacement, and verifies the result. It never deletes unrelated packages or user files. If pip cannot complete the replacement, the repeated Health Check reports the package and the suggested manual recovery command.

Model downloads have installer-owned byte-based progress bars. Package installation has its own `Installation` section: pip output is intercepted and summarized, dependency downloads show a live installer-owned activity bar plus staged-artifact counts, and local installation shows a real count when pip reports completion. The installer avoids displaying fabricated byte totals. Pip, network, and model-library output are suppressed.

## GPU acceleration

If `nvidia-smi` detects an NVIDIA GPU and CUDA-enabled PyTorch is already installed and verified, the installer reports it and skips the question and installation. If PyTorch is installed but CPU-only, it offers the CUDA installation. If no supported NVIDIA GPU is detected, the installer reports that GPU setup is unavailable and continues without asking.

## Optional packages

The installer checks each optional feature before asking. Satisfied features are reported as already installed and verified; they are not downloaded or reinstalled:

- `fer>=25.10.3` for emotion tracking (default: Yes).
- `mediapipe==0.10.21` for hand landmarks, finger counts, and gesture labels (default: Yes; the project uses the legacy Solutions API).
- `scikit-learn` and `sentence-transformers` for semantic search (default: Yes).
- `scipy` for faster Hungarian tracking assignment (default: Yes).

If an optional feature is enabled without its package, the application logs a warning and continues with that feature disabled.

## Starting the application

After setup, use the root README for the supported application start commands.

## Troubleshooting

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If installation fails, review `logs/installer.log`, confirm network access, and ensure the active venv uses Python 3.11 or newer.

On failure or cancellation, the installer removes partial model downloads and restores or removes a configuration file created during that invocation. Installer diagnostics are stored under `logs/`. During an explicit repair, only distributions identified as incompatible are removed before replacement; unrelated environment packages are not removed.
