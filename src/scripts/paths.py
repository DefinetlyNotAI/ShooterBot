"""Project-root paths for utility scripts and their generated artifacts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_GENERATED_DIRECTORIES = {"files", "logs", "models"}


def generated_directory(name: str, *, create: bool = False) -> Path:
    """Return an approved root-level generated directory.

    Scripts must not write beside themselves or into the current working
    directory. Only these root-level directories may be created by a script.
    """
    if name not in _GENERATED_DIRECTORIES:
        raise ValueError(f"Unsupported generated directory: {name}")
    directory = ROOT / name
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def files_dir() -> Path:
    """Create and return the root-level directory for script output files."""
    return generated_directory("files", create=True)


def models_dir() -> Path:
    """Return the single permitted model directory without creating it."""
    return generated_directory("models")


def model_path(name: str) -> Path:
    """Resolve a model filename inside the root-level models directory."""
    filename = Path(name)
    if filename.name != name:
        raise ValueError("Model names must be filenames, not paths")
    return models_dir() / filename
