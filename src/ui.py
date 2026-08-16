"""Small reusable terminal presentation helpers for NIRT ShooterBot."""

from __future__ import annotations

import os
import sys


def print_banner(title: str) -> None:
    """Print a compact startup banner without depending on installer UI code."""
    line = "=" * 72
    color = "" if os.environ.get("NO_COLOR") or not sys.stdout.isatty() else "\033[96m"
    reset = "\033[0m" if color else ""
    print(f"{color}{line}{reset}")
    print(f"{color}  {title}{reset}")
    print(f"{color}{line}{reset}")
