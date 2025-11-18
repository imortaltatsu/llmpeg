"""
LLM-powered FFmpeg assistant CLI.

This package exposes:
    - Typer-powered CLI application (`llmpeg.cli.app`)
    - Tooling utilities for executing ffmpeg commands
    - GPT-compatible API client for LLM completions
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version."""
    try:
        return version("llmpeg")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]

