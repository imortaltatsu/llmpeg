"""
Configuration models for the llmpeg CLI runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, field_validator


class ModelConfig(BaseModel):
    """Settings that control the GPT-compatible API client."""

    provider: Literal["openrouter", "ollama", "custom"] = Field(
        default="openrouter",
        description="API provider: openrouter, ollama, or custom.",
    )
    api_key: str = Field(
        default="",

        description="API key for authentication (required for OpenRouter, optional for Ollama).",
    )
    base_url: str = Field(
        default="",
        description="Base URL for custom endpoints. Auto-set for OpenRouter/Ollama if empty.",
    )
    model_name: str = Field(
        default="gpt-oss:20b",
        description="Model name to use for completions.",
    )
    max_output_tokens: PositiveInt = Field(
        default=2048,
        description="Maximum number of generated tokens per response.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        """Set default base_url based on provider if not specified."""
        if not self.base_url:
            if self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"


class RuntimeConfig(BaseModel):
    """Execution-time settings that affect tool behavior."""

    workdir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Working directory for running ffmpeg commands.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, log commands without executing them.",
    )
    verbose: bool = False
    confirm: bool = Field(
        default=False,
        description="Prompt before executing each tool call.",
    )

    @field_validator("workdir")
    @classmethod
    def _ensure_workdir(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir():
            raise ValueError(f"Workdir {resolved} is not a directory")
        return resolved


class CLIOptions(BaseModel):
    """Top-level bundle combining model + runtime behavior."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    log_json: bool = Field(
        default=False, description="Emit structured JSON log lines."
    )
    show_reasoning: bool = Field(
        default=True,
        description="Print the LLM-provided context/explanation.",
    )


__all__ = ["CLIOptions", "ModelConfig", "RuntimeConfig"]

