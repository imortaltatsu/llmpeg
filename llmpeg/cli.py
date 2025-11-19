"""
Typer CLI entrypoint for the llmpeg assistant.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer

from .config import CLIOptions, ModelConfig, RuntimeConfig
from .conversation import FFmpegExecutor
from .llm import LLMClient, LLMClientError

app = typer.Typer(help="LLM-powered FFmpeg command assistant.", add_completion=False)

# Import setup commands
from . import setup as setup_module
app.add_typer(setup_module.app, name="setup")


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False
    
    # Verify ffmpeg works by checking version
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _check_config() -> bool:
    """Check if config is initialized."""
    config_py = Path.home() / ".llmpeg" / "config.py"
    config_json = Path.home() / ".llmpeg" / "config.json"
    
    # Check if either config file exists
    if config_py.exists() or config_json.exists():
        return True
    
    # Also check if API key is set via environment variables
    env_api_key = os.getenv("LLMPEG_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    env_provider = os.getenv("LLMPEG_PROVIDER")
    
    # If provider is ollama, config might not be needed (no API key required)
    if env_provider == "ollama":
        return True
    
    # If API key is set via env, consider config initialized
    if env_api_key:
        return True
    
    return False


def _build_executor(options: CLIOptions) -> FFmpegExecutor:
    llm = LLMClient(options.model)
    return FFmpegExecutor(llm=llm, options=options)


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Natural language command. Omit for interactive mode.",
    ),
) -> None:
    """Main CLI entrypoint."""
    # If a subcommand was invoked, don't run the main callback
    if ctx.invoked_subcommand is not None:
        return
    
    # Check FFmpeg installation
    if not _check_ffmpeg():
        typer.secho(
            "❌ FFmpeg is not installed or not available in PATH.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "\nPlease install FFmpeg:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  • macOS: brew install ffmpeg",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  • Ubuntu/Debian: sudo apt-get install ffmpeg",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  • Windows: Download from https://ffmpeg.org/download.html",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "\nAfter installing, verify with: ffmpeg -version",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    
    # Check config initialization
    if not _check_config():
        typer.secho(
            "⚠️  Configuration not initialized.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "\nPlease run: llmpeg setup init",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "\nOr set environment variables:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  export LLMPEG_PROVIDER=openrouter",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  export LLMPEG_API_KEY=your-api-key",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    
    # Try to load config from Python config.py file first (static path)
    config_py_file = Path.home() / ".llmpeg" / "config.py"
    base_model = ModelConfig()
    config_loaded = False
    
    if config_py_file.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("llmpeg_config", config_py_file)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                if hasattr(config_module, "model_config"):
                    base_model = config_module.model_config
                    config_loaded = True
        except Exception:
            # Fall back to JSON config
            pass
    
    # Try to load config from JSON file if Python config not found
    if not config_loaded:
        config_file = Path.home() / ".llmpeg" / "config.json"
        if config_file.exists():
            try:
                import json
                with open(config_file, "r") as f:
                    saved_config = json.load(f)
                    # Update base_model with saved config
                    base_model = base_model.model_copy(update=saved_config)
                    config_loaded = True
            except Exception:
                pass  # Fall back to defaults
    
    # Allow environment variables to override config
    model_updates = {}
    env_provider = os.getenv("LLMPEG_PROVIDER")
    if env_provider:
        model_updates["provider"] = env_provider.strip()
    
    env_api_key = os.getenv("LLMPEG_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if env_api_key:
        model_updates["api_key"] = env_api_key.strip()
    
    env_base_url = os.getenv("LLMPEG_BASE_URL")
    if env_base_url:
        model_updates["base_url"] = env_base_url.strip()
    
    env_model_name = os.getenv("LLMPEG_MODEL_NAME")
    if env_model_name:
        model_updates["model_name"] = env_model_name.strip()
    
    env_top_p = os.getenv("LLMPEG_TOP_P")
    if env_top_p:
        try:
            model_updates["top_p"] = float(env_top_p.strip())
        except ValueError:
            typer.secho(f"Invalid LLMPEG_TOP_P value: {env_top_p}", fg=typer.colors.YELLOW)
    
    options = CLIOptions(
        model=base_model.model_copy(update=model_updates),
        runtime=RuntimeConfig(),
    )

    executor = _build_executor(options)

    try:
        if not prompt:
            executor.repl()
        else:
            result = executor.run_once(prompt)
            typer.echo(result)
    except (LLMClientError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the llmpeg CLI."""
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()

