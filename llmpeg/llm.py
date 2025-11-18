"""
OpenAI-compatible API client for LLM completions.
Supports OpenRouter, Ollama, and custom endpoints.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from .config import ModelConfig


class LLMClientError(RuntimeError):
    """Raised when the API request fails."""


@dataclass
class LLMClient:
    """Chat-completion style client targeting OpenAI-compatible APIs."""

    config: ModelConfig

    def __post_init__(self) -> None:
        """Initialize the API client."""
        if OpenAI is None:
            raise LLMClientError(
                "openai package is not installed. Run `uv add openai`."
            )
        
        self._last_thinking_content = ""  # Store thinking content for display if available
        
        # Get API key from config or environment
        api_key = self.config.api_key or self._get_api_key_from_env()
        
        # Initialize OpenAI client
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment based on provider."""
        if self.config.provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY", "")
        elif self.config.provider == "ollama":
            # Ollama doesn't require API key, but some setups might use one
            return os.getenv("OLLAMA_API_KEY", "")
        else:
            return os.getenv("LLMPEG_API_KEY", "")

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute a chat completion and return the assistant message content.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_output_tokens,
            )
        except Exception as exc:
            raise LLMClientError(f"API request failed: {exc}") from exc
        
        # Extract the response content
        if not response.choices:
            raise LLMClientError("API response did not include choices.")
        
        message = response.choices[0].message
        content = message.content
        
        if not isinstance(content, str):
            raise LLMClientError("API response missing string content.")
        
        # Check for thinking/reasoning content if available (some APIs include this)
        # This is typically in response metadata or message annotations
        if hasattr(message, "reasoning") and message.reasoning:
            self._last_thinking_content = message.reasoning
        
        return content
