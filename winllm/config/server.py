"""API server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_alias: Optional[str] = None
    api_key: Optional[str] = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    stream_token_timeout: float = 120.0  # Seconds to wait for each token during streaming
