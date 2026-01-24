"""Configuration helpers for the backend.

Loads environment variables (including .env) and exposes common settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Config:
    """App configuration loaded from environment variables.

    Loads values from a .env file (if present) and provides helpers to
    retrieve typed values. Common variables are exposed as properties.
    """

    def __init__(self, dotenv_path: Optional[str | Path] = None, override: bool = False) -> None:
        resolved_path = Path(dotenv_path).expanduser() if dotenv_path else None
        load_dotenv(dotenv_path=resolved_path, override=override)
        self._dotenv_path = str(resolved_path) if resolved_path else None

    def get(self, key: str, default: Any = None, cast: Optional[Callable[[str], Any]] = None) -> Any:
        value = os.getenv(key)
        if value is None or value == "":
            return default
        return cast(value) if cast else value

    def require(self, key: str, cast: Optional[Callable[[str], Any]] = None) -> Any:
        value = self.get(key, default=None, cast=cast)
        if value is None or value == "":
            raise KeyError(f"Missing required environment variable: {key}")
        return value

    @property
    def env(self) -> str:
        return self.get("ENV", "development")

    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL", "info")

    @property
    def app_host(self) -> str:
        return self.get("APP_HOST", "0.0.0.0")

    @property
    def app_port(self) -> int:
        return self.get("APP_PORT", 8000, cast=int)

    @property
    def debug(self) -> bool:
        return self.get("DEBUG", False, cast=_to_bool)

    @property
    def openai_api_key(self) -> str:
        return self.get("OPENAI_API_KEY", "")

    @property
    def openai_base_url(self) -> str:
        return self.get("OPENAI_BASE_URL", "")

    @property
    def model_name(self) -> str:
        return self.get("MODEL_NAME", "")

    @property
    def models_dir(self) -> Path:
        return Path(self.get("MODELS_DIR", r"D:\\work\\models"))

    @property
    def qwen_dir(self) -> Path:
        return self.models_dir / "Qwen"

    @property
    def qwen3_tts_tokenizer_12hz_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-Tokenizer-12Hz"

    @property
    def qwen3_tts_12hz_17b_customvoice_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice"

    @property
    def qwen3_tts_12hz_17b_voicedesign_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    @property
    def qwen3_tts_12hz_17b_base_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-12Hz-1.7B-Base"

    @property
    def qwen3_tts_12hz_06b_customvoice_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-12Hz-0.6B-CustomVoice"

    @property
    def qwen3_tts_12hz_06b_base_dir(self) -> Path:
        return self.qwen_dir / "Qwen3-TTS-12Hz-0.6B-Base"

    @property
    def faster_whisper_large_v3_dir(self) -> Path:
        return self.models_dir / "Systran" / "faster-whisper-large-v3"


config = Config()

__all__ = ["Config", "config"]
