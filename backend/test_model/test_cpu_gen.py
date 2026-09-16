"""Generate a short VoiceCanvas sample with Qwen3-TTS on CPU only."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "output" / "cpu" / "voicecanvas_cpu.wav"
DEFAULT_TEXT = "Hello from VoiceCanvas. This speech was generated on a CPU."
DEFAULT_INSTRUCTION = "A warm, clear narrator speaking at a relaxed pace."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one Qwen3-TTS VoiceDesign WAV file on CPU only."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model ID.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="English text to synthesize.")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Natural-language voice and delivery direction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"WAV destination (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for PyTorch (default: 4).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated codec tokens (default: 512).",
    )
    return parser


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def peak_memory_mib() -> float | None:
    """Return the process high-water mark where the platform exposes it."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None

    # macOS reports bytes; Linux reports KiB.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def validate_audio(output_path: Path) -> tuple[int, float, float]:
    samples, sample_rate = sf.read(output_path, always_2d=False)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {output_path}: {sample_rate}")
    if samples.size == 0:
        raise ValueError(f"Generated file contains no samples: {output_path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Generated file contains non-finite samples: {output_path}")

    peak_amplitude = float(np.max(np.abs(samples)))
    if peak_amplitude == 0.0:
        raise ValueError(f"Generated file contains only silence: {output_path}")

    duration_seconds = samples.shape[0] / sample_rate
    return sample_rate, duration_seconds, peak_amplitude


def main() -> int:
    args = build_parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("voicecanvas.cpu")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    logger.info("Python: %s", sys.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("Torch CUDA build: %s", torch.version.cuda or "none")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Qwen-TTS: %s", package_version("qwen-tts"))
    logger.info("SoundFile: %s", package_version("soundfile"))
    logger.info("Transformers: %s", package_version("transformers"))
    logger.info("CPU threads: %d", args.threads)

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("This test requires a CPU-only PyTorch installation")

    logger.info("Loading model on CPU: %s", args.model)
    load_started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    load_seconds = time.perf_counter() - load_started

    devices = {parameter.device.type for parameter in model.model.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError(f"Model parameters are not exclusively on CPU: {sorted(devices)}")
    logger.info("Model loaded on CPU in %.2f seconds", load_seconds)

    logger.info("Generating one sample")
    generation_started = time.perf_counter()
    wavs, sample_rate = model.generate_voice_design(
        text=args.text,
        language="English",
        instruct=args.instruction,
        max_new_tokens=args.max_new_tokens,
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(wavs) != 1:
        raise ValueError(f"Expected one waveform, received {len(wavs)}")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], sample_rate)
    verified_sample_rate, duration_seconds, peak_amplitude = validate_audio(output_path)

    realtime_factor = generation_seconds / duration_seconds
    peak_memory = peak_memory_mib()
    logger.info("Output: %s", output_path)
    logger.info("Sample rate: %d Hz", verified_sample_rate)
    logger.info("Audio duration: %.2f seconds", duration_seconds)
    logger.info("Generation time: %.2f seconds", generation_seconds)
    logger.info("Generation-to-audio ratio: %.2fx", realtime_factor)
    logger.info("Peak audio amplitude: %.6f", peak_amplitude)
    if peak_memory is not None:
        logger.info("Peak process memory: %.1f MiB", peak_memory)
    else:
        logger.info("Peak process memory: unavailable")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("CPU generation failed")
        raise SystemExit(1)
