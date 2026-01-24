from pathlib import Path
import logging
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_poc_qwen3.config import config


# configure logging: suppress library/info noise, keep this module's INFO logs
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_voice_clone")
logger.setLevel(logging.INFO)

model_path = config.qwen3_tts_12hz_17b_base_dir

logger.info("Model path resolved: %s", model_path)

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

logger.info("Model loaded successfully.")

source_audio_path = config.data_dir / "input" / "source_audio.wav"
source_audio_transcript_path = config.data_dir / "input" / "source_audio_transcript.txt"

if not source_audio_path.is_file():
    raise FileNotFoundError(f"Source audio file not found: {source_audio_path}")
else:
    logger.info("Source audio file found: %s", source_audio_path)

if not source_audio_transcript_path.is_file():
    raise FileNotFoundError(f"Source audio transcript file not found: {source_audio_transcript_path}")
else:
    logger.info("Source audio transcript file found: %s", source_audio_transcript_path)


ref_audio = source_audio_path
ref_text = source_audio_transcript_path.read_text(encoding="utf-8").strip()
