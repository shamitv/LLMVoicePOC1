from pathlib import Path
import logging
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_poc_qwen3.config import config


# configure logging
_LOG_LEVEL = getattr(logging, config.log_level.upper(), logging.INFO)
logging.basicConfig(
	level=_LOG_LEVEL,
	format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_audio_gen")


def main() -> None:
	start_total = time.perf_counter()
	logger.info("Starting audio generation example")

	model_path = config.qwen3_tts_12hz_17b_customvoice_dir
	logger.info("Model path resolved: %s", model_path)

	try:
		t0 = time.perf_counter()
		logger.info("Loading model from %s", model_path)
		model = Qwen3TTSModel.from_pretrained(
			str(model_path),
			device_map="cuda:0",
			dtype=torch.bfloat16,
			attn_implementation="sdpa",
		)
		load_time = time.perf_counter() - t0
		logger.info("Model loaded (%.2fs)", load_time)

		t1 = time.perf_counter()
		logger.info("Generating audio...")
		wavs, sr = model.generate_custom_voice(
			text="I realized I'm especially good at noticing other people's emotions.",
			language="English",
			speaker="Vivian",
			instruct="Say it in a very angry tone.",
		)
		gen_time = time.perf_counter() - t1
		logger.info("Audio generation completed (%.2fs) — produced %d wave(s), sample rate %d", gen_time, len(wavs), sr)

		output_path = Path(config.data_dir) / "output_custom_voice.wav"
		output_path.parent.mkdir(parents=True, exist_ok=True)

		t2 = time.perf_counter()
		logger.info("Writing output to %s", output_path)
		sf.write(str(output_path), wavs[0], sr)
		write_time = time.perf_counter() - t2
		logger.info("Wrote output (%.2fs)", write_time)

		total_time = time.perf_counter() - start_total
		logger.info("Finished all steps in %.2fs (load: %.2fs, gen: %.2fs, write: %.2fs)", total_time, load_time, gen_time, write_time)

	except Exception:
		logger.exception("Error during audio generation example")


if __name__ == "__main__":
	main()
