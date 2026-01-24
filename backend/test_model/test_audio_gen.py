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


def main(text: str, instructions: str) -> None:
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
			chosen_speaker = config.default_speaker
			logger.info("Using speaker: %s", chosen_speaker)

			wavs, sr = model.generate_custom_voice(
				text=text,
				language="English",
				speaker=chosen_speaker,
				instruct=instructions,
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
	text = """
	Friends. Mumbaikars. Lovers of the carb-on-carb miracle.

I stand before you today not to talk about potatoes. Oh, the potato is important, yes. 
The batata vada needs its spicy heart, the mustard seeds popping in hot oil, the turmeric bleeding its sunshine hue into the mash, the curry leaves singing their aromatic song. 
We know this. We respect this.

But what good is a heart... if it has no ribcage?

What good is a soul... if it has no body to house it?

I am here to speak of the unsung hero. The silent guardian. The golden armor that stands between culinary ecstasy and absolute, soggy disaster. I am here to speak... of the Besan Batter.
	"""
	instructions = "Speak in a cheerful and energetic tone."
	main(text, instructions)