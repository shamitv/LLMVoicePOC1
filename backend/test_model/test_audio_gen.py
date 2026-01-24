from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_poc_qwen3.config import config


def main() -> None:
	model_path = config.qwen3_tts_12hz_17b_customvoice_dir

	model = Qwen3TTSModel.from_pretrained(
		str(model_path),
		device_map="cuda:0",
		dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
	)

	wavs, sr = model.generate_custom_voice(
		text="I realized I'm especially good at noticing other people's emotions.",
		language="English",
		speaker="Vivian",
		instruct="Say it in a very angry tone.",
	)

	output_path = Path(config.data_dir) / "output_custom_voice.wav"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(output_path), wavs[0], sr)


if __name__ == "__main__":
	main()
