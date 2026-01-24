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
		text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
		language="Chinese",
		speaker="Vivian",
		instruct="用特别愤怒的语气说",
	)

	output_path = Path(config.data_dir) / "output_custom_voice.wav"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(output_path), wavs[0], sr)


if __name__ == "__main__":
	main()
