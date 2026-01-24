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
logger = logging.getLogger("test_audio_gen")
logger.setLevel(logging.INFO)


def create_voice(model, text: str, instructions: str, filename: str) -> None:
    start_total = time.perf_counter()

    try:
        t1 = time.perf_counter()
        logger.debug("Generating audio for file: %s", filename)
        chosen_speaker = config.default_speaker
        logger.debug("Using speaker: %s", chosen_speaker)

        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker=chosen_speaker,
            instruct=instructions,
        )
        gen_time = time.perf_counter() - t1
        logger.info("Audio generation completed (%.2fs) — produced %d wave(s), sample rate %d", gen_time, len(wavs), sr)

        output_path = Path(config.data_dir) / "output" / "patty" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t2 = time.perf_counter()
        logger.debug("Writing output to %s", output_path)
        sf.write(str(output_path), wavs[0], sr)
        write_time = time.perf_counter() - t2
        logger.debug("Wrote output (%.2fs)", write_time)

        total_time = time.perf_counter() - start_total
        logger.debug("Finished generation step in %.2fs (gen: %.2fs, write: %.2fs)", total_time, gen_time, write_time)

    except Exception:
        logger.exception("Error during audio generation for %s", filename)


if __name__ == "__main__":
    text = """

        You think this is just meat? You think this is merely a commodity to be buried under cheap cheddar and wilted lettuce? You tragic, hollow soul. This isn't dinner. This is architecture.

Look at this grind. Look at the ratio. That fat isn't just white speckling; it is the holy spirit of the burger, waiting to be released. If you mistreat it... if you pack it too tight... you are strangling an angel. The toppings are the choir, the bun is the pew, but the patty is the sermon. And if the sermon is dry, the church is empty.

And you want to press on it? You want to take your spatula, that blunt instrument of ignorance, and squeeze the life force out of it? To hear it hiss? That hiss isn't cooking. That hiss is the sound of the burger’s soul screaming as it evaporates into the hood vent! You are robbing it of its juice, its essence, its very reason for existing!

We are looking for a crucible here! A war between the heat of the iron and the cool integrity of the center. If you overwork this meat, if you handle it with your warm, clumsy hands until the fat melts before it hits the pan, you have failed. You have created a hockey puck of despair.

When a customer bites into this, they shouldn't taste "beef." They should taste victory. They should taste the iron of the earth and the smoke of the gods. So do not talk to me about speed. Everything else—the cheese, the sauce, the pickle—is just a lie we tell ourselves to hide from the truth. And the truth is the meat.

Now. Wash your hands. And try again.


    """
    instructions = "dramatic, emotional, and deliberately over-the-top"

    audio_styles = [
    {
        "title": "The Kitchen Nightmare",
        "description": "Pure Rage",
        "instruction": "furious, shouting, abusive, fast-paced, high energy, explosive",
        "filename": "audio_rage.wav"
    },
    {
        "title": "The Noir Detective",
        "description": "Inner Monologue",
        "instruction": "gravelly, cynical, slow-paced, film noir voiceover, tired, low pitch",
        "filename": "audio_noir.wav"
    },
    {
        "title": "The Televangelist",
        "description": "The Sermon",
        "instruction": "preaching, evangelical, rhythmic, booming resonance, Southern drawl, passionate",
        "filename": "audio_sermon.wav"
    },
    {
        "title": "Unsettling ASMR",
        "description": "The Creepy Whisper",
        "instruction": "soft whisper, intimate, breathy, slow, tingling, menacingly quiet",
        "filename": "audio_asmr.wav"
    },
    {
        "title": "The Shakespearean Actor",
        "description": "The Thespian",
        "instruction": "theatrical, grandiose, projecting, old-fashioned, articulated, regal, dramatic",
        "filename": "audio_theatrical.wav"
    },
    {
        "title": "The Broken Chef",
        "description": "The Breakdown",
        "instruction": "sobbing, trembling voice, shaky, hysterical, sorrowful, on the verge of tears",
        "filename": "audio_crying.wav"
    },
    {
        "title": "The Hal 9000",
        "description": "The Cold Analysis",
        "instruction": "robotic, monotone, flat, precise, unemotional, synthetic, cold",
        "filename": "audio_robotic.wav"
    },
    {
        "title": "The Sarcastic Teenager",
        "description": "The Bore",
        "instruction": "bored, monotone, vocal fry, dismissive, sarcastic, eye-rolling tone",
        "filename": "audio_bored.wav"
    },
    {
        "title": "The Movie Trailer Guy",
        "description": "The Hype",
        "instruction": "deep voice, epic, gravelly, trailer narration, intense, suspenseful",
        "filename": "audio_trailer.wav"
    },
    {
        "title": "The Bedtime Story",
        "description": "The Gentle Parent",
        "instruction": "gentle, soothing, soft, melodic, slow, warm, storytelling",
        "filename": "audio_bedtime.wav"
    }
    ]

    model_path = config.qwen3_tts_12hz_17b_customvoice_dir
    logger.debug("Model path resolved: %s", model_path)

    logger.info("Loading model...")
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    for style in audio_styles:
        logger.info("Processing style: %s (%s)", style['title'], style['description'])
        create_voice(model, text, style['instruction'], style['filename'])