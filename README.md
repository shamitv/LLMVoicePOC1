# LLMVoicePOC1

An early voice creation studio for expressive narration and character voices, powered by Qwen3-TTS.

LLMVoicePOC1 explores turning written text and plain-language voice descriptions into speech. The aim is to let creators direct how a voice sounds—its tone, emotion, pace, and character—for storytelling, voiceovers, and character dialogue.

For example, the same passage could be delivered as a weary noir detective, a theatrical narrator, or an energetic movie-trailer voice. Changing the voice direction lets you explore different performances of the same words.

## Product vision

The intended studio workflow is:

1. Enter the text you want spoken.
2. Describe the voice and delivery, such as “a low, gravelly voice, slow-paced and tired.”
3. Generate variations with different voice directions.
4. Preview and compare the performances.
5. Export the selected audio for use in your project.

The studio interface, interactive preview, and selection experience are future capabilities. Today, the repository contains a Python speech-generation prototype; the frontend is a placeholder, and no application API is implemented.

## What the prototype does

The [audio-generation script](backend/test_model/test_audio_gen.py) loads a local model and applies ten example voice directions to a shared English passage. These include angry, noir, sermon, whisper, theatrical, tearful, robotic, sarcastic, trailer, and storytelling deliveries.

Text and voice descriptions are currently edited in the script. It calls the model for each style, writes generated audio as WAV files under `DATA_DIR/output/patty/`, and logs model-loading and generation times. `DATA_DIR` defaults to `./data`, relative to the working directory.

This is a foundation for experimenting with expressive speech, with the broader studio experience still to be built.

## The model

The prototype uses [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), a speech-generation model in the Qwen3-TTS family. VoiceDesign uses natural-language descriptions to guide voice characteristics and delivery, including tone, emotion, and speaking style.

The current script uses English and loads model files from `MODELS_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Inference is configured for an NVIDIA GPU through CUDA (`cuda:0`), using PyTorch BF16 precision (`torch.bfloat16`) and scaled dot-product attention (`sdpa`). These are the script's current settings, not measured performance guarantees.

Configuration also defines paths for other Qwen3-TTS variants (CustomVoice and Base), the 12Hz tokenizer, and Faster-Whisper large-v3. Those path definitions do not represent additional implemented product workflows; the generation script selects VoiceDesign. Likewise, the configured speaker list should not be read as a completed voice-selection feature in the studio.

## Tech stack

| Layer | Technology | Current role |
| --- | --- | --- |
| Language | Python | Generation script and backend configuration package |
| Speech generation | Qwen-TTS (`qwen-tts`) | Loads Qwen3-TTS and invokes voice-design generation |
| Model execution | PyTorch, CUDA | Runs local inference with BF16 and SDPA settings |
| Audio output | SoundFile | Writes generated waveforms to WAV files |
| Configuration | python-dotenv | Loads environment settings, including model and output locations |
| Backend foundation | FastAPI, Uvicorn, Pydantic | Declared dependencies; no application API is implemented yet |
| Integration dependencies | Requests, aiohttp, OpenAI Python SDK | Declared dependencies; unused by the current generation script |
| Frontend | No framework selected | Placeholder directory for the future studio interface |

The current speech-generation path uses local model inference. It does not call an OpenAI API.

## Explore the project

- [Generation prototype](backend/test_model/test_audio_gen.py): example text, voice directions, model loading, and WAV output.
- [Backend documentation](backend/README.md): environment configuration and speaker reference.
- [Configuration](backend/src/voice_poc_qwen3/config.py): model paths and runtime settings.
- [Frontend placeholder](frontend/README.md): starting point for the future studio interface.

## License

This repository is licensed under [Apache-2.0](LICENSE).
