# Backend

Initial backend directory for the project.

## Environment configuration

Copy the example file and update values:

```
cp .env.example .env
```

The app reads settings via `dotenv` in `voice_poc_qwen3.config.Config`. See `.env.example` for the full list of variables.

## Speakers

The TTS models include several pre-defined speaker voices. Use the speaker identifier when calling the generation API.

| Speaker | Voice Description | Native language |
|---|---|---|
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

Programmatically accessible speaker names are exposed via `voice_poc_qwen3.config.config.valid_speakers`.

## CPU VoiceDesign test

`test_model/test_cpu_gen.py` generates one short English WAV with
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` on CPU. It uses a natural-language
voice instruction instead of a preset speaker, downloads the model into the
standard Hugging Face cache when it is not already present, and writes the
result to `../data/output/cpu/voicecanvas_cpu.wav` when run from this directory.

From the repository root, create a local environment and install the CPU-only
PyTorch wheels before installing the remaining dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchaudio==2.7.1
.venv/bin/python -m pip install qwen-tts soundfile
.venv/bin/python -m pip install -e ./backend
```

Run the test with four CPU threads:

```bash
.venv/bin/python backend/test_model/test_cpu_gen.py --threads 4
```

Use `--help` to see options for the input text, voice instruction, model path,
output path, CPU threads, and token ceiling. The script validates that the
PyTorch build and loaded model are CPU-only, then reports package versions,
model-load time, generation time, audio duration, generation-to-audio ratio,
peak process memory when available, and WAV file checks.

### Measured CPU run

The following result was measured with the default text and voice instruction
on a 13th Gen Intel Core i3-13100T using four PyTorch threads. The 1.7B model
was already present in the Hugging Face cache, so the load time excludes its
initial download.

| Item | Result |
| --- | --- |
| Python | 3.12.3 |
| PyTorch / torchaudio | 2.7.1+cpu / 2.7.1+cpu |
| Qwen-TTS / Transformers / SoundFile | 0.1.1 / 4.57.3 / 0.14.0 |
| Model load time | 7.50 seconds |
| Generation time | 32.80 seconds |
| Generated audio | 3.92 seconds at 24,000 Hz |
| Generation-to-audio ratio | 8.37x |
| Peak process memory | 11,629.8 MiB |
| WAV checks | 94,080 finite samples with a nonzero peak amplitude (0.770142) |

The run completed with a CPU-only PyTorch build (`torch.version.cuda` was
`None` and `torch.cuda.is_available()` was `False`). Qwen-TTS emits a warning
when the system `sox` executable is absent; the VoiceDesign CPU test completed
successfully despite that warning.
