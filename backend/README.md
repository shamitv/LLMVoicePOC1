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
