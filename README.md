# Clipster
The AI for your clipboard
<img src="Clipster_icon.png" alt="logo" width="200"/>

## Use

### Env
Set env vars with your API key to activate AI endpoints
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- XAI_API_KEY
```bash
ANTHROPIC_API_KEY=secret XAI_API_KEY=secret clipster
```
And follow the instructions (hotkeys)\
The result goes to your clipboard

## Models
- Anthropic: Sonnet 4.5
- OpenAI: GPT 5.1
- xAI: Grok 4
- STT: Whisper Tiny, in `./models/ggml-tiny.en.bin`

## STT (Speach To Text)
To upgrade whisper model, download the model you want
```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin -O models/ggml-medium.en.bin
```
and change it in the code in main.rs\
recompile
```bash
cargo build --release
```
or just run it
```bash
cargo run --release
```

<img src="Clipster.png" alt="logo"/>
