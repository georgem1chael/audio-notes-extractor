# audio-notes-extractor

Turn a lecture recording into study notes: a summary, key concepts, topics in order, assignments, exam hints, and review questions.

Pipeline: audio (`.wav`, `.mp3`, `.m4a`) → Whisper transcript → Claude → notes saved as JSON and in Notion.

> Work in progress. This is being built step by step; usage instructions arrive with the CLI.

## Requirements

- Python 3.10 or newer
- ffmpeg, for reading audio: `brew install ffmpeg` on macOS
- [Ollama](https://ollama.com), running, with the notes model downloaded: `ollama pull gemma4:12b`