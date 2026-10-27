# audio-notes-extractor

Turn a lecture recording into study notes: a summary, key concepts, topics in order, assignments, exam hints, and review questions.

Pipeline: audio (`.wav`, `.mp3`, `.m4a`) → Whisper transcript → a local model through Ollama → study notes saved as Markdown (import them into Notion).

## Requirements

- Python 3.10 or newer
- ffmpeg, for reading audio: `brew install ffmpeg` on macOS
- [Ollama](https://ollama.com), running, with the notes model downloaded: `ollama pull gemma4:12b`

## Usage

```bash
python extract_notes.py lecture.m4a
```

This writes `lecture.notes.md` next to the recording (drag it into Notion), plus
`lecture.transcript.json`, so running it again skips Whisper.

Options:

- `--whisper-model small`: a bigger Whisper model makes fewer mistakes, but is slower (default: `base`)
- `--language el`: tell Whisper the lecture's language instead of detecting it
