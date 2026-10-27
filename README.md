# audio-notes-extractor

Turn a lecture recording into study notes: a summary, key concepts, topics in order, assignments, exam hints, and review questions.

Pipeline: audio (`.wav`, `.mp3`, `.m4a`) → Whisper transcript → a local model through Ollama → study notes saved as Markdown (import them into Notion). Everything runs on your own computer: it's free, and the recording never leaves it.

Whisper runs where you tell it to, with `--device`:

| `--device` | For | Whisper package |
|---|---|---|
| `mac` | Macs with Apple Silicon (M1 or newer), on the GPU | [mlx-whisper](https://pypi.org/project/mlx-whisper/) |
| `nvidia` | Linux or Windows with an NVIDIA graphics card | [faster-whisper](https://pypi.org/project/faster-whisper/) |
| `cpu` | Any computer, on the processor (slower) | faster-whisper |

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download), running, with the notes model downloaded: `ollama pull gemma4:12b` (it needs about 10 GB of memory, and uses an NVIDIA card or Apple Silicon by itself)
- On a Mac: ffmpeg, for reading audio: `brew install ffmpeg`. Linux and Windows don't need it.
- For `--device nvidia`: NVIDIA's cuBLAS library for CUDA 12 (see below)

## Setup

macOS and Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

This installs mlx-whisper on Apple Silicon Macs, and faster-whisper everywhere else.

### NVIDIA on Linux

If the machine already has CUDA 12, there's nothing more to do. Otherwise, install cuBLAS into the venv (no admin rights needed), and tell Linux where it is in each new terminal before running:

```bash
.venv/bin/pip install nvidia-cublas-cu12
export LD_LIBRARY_PATH=$(.venv/bin/python -c "import nvidia.cublas, os; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))")
```

### NVIDIA on Windows

Install NVIDIA's [CUDA 12 Toolkit](https://developer.nvidia.com/cuda-toolkit-archive), which puts cuBLAS on your `PATH`.

## Usage

```bash
.venv/bin/python extract_notes.py lecture.m4a --device mac
```

Use `--device nvidia` or `--device cpu` on other computers. On Windows, run `.venv\Scripts\python` instead of `.venv/bin/python`.

The first run downloads the Whisper model (about 1.6 GB). After that, a one-hour lecture takes about 4 minutes on an M1 Max.

This writes `lecture.notes.md` next to the recording (drag it into Notion), plus
`lecture.transcript.json`, so running it again skips Whisper.

The notes are written in Greek, with technical terms kept in English. To change that, edit the last paragraph of `SYSTEM_PROMPT` in `extract_notes.py`.

Option:

- `--language el`: tell Whisper the lecture's language instead of detecting it

## Tests

```bash
.venv/bin/python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
