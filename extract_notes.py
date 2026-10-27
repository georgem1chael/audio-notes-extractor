"""Turn a lecture recording into structured study notes.

Pipeline: audio -> Whisper transcript -> Claude -> notes saved as JSON + Notion.
"""

import json
import shutil
from pathlib import Path
from types import ModuleType

import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a"}
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_NOTES_MODEL = "gemma4:12b"
MAX_NOTES_TOKENS = 8000

# Errors
#
# Each kind of failure gets its own exception class. Later, the CLI will
# turn each one into a clear message instead of a long traceback.


class NotesError(Exception):
    """Base for every failure we expect and can explain to the user."""


class AudioFileError(NotesError):
    """The input file is missing, empty, or not a supported format."""


class WhisperUnavailableError(NotesError):
    """openai-whisper or ffmpeg is not installed."""


class TranscriptionError(NotesError):
    """Whisper ran but failed, or heard nothing usable."""


class NotesGenerationError(NotesError):
    """The local model couldn't be reached or didn't return usable notes."""


# What Whisper gives us


class TranscriptSegment(BaseModel):
    """One stretch of speech and when it happened, in seconds from the start"""

    model_config = ConfigDict(str_strip_whitespace=True)

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    """Everything Whisper heard in one recording."""

    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


# What we ask Claude to give us
#
# Claude will see these field descriptions, so they double as instructions.
# A field typed `str | None` with no default must be present but may be null:
# Claude has to say "unknown" explicitly rather than quietly leave it out.


class _Strict(BaseModel):
    """Base for the notes models: reject any field we didn't ask for."""

    model_config = ConfigDict(extra="forbid")

class Concept(_Strict):
    """A term or idea the lecture introduced or relied on."""

    term: str = Field(description="The name of the concept, as the tutor used it.")
    explanation: str = Field(description="What it means, in one or two plain sentences a student could revise from.")

class Topic(_Strict):
    """One part of the lecture, in the order it was taught."""

    heading: str = Field(description="A short heading for this part of the lecture.")
    starts_at: str | None = Field(description="Where this part begins in the recording, as HH:MM:SS; null if unclear.")
    points: list[str] = Field(description="The main points, examples and explanations from this part, one per item.")

class Assignment(_Strict):
    """Work the tutor set: homework, reading, or preparation for next time."""

    task: str = Field(description="What to do, e.g. 'Read chapter 4' or 'Finish exercise sheet 3'.")
    due: str | None = Field(description="Deadline as stated (e.g. 'next Tuesday'); null if none was given.")

class LectureNotes(_Strict):
    """Study notes for one lecture."""

    title: str = Field(min_length=1, description="A short title for the lecture, 3-8 words, e.g. 'Database normalization'.")
    summary: str = Field(
        min_length=1,
        description="3-6 sentences: what the lecture covered and how its parts fit together.",
    )
    key_concepts: list[Concept]
    topics: list[Topic] = Field(min_length=1, description="The lecture's parts, in the order they were taught.")
    assignments: list[Assignment]
    exam_hints: list[str] = Field(
        description="Anything the tutor stressed as important or likely to be examined, close to their own words."
    )
    review_questions: list[str] = Field(
        description="3-5 questions a student could answer afterwards to check they understood the lecture."
    )


# Transcription


def validate_audio_path(path: Path) -> Path:
    """Check that `path` is a non-empty audio file we can transcribe."""
    if not path.exists():
        raise AudioFileError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioFileError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        extension = path.suffix or "(no extension)"
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise AudioFileError(f"Unsupported format {extension} for {path.name}. Supported: {supported}")
    if path.stat().st_size == 0:
        raise AudioFileError(f"Audio file is empty: {path}")
    return path


def load_whisper() -> ModuleType:
    """Import Whisper, or explain how to install whatever is missing."""
    try:
        import whisper
    except ImportError as e:
        raise WhisperUnavailableError("Whisper is not installed. Run: pip install openai-whisper") from e
    if not hasattr(whisper, "load_model"):
        raise WhisperUnavailableError("This 'whisper' is the wrong package. Install openai-whisper instead.")
    if shutil.which("ffmpeg") is None:
        raise WhisperUnavailableError("ffmpeg not found; Whisper needs it to read audio. Run: brew install ffmpeg")
    return whisper


def parse_whisper_result(result: dict) -> Transcript:
    """Turn the dict that Whisper's transcribe() returns into a Transcript."""
    try:
        segments = [TranscriptSegment.model_validate(s) for s in result.get("segments", [])]
    except ValidationError as e:
        raise TranscriptionError(f"Whisper returned a malformed segment: {e}") from e
    segments = [s for s in segments if s.text]

    text = result.get("text", "").strip() or " ".join(s.text for s in segments)
    if not text:
        raise TranscriptionError("Whisper heard no speech in this recording.")
    return Transcript(text=text, language=result.get("language"), segments=segments)


def transcribe(
    audio_path: Path, whisper_model: str = DEFAULT_WHISPER_MODEL, language: str | None = None
) -> Transcript:
    """Run Whisper on an audio file and return what it heard."""
    validate_audio_path(audio_path)
    whisper = load_whisper()

    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:  # unknown model name, failed download, damaged model file
        raise TranscriptionError(f"Could not load Whisper model '{whisper_model}': {e}") from e

    try:
        result = model.transcribe(str(audio_path), language=language, fp16=False, verbose=False)
    except Exception as e:  # ffmpeg couldn't decode the file, or Whisper itself failed
        # ffmpeg's errors run to dozens of lines; the last one says what went wrong.
        last_line = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
        raise TranscriptionError(f"Whisper failed on {audio_path.name}: {last_line}") from e

    return parse_whisper_result(result)


SYSTEM_PROMPT = """\
You turn lecture transcripts into study notes for a student revising for exams.

The transcript comes from automatic speech recognition, so some words are misheard \
(for example "free" instead of "three"). Fix obvious mistakes from the context, but \
only write down what the tutor actually said: never add facts, examples or deadlines \
that are not in the lecture.

Each line of the transcript starts with the time it was said, as [HH:MM:SS]. Use \
those times for each topic's starts_at.

Write the notes in the same language as the lecture."""


def format_timestamp(seconds: float) -> str:
    """Turn a number of seconds, like 3725.5, into "01:02:05"."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_prompt(transcript: Transcript) -> str:
    """The message for the model: the JSON it must reply with, then the timestamped transcript."""
    schema = json.dumps(LectureNotes.model_json_schema(), indent=2)
    lines = [f"[{format_timestamp(s.start)}] {s.text}" for s in transcript.segments]
    text = "\n".join(lines) or transcript.text
    return f"Reply with JSON that follows this schema:\n{schema}\n\nLecture transcript:\n{text}"


def generate_notes(
    transcript: Transcript, model: str = DEFAULT_NOTES_MODEL, attempts: int = 3
) -> LectureNotes:
    """Ask the local model for study notes, trying again if a reply comes back unusable."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(transcript)},
    ]
    problem = ""
    for attempt in range(1, attempts + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                format=LectureNotes.model_json_schema(),
                options={"temperature": 0, "num_predict": MAX_NOTES_TOKENS},
                think=False,
            )
        except ConnectionError as e:
            raise NotesGenerationError("Ollama isn't running. Open the Ollama app and try again.") from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise NotesGenerationError(f"{model} isn't downloaded. Run: ollama pull {model}") from e
            raise NotesGenerationError(f"Ollama error: {e.error}") from e

        if response.done_reason == "length":
            problem = f"the reply ran past {MAX_NOTES_TOKENS} tokens"
        else:
            try:
                return LectureNotes.model_validate_json(response.message.content)
            except ValidationError as e:
                problem = f"the reply didn't match the notes format ({e.error_count()} problems)"
        print(f"Attempt {attempt} of {attempts} failed: {problem}")

    raise NotesGenerationError(f"No usable notes after {attempts} attempts: {problem}")
