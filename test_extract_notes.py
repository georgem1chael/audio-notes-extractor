"""Tests for extract_notes.py. Whisper and Ollama are faked, so they run in about a second."""

import json
from types import SimpleNamespace

import ollama
import pytest

import extract_notes
from extract_notes import (
    AudioFileError,
    LectureNotes,
    NotesGenerationError,
    Transcript,
    TranscriptionError,
    build_prompt,
    format_timestamp,
    generate_notes,
    load_or_transcribe,
    parse_whisper_result,
    render_markdown,
    save_notes,
    validate_audio_path,
)


GOOD_NOTES = {
    "title": "Database normalization",
    "summary": "An introduction to normal forms.",
    "key_concepts": [{"term": "1NF", "explanation": "One value per cell."}],
    "topics": [{"heading": "First normal form", "starts_at": "00:04:10", "points": ["One value per cell"]}],
    "assignments": [{"task": "Exercise sheet 3", "due": "next Tuesday"}],
    "exam_hints": ["Normalize a table to 3NF by hand"],
    "review_questions": ["What does 2NF add to 1NF?"],
}
TRANSCRIPT = Transcript(text="Today we talk about normalization.")


def fake_reply(content: str, done_reason: str = "stop") -> SimpleNamespace:
    """Something shaped like the response ollama.chat returns, with only the parts we read."""
    return SimpleNamespace(done_reason=done_reason, message=SimpleNamespace(content=content))


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65.9) == "00:01:05"
    assert format_timestamp(3725.5) == "01:02:05"


def test_a_real_recording_is_accepted(tmp_path):
    audio = tmp_path / "lecture.M4A"
    audio.write_bytes(b"pretend audio")
    assert validate_audio_path(audio) == audio


def test_a_missing_file_is_rejected(tmp_path):
    with pytest.raises(AudioFileError, match="not found"):
        validate_audio_path(tmp_path / "missing.m4a")


def test_a_wrong_format_is_rejected(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    with pytest.raises(AudioFileError, match="Unsupported format"):
        validate_audio_path(notes)


def test_an_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.m4a"
    empty.touch()
    with pytest.raises(AudioFileError, match="empty"):
        validate_audio_path(empty)


def test_whisper_output_is_cleaned_up():
    result = {
        "text": " Hello everyone. ",
        "language": "el",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.8, "text": " Hello everyone.", "tokens": [50364, 2425]},
            {"id": 1, "start": 1.8, "end": 2.1, "text": "   "},
        ],
    }
    transcript = parse_whisper_result(result)
    assert transcript.text == "Hello everyone."
    assert transcript.language == "el"
    assert [s.text for s in transcript.segments] == ["Hello everyone."]


def test_silence_is_rejected():
    with pytest.raises(TranscriptionError, match="no speech"):
        parse_whisper_result({"text": "", "segments": []})


def test_a_broken_segment_is_rejected():
    with pytest.raises(TranscriptionError, match="malformed segment"):
        parse_whisper_result({"text": "hi", "segments": [{"start": -1, "end": 2, "text": "hi"}]})


# The model's side, with ollama.chat replaced by a fake


def test_the_prompt_has_the_schema_and_timestamps():
    transcript = Transcript(text="Hello.", segments=[{"start": 65, "end": 70, "text": "Hello."}])
    prompt = build_prompt(transcript)
    assert "[00:01:05] Hello." in prompt
    assert "3-5 questions a student could answer" in prompt


def test_a_good_reply_becomes_notes(monkeypatch):
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: fake_reply(json.dumps(GOOD_NOTES)))
    notes = generate_notes(TRANSCRIPT)
    assert notes.title == "Database normalization"
    assert notes.assignments[0].due == "next Tuesday"


def test_a_runaway_reply_is_retried(monkeypatch):
    replies = iter([fake_reply("{    ", done_reason="length"), fake_reply(json.dumps(GOOD_NOTES))])
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: next(replies))
    assert generate_notes(TRANSCRIPT).title == "Database normalization"


def test_it_gives_up_after_three_bad_replies(monkeypatch):
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: fake_reply('{"title": ""}'))
    with pytest.raises(NotesGenerationError, match="after 3 attempts"):
        generate_notes(TRANSCRIPT)


def test_a_missing_model_says_how_to_download_it(monkeypatch):
    def not_found(**kwargs):
        raise ollama.ResponseError("model 'nope:1b' not found", 404)

    monkeypatch.setattr(ollama, "chat", not_found)
    with pytest.raises(NotesGenerationError, match="ollama pull nope:1b"):
        generate_notes(TRANSCRIPT, model="nope:1b")


def test_ollama_not_running_is_explained(monkeypatch):
    def refuse(**kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(ollama, "chat", refuse)
    with pytest.raises(NotesGenerationError, match="isn't running"):
        generate_notes(TRANSCRIPT)


# Saving


def test_markdown_layout():
    md = render_markdown(LectureNotes.model_validate(GOOD_NOTES), "lecture.m4a")
    assert md.startswith("# Database normalization\n")
    assert "### First normal form (00:04:10)" in md
    assert "- [ ] Exercise sheet 3 (due next Tuesday)" in md
    assert "1. What does 2NF add to 1NF?" in md


def test_empty_sections_say_none_mentioned():
    notes = LectureNotes.model_validate({**GOOD_NOTES, "assignments": [], "exam_hints": []})
    assert render_markdown(notes, "lecture.m4a").count("*None mentioned.*") == 2


def test_notes_are_saved_next_to_the_recording(tmp_path):
    path = save_notes(LectureNotes.model_validate(GOOD_NOTES), tmp_path / "lecture.m4a")
    assert path == tmp_path / "lecture.notes.md"
    assert path.read_text(encoding="utf-8").startswith("# Database normalization")


def test_a_saved_transcript_is_reused(tmp_path, monkeypatch):
    calls = []

    def fake_transcribe(*args):
        calls.append(args)
        return Transcript(text="Hello.")

    monkeypatch.setattr(extract_notes, "transcribe", fake_transcribe)
    audio = tmp_path / "lecture.m4a"
    first = load_or_transcribe(audio)
    second = load_or_transcribe(audio)
    assert first == second
    assert len(calls) == 1
    assert (tmp_path / "lecture.transcript.json").exists()