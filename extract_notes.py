"""Turn a lecture recording into structured study notes.

Pipeline: audio -> Whisper transcript -> Claude -> notes saved as JSON + Notion.
"""

from pydantic import BaseModel, ConfigDict, Field

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
