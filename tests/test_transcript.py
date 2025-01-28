from pathlib import Path

import pytest
from src.sentiment_classification.transcript import CustomerTranscript, CustomerTranscriptFactory


@pytest.fixture
def sample_text():
    return """Agent: Hello, how can I assist you today?
Member: I need help with my account.
Agent: Sure, let me pull up your account details.
Member: Thank you.
Agent: You're welcome. Anything else?"""


@pytest.fixture
def empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    return empty_file


@pytest.fixture
def sample_file(tmp_path, sample_text):
    file = tmp_path / "sample.txt"
    file.write_text(sample_text, encoding="utf-8")
    return file


@pytest.fixture
def sample_folder(tmp_path, sample_text):
    folder = tmp_path / "sample_folder"
    folder.mkdir()
    for i in range(3):
        file = folder / f"transcript_{i}.txt"
        file.write_text(sample_text, encoding="utf-8")
    return folder


def test_customer_transcript_agent_transcript(sample_text):
    transcript = CustomerTranscript(text=sample_text, file_name="test.txt")
    expected_agent_transcript = """Agent: Hello, how can I assist you today?
Agent: Sure, let me pull up your account details.
Agent: You're welcome. Anything else?"""
    assert transcript.agent_transcript == expected_agent_transcript


def test_customer_transcript_customer_transcript(sample_text):
    transcript = CustomerTranscript(text=sample_text, file_name="test.txt")
    expected_customer_transcript = """Member: I need help with my account.
Member: Thank you."""
    assert transcript.customer_transcript == expected_customer_transcript


def test_factory_from_text_file(sample_file):
    factory = CustomerTranscriptFactory()
    transcript = factory.from_text_file(sample_file)
    assert transcript.file_name == str(sample_file)
    assert "Agent: Hello, how can I assist you today?" in transcript.text


def test_factory_from_text_file_empty_file(empty_file):
    factory = CustomerTranscriptFactory()
    with pytest.raises(ValueError, match="File empty"):
        factory.from_text_file(empty_file)


def test_factory_from_folder(sample_folder):
    factory = CustomerTranscriptFactory()
    transcripts = factory.from_folder(sample_folder)

    transcripts = sorted(transcripts, key=lambda t: Path(t.file_name).name)

    assert len(transcripts) == 3
    for i, transcript in enumerate(transcripts):
        file_name = Path(transcript.file_name).name
        assert file_name.endswith(f"transcript_{i}.txt")
        assert "Agent: Hello, how can I assist you today?" in transcript.text