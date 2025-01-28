from unittest.mock import MagicMock
from src.sentiment_classification.classifier import SentimentClassifier, CustomerTranscript, SentimentClassification
from instructor.client import Instructor

import pytest


def test_sentiment_classifier_invalid_mode():
    # Test that invalid mode raises an exception
    with pytest.raises(ValueError, match="mode must be one of 'customer' or 'full'"):
        SentimentClassifier(instructor_client=MagicMock(), model_name="test_model", mode="invalid")


def test_sentiment_classifier_predict_customer_mode():
    # Test prediction in "customer" mode
    transcript_text = """Member: Hello, I need help.\nAgent: Sure, how can I assist you?"""
    transcript = CustomerTranscript(file_name="test_transcript.txt", text=transcript_text)

    mock_client = MagicMock(spec=Instructor)
    mock_chat_completion = MagicMock()
    mock_chat_completion.create.return_value = SentimentClassification(
        file_name="test_transcript.txt",
        chain_of_thought="Step by step reasoning",
        sentiment="positive",
        follow_up_action="issue_resolved"
    )
    mock_client.chat.completions.create = mock_chat_completion.create

    sentiment_classifier = SentimentClassifier(instructor_client=mock_client, model_name="test_model", mode="customer")

    result = sentiment_classifier.predict(transcript)

    assert result.file_name == "test_transcript.txt"
    assert result.sentiment == "positive"
    assert result.follow_up_action == "issue_resolved"


def test_sentiment_classifier_predict_full_mode():
    # Test prediction in "full" mode
    transcript_text = """Member: Hello, I need help.\nAgent: Sure, how can I assist you?"""
    transcript = CustomerTranscript(file_name="test_transcript.txt", text=transcript_text)

    mock_client = MagicMock(spec=Instructor)
    mock_chat_completion = MagicMock()
    mock_chat_completion.create.return_value = SentimentClassification(
        file_name="test_transcript.txt",
        chain_of_thought="Step by step reasoning",
        sentiment="neutral",
        follow_up_action="follow_up_call_needed"
    )
    mock_client.chat.completions.create = mock_chat_completion.create

    sentiment_classifier = SentimentClassifier(instructor_client=mock_client, model_name="test_model", mode="full")

    result = sentiment_classifier.predict(transcript)

    assert result.file_name == "test_transcript.txt"
    assert result.sentiment == "neutral"
    assert result.follow_up_action == "follow_up_call_needed"