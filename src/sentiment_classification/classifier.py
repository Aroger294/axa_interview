from typing import Literal
from instructor.client import Instructor
from pydantic import BaseModel, Field

from src.sentiment_classification.transcript import CustomerTranscript


class SentimentClassification(BaseModel):
    file_name: str = Field("", description="Path to the transcript")
    chain_of_thought: str = Field(
        ...,
        description="Think step by step to determine the correct sentiment classification "
        "and follow up action classification",
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ...,
        description="The predicted sentiment class label",
    )
    follow_up_action: Literal["issue_resolved", "follow_up_call_needed"] = Field(
        ...,
        description="The outcome of the call - was the issue resolved or is another call needed?",
    )


class SentimentClassifier:
    def __init__(
        self,
        instructor_client: Instructor,
        model_name: str,
        mode: Literal["customer", "full"],
    ):

        if mode not in ["customer", "full"]:
            raise ValueError("mode must be one of 'customer' or 'full'")

        self.client = instructor_client
        self.model_name = model_name
        self.mode = mode

    def predict(self, transcript: CustomerTranscript) -> SentimentClassification:

        prompt = {}

        if self.mode == "customer":
            prompt = {
                "role": "user",
                "content": f"""
                    You have been provided with only the customer portion of a customer service call transcript.
                    
                    Classify the sentiment and the outcome of the call from the following: 
                    {transcript.customer_transcript}
                    """,
            }

        if self.mode == "full":
            prompt = {
                "role": "user",
                "content": f"""
                    You have been provided with of a call transcript.
                    
                    Classify the sentiment and the outcome of the call from the following: 
                    {transcript.text}
                    """,
            }

        classification = self.client.chat.completions.create(
            model=self.model_name,
            messages=[prompt],
            response_model=SentimentClassification,
        )

        classification.file_name = transcript.file_name

        return classification
