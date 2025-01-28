import json
import os
from typing import Literal

import typer
from rich.console import Console
from rich.progress import Progress
from pathlib import Path

import instructor
from openai import OpenAI

from classifier import SentimentClassifier
from transcript import CustomerTranscriptFactory


app = typer.Typer()

@app.command()
def classify(
    mode: str = typer.Option(
        ..., help='Mode for sentiment classification: "customer" or "full".', case_sensitive=False
    )
):
    """
    Classify customer transcripts using the specified mode.
    """
    if mode not in ["customer", "full"]:
        raise typer.BadParameter("Invalid mode. Must be 'customer' or 'full'.")

    transcript_folder = Path("../data/transcripts_v3")
    MODEL = "gpt-4o-mini"

    console = Console()
    console.log(f"[cyan]Using {MODEL} in {mode} mode...")

    instructor_client = instructor.from_openai(
        OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        mode=instructor.Mode.JSON,
    )

    transcript_factory = CustomerTranscriptFactory()
    classifier = SentimentClassifier(instructor_client, MODEL, mode)

    transcripts = transcript_factory.from_folder(transcript_folder)

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Classifying transcripts...", total=len(transcripts)
        )

        classified_transcripts = []
        for transcript in transcripts:
            classified_transcript = classifier.predict(transcript)
            classified_transcripts.append(classified_transcript.model_dump())

            # Update progress
            progress.update(task, advance=1)

    with open(f"classifications_{mode}.json", "w") as json_file:
        json.dump(classified_transcripts, json_file, indent=4)

    console.print("[bold green]Classification completed successfully![/bold green]")


if __name__ == "__main__":
    app()