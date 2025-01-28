from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, computed_field


class CustomerTranscript(BaseModel):
    file_name: str = Field(..., description="The path to the transcript")
    text: str = Field(..., description="A transcript of a customer interaction")

    @computed_field
    def agent_transcript(self) -> str:
        non_member_lines = [
            line for line in self.text.split("\n") if not line.startswith("Member:")
        ]
        return "\n".join(non_member_lines)

    @computed_field
    def customer_transcript(self) -> str:
        member_lines = [
            line for line in self.text.split("\n") if line.startswith("Member:")
        ]
        return "\n".join(member_lines)


class CustomerTranscriptFactory:

    def from_text_file(self, file_path: Path) -> CustomerTranscript:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        if len(content) == 0:
            raise ValueError("File empty")
        return CustomerTranscript(text=content, file_name=str(file_path))

    def from_folder(self, folder_path: Path) -> List[CustomerTranscript]:
        return [self.from_text_file(path) for path in Path(folder_path).glob("*.txt")]
