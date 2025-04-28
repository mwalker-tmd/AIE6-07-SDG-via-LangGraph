from typing import List
from langchain.schema import Document
from pydantic import BaseModel, Field

class SDGState(BaseModel):
    input: str = Field(default="")
    documents: List[Document] = Field(default_factory=list)
    evolved_questions: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    answer: str = Field(default="")
    num_evolve_passes: int = Field(default=2)

    @property
    def evolved_question(self):
        return self.evolved_questions[-1] if self.evolved_questions else ""