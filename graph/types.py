from typing import List
from langchain.schema import Document
from pydantic import BaseModel, Field

class SDGState(BaseModel):
    input: str = Field(default="")
    documents: List[Document] = Field(default_factory=list)
    evolved_question: str = Field(default="")
    context: List[str] = Field(default_factory=list)
    answer: str = Field(default="")
    num_evolve_passes: int = Field(default=2)