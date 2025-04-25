from typing import List
from langchain.schema import Document
from pydantic import BaseModel

class SDGState(BaseModel):
    input: str
    documents: List[Document] = []
    evolved_question: str = ""
    context: List[str] = []
    answer: str = ""