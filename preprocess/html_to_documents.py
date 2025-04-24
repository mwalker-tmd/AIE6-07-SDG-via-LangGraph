from bs4 import BeautifulSoup
from langchain.schema import Document
from pathlib import Path
from typing import List


def extract_documents_from_html(file_path: str, source_label: str) -> List[Document]:
    """
    Parses an HTML file and extracts paragraphs into LangChain Document objects.

    Parameters:
    - file_path (str): Path to the HTML file.
    - source_label (str): Identifier to use in the metadata of each Document.

    Returns:
    - List[Document]: Extracted documents from the HTML content.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]

    documents = [
        Document(page_content=para, metadata={"source": source_label, "filename": file_path.name})
        for para in paragraphs
    ]
    return documents
