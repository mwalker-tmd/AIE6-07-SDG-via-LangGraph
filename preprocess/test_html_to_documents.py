import pytest
from preprocess.html_to_documents import extract_documents_from_html
from langchain.schema import Document

"""
Test the HTML to documents conversion functionality
"""

def test_extract_documents_from_html_creates_documents_correctly(tmp_path):
    html_content = """
    <html>
      <body>
        <p>This is the first paragraph.</p>
        <p>This is the second paragraph.</p>
      </body>
    </html>
    """
    temp_file = tmp_path / "sample.html"
    temp_file.write_text(html_content)

    docs = extract_documents_from_html(str(temp_file), source_label="test-source")

    assert isinstance(docs, list)
    assert len(docs) == 2

    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.metadata["source"] == "test-source"
        assert doc.metadata["filename"] == "sample.html"

    contents = [doc.page_content for doc in docs]
    assert "This is the first paragraph." in contents
    assert "This is the second paragraph." in contents


def test_extract_documents_from_html_file_not_found():
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_documents_from_html("non_existent_file.html", source_label="test-source")

    assert "HTML file not found" in str(exc_info.value)