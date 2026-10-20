from pathlib import Path
from pypdf import PdfReader

# Path to the project folder: ingest.py -> docqa -> src -> doc-qa-rag
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_pdf(path):
    """Read a PDF and return a list of (page_number, text) pairs."""
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            pages.append((page_number, text))
    return pages


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping pieces of about chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def chunk_pdf(path, chunk_size=1000, overlap=200):
    """Load a PDF and return chunks that remember their file and page."""
    chunks = []
    for page_number, text in load_pdf(path):
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append({"source": Path(path).name, "page": page_number, "text": piece})
    return chunks


if __name__ == "__main__":
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}")
    else:
        chunks = chunk_pdf(pdfs[0])
        print(f"{pdfs[0].name}: {len(chunks)} chunks")
        if chunks:
            print("--- first chunk ---")
            print(chunks[0]["text"])
            print(f"(from page {chunks[0]['page']})")
        else:
            print("No text found. It might be a scanned PDF; try a different one.")