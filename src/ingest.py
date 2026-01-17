from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pdf(file_path: Path) -> str:
    """Extract text from a single PDF file."""
    reader = PdfReader(file_path)
    text = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)


def ingest_pdfs():
    """Load all PDFs from data directory and save extracted text."""
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data directory.")
        return

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        content = load_pdf(pdf)

        output_file = OUTPUT_DIR / f"{pdf.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved extracted text to: {output_file}")


if __name__ == "__main__":
    ingest_pdfs()
