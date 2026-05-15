"""PDF text extraction. Takes a PDF path, returns extracted text."""
from pathlib import Path
from pypdf import PdfReader

def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_and_save(pdf_path: str, output_dir: str = "data/processed") -> str:
    """Extract text from a PDF and save to a .txt file. Returns output path."""
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir) / f"{pdf_name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = extract_text(pdf_path)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <pdf_path>")
        sys.exit(1)
    output = extract_and_save(sys.argv[1])
    print(f"Extracted text saved to: {output}")
