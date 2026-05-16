"""Text chunking. Splits raw text into chunks suitable for embedding."""
import re
import statistics
import tiktoken
from pathlib import Path

PAPER_YEARS = {
    "1706.03762": 2017, "1810.04805": 2018, "1907.11692": 2019,
    "1909.11942": 2019, "1910.01108": 2019, "1906.08237": 2019,
    "2005.14165": 2020, "1910.10683": 2019, "1901.02860": 2019,
    "2001.04451": 2020, "2005.12872": 2020, "2005.00743": 2020,
    "2002.12327": 2020, "2006.16236": 2020, "1807.03819": 2018,
    "2010.11929": 2020, "2103.00020": 2021, "2101.03961": 2021,
    "2003.05997": 2020, "2003.10555": 2020,
}


def load_paper_titles(index_path: str = "data/papers/index.md") -> dict[str, str]:
    """Parse index.md and return {arxiv_id: title} dict."""
    titles = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|.*?\|\s*(\d{4}\.\d{4,5})\s*\|", line)
            if match:
                title = match.group(1).strip()
                arxiv_id = match.group(2).strip()
                titles[arxiv_id] = title
    return titles


def chunk_text(
    text: str,
    paper_id: str,
    paper_title: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """Split text into overlapping token-based chunks with metadata."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    chunk_index = 0
    step = chunk_size - overlap
    i = 0

    year = PAPER_YEARS.get(paper_id, "unknown")
    prefix = f"[Paper: {paper_title}, Year: {year}]\n"

    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        if len(chunk_tokens) < 50:
            break
        chunks.append({
            "chunk_id": f"{paper_id}_chunk_{chunk_index}",
            "paper_id": paper_id,
            "paper_title": paper_title,
            "chunk_index": chunk_index,
            "text": prefix + enc.decode(chunk_tokens),
            "token_count": len(chunk_tokens),
        })
        chunk_index += 1
        i += step

    return chunks


if __name__ == "__main__":
    path = Path("data/processed/1706.03762.txt")
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, "1706.03762", "Attention Is All You Need")

    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk ({chunks[0]['token_count']} tokens):")
    print(chunks[0]["text"][:200])
    print(f"\nLast chunk ({chunks[-1]['token_count']} tokens):")
    print(chunks[-1]["text"][:200])

    token_counts = [c["token_count"] for c in chunks]
    print(f"\nToken count distribution:")
    print(f"  min:  {min(token_counts)}")
    print(f"  max:  {max(token_counts)}")
    print(f"  mean: {statistics.mean(token_counts):.1f}")

    print("\nPaper titles from index.md:")
    titles = load_paper_titles()
    for arxiv_id, title in sorted(titles.items()):
        print(f"  {arxiv_id}: {title}")
