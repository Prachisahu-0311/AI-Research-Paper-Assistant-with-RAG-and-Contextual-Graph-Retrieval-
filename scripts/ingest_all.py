"""One-shot ingestion: chunk + embed + store all 20 papers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.chunking import chunk_text, load_paper_titles
from src.embeddings import get_model
from src.vector_store import VectorStore

BATCH_SIZE = 32
STORE_BATCH_SIZE = 500


def main() -> None:
    confirm = input(
        "This will reset the vector store and re-ingest all papers. Continue? (y/n): "
    ).strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    titles = load_paper_titles()
    processed_dir = Path("data/processed")
    txt_files = sorted(processed_dir.glob("*.txt"))

    all_chunks: list[dict] = []
    for i, txt_path in enumerate(txt_files, 1):
        paper_id = txt_path.stem
        paper_title = titles.get(paper_id, paper_id)
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, paper_id, paper_title)
        print(f"[{i}/{len(txt_files)}] {paper_id} ({paper_title[:45]}) — {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all papers: {len(all_chunks)}")

    print("\nEmbedding chunks...")
    model = get_model()
    all_embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch_texts = [c["text"] for c in all_chunks[i : i + BATCH_SIZE]]
        vecs = model.encode(batch_texts, batch_size=BATCH_SIZE, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())

    print("\nStoring in vector store...")
    vs = VectorStore()
    vs.reset()
    for i in tqdm(range(0, len(all_chunks), STORE_BATCH_SIZE)):
        vs.add_chunks(
            all_chunks[i : i + STORE_BATCH_SIZE],
            all_embeddings[i : i + STORE_BATCH_SIZE],
        )

    print(f"\nIngested {vs.count()} chunks from {len(txt_files)} papers into vector store.")


if __name__ == "__main__":
    main()
