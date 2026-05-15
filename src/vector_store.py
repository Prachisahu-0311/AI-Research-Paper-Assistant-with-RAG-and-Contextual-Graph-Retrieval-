"""Vector database wrapper for storing and querying embeddings."""
import chromadb
from pathlib import Path


class VectorStore:
    def __init__(self, collection_name: str = "papers") -> None:
        self._client = chromadb.PersistentClient(path="./data/chroma_db")
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Batch add chunks and their embeddings to the collection."""
        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "paper_id": c["paper_id"],
                    "paper_title": c["paper_title"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return top-k chunks closest to the query embedding."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                "chunk_id": results["ids"][0][i],
                "paper_id": meta["paper_id"],
                "paper_title": meta["paper_title"],
                "chunk_index": meta["chunk_index"],
                "text": results["documents"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })
        return output

    def query_scoped(
        self, query_embedding: list[float], paper_ids: list[str], top_k: int = 5
    ) -> list[dict]:
        """Return top-k chunks restricted to the given paper_ids."""
        if not paper_ids:
            return self.query(query_embedding, top_k)
        n = min(top_k, self.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"paper_id": {"$in": paper_ids}},
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                "chunk_id": results["ids"][0][i],
                "paper_id": meta["paper_id"],
                "paper_title": meta["paper_title"],
                "chunk_index": meta["chunk_index"],
                "text": results["documents"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        """Return number of chunks stored in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


if __name__ == "__main__":
    vs = VectorStore()
    print(f"Chunks in vector store: {vs.count()}")
