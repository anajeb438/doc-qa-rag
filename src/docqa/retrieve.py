import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import DATA_DIR, chunk_pdf

# A small, free embedding model that runs on your own computer
MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)
        texts = [chunk["text"] for chunk in chunks]
        # One embedding (384 numbers) per chunk.
        # normalize_embeddings=True makes the similarity maths simple.
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, question, top_k=3):
        """Return the top_k chunks closest in meaning to the question."""
        question_embedding = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.embeddings @ question_embedding  # similarity with every chunk at once
        best = np.argsort(scores)[::-1][:top_k]         # indexes of the highest scores
        return [(self.chunks[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    chunks = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf))
    print(f"Embedding {len(chunks)} chunks...")

    retriever = Retriever(chunks)

    question = "How does multi-head attention work?"
    print(f"\nQuestion: {question}\n")
    for chunk, score in retriever.search(question):
        print(f"--- {chunk['source']}, page {chunk['page']} (score {score:.2f}) ---")
        print(chunk["text"][:300])
        print()