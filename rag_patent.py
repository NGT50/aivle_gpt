from typing import List, Iterable

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

import numpy as np

# openai is required for actual use but may not be available in the testing environment
try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None


class PatentRAG:
    """Simple RAG pipeline for generating patent specifications."""

    def __init__(self, embedding_model: str = "text-embedding-ada-002") -> None:
        self.embedding_model = embedding_model
        self._texts: List[str] = []
        self._index = None
        if tiktoken:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        else:  # fallback tokenizer
            self._tokenizer = None

    # ------------------------------------------------------------------
    # Token utilities
    def _tokenize(self, text: str) -> List[str]:
        if self._tokenizer:
            return self._tokenizer.encode(text)
        return text.split()

    def _detokenize(self, tokens: Iterable) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(list(tokens))
        return " ".join(tokens)

    # ------------------------------------------------------------------
    def add_document(self, path: str, chunk_size: int = 500, overlap: int = 50) -> None:
        """Load a document from ``path`` and split into chunks."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = self._tokenize(text)
        step = chunk_size - overlap
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + chunk_size]
            self._texts.append(self._detokenize(chunk_tokens))

    def build_index(self) -> None:
        """Embed chunks and build a vector index."""
        if not openai:
            raise ImportError("openai package is required to build the index")
        if not faiss:
            raise ImportError("faiss package is required to build the index")

        embeddings = []
        for txt in self._texts:
            resp = openai.Embedding.create(input=txt, model=self.embedding_model)
            emb = np.array(resp["data"][0]["embedding"], dtype="float32")
            embeddings.append(emb)

        embs = np.vstack(embeddings)
        dim = embs.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embs)

    def _embed_query(self, query: str) -> np.ndarray:
        if not openai:
            raise ImportError("openai package is required for embedding")
        resp = openai.Embedding.create(input=query, model=self.embedding_model)
        return np.array(resp["data"][0]["embedding"], dtype="float32")

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Return ``k`` most relevant text chunks for the query."""
        if self._index is None:
            raise RuntimeError("index not built")
        q_emb = self._embed_query(query)
        D, I = self._index.search(np.array([q_emb]), k)
        return [self._texts[i] for i in I[0]]

    def generate_specification(self, summary: str, k: int = 4, model: str = "gpt-4", temperature: float = 0.2) -> str:
        """Generate a patent specification using retrieved text chunks."""
        if not openai:
            raise ImportError("openai package is required to generate text")

        context = "\n\n".join(self.retrieve(summary, k))
        prompt = (
            "Use the following reference materials when drafting the patent specification:"\
            f"\n\n{context}\n\n"\
            f"Write a detailed patent specification based on: {summary}"
        )

        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"].strip()


__all__ = ["PatentRAG"]
