# aivle_gpt

Utilities for drafting patent specifications with GPT models.

## Requirements

- Python 3.8+
- `openai`
- `faiss-cpu`
- `tiktoken` (optional, for token-accurate chunking)

Install dependencies with:

```bash
pip install openai faiss-cpu tiktoken
```

## RAG-based patent generation

The `rag_patent.py` module implements a simple retrieval augmented generation workflow.

```python
from rag_patent import PatentRAG

rag = PatentRAG()
rag.add_document("docs/reference1.txt")
rag.add_document("docs/reference2.txt")
rag.build_index()

summary = "Short summary of the invention"
spec = rag.generate_specification(summary)
print(spec)
```

Provide your OpenAI API key using the `OPENAI_API_KEY` environment variable. The
class splits input documents into chunks, embeds them, retrieves the most
relevant portions for a summary and calls GPT-4 to draft the specification.
