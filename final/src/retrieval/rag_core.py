import os
import pandas as pd
from pyserini.search import FaissSearcher
from groq import Groq

try:
    # python-dotenv lets us load a .env file automatically if present.
    # If it isn't installed, we just skip this step silently -
    # the environment variable can still be set manually.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
COLLECTION = os.path.join(DATA_DIR, "collection.csv")
INDEX = os.path.join(
    os.path.dirname(__file__), "..", "..", "target", "indexes",
    "tct_colbert-v2-hnp-msmarco-faiss"
)
QUERY_ENCODER = "castorini/tct_colbert-v2-hnp-msmarco"
GROQ_MODEL = "openai/gpt-oss-20b"

_searcher = None
_collection_df = None
_client = None
_id_col = None
_text_col = None


def get_api_key() -> str:
    """Read the Groq API key from the environment. Raise a clear error if missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Create a .env file (copy .env.example) or run:\n"
            "  export GROQ_API_KEY=your_real_key_here"
        )
    return key


def load_resources():
    """
    Load the FAISS searcher, the passage collection, and the Groq client
    ONCE, and reuse them on every question. This is important for a web
    app (Streamlit re-runs your script on every click) so we don't reload
    a multi-second index on every single interaction.
    """
    global _searcher, _collection_df, _client, _id_col, _text_col

    if _searcher is None:
        _searcher = FaissSearcher(INDEX, QUERY_ENCODER)

    if _collection_df is None:
        _collection_df = pd.read_csv(COLLECTION)
        _id_col = "passage_id" if "passage_id" in _collection_df.columns else _collection_df.columns[0]
        _text_col = "passage" if "passage" in _collection_df.columns else _collection_df.columns[1]

    if _client is None:
        _client = Groq(api_key=get_api_key())

    return _searcher, _collection_df, _client, _id_col, _text_col


def retrieve(query: str, top_k: int = 3):
    """
    Search the FAISS index for the top_k passages most similar to `query`.
    Returns a list of dicts: {"passage_id": ..., "text": ..., "score": ...}
    """
    searcher, collection_df, _, id_col, text_col = load_resources()
    hits = searcher.search(query, top_k)

    results = []
    for hit in hits:
        matched = collection_df[collection_df[id_col].astype(str) == str(hit.docid)]
        if not matched.empty:
            results.append({
                "passage_id": str(hit.docid),
                "text": matched.iloc[0][text_col],
                "score": float(hit.score),
            })
    return results


UNANSWERED_PHRASES = [
    "knowledge base doesn't cover",
    "knowledge base does not cover",
    "not covered",
    "i don't have information",
    "i do not have information",
    "cannot answer",
    "can't answer",
]


def is_unanswered(answer_text: str) -> bool:
    """
    Very simple heuristic used by the evaluation script to flag an answer
    as "declined / out of scope", mirroring Walert's "% unanswered" metric.
    This is intentionally simple - for your report, spot-check a sample of
    these flags by hand rather than trusting the heuristic blindly.
    """
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in UNANSWERED_PHRASES)


def generate(query: str, passages) -> str:
    """
    Send the retrieved passages + the user's question to the Groq-hosted
    LLM, and return the generated answer text.
    `passages` can be a list of strings, or a list of dicts with a "text" key.
    """
    _, _, client, _, _ = load_resources()

    texts = [p["text"] if isinstance(p, dict) else p for p in passages]
    context = "\n\n".join(f"[Doc {i+1}]: {t}" for i, t in enumerate(texts))

    prompt = f"""You are a helpful Australian Food Safety Assistant.
Answer the question based strictly on the retrieved passages below. If the information isn't present, state clearly that the knowledge base doesn't cover it.

Context:
{context}

Question: {query}
Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.1,
    )
    return chat_completion.choices[0].message.content


def answer_question(query: str, top_k: int = 3):
    """Convenience wrapper: retrieve + generate in one call."""
    passages = retrieve(query, top_k=top_k)
    answer = generate(query, passages)
    return answer, passages
