import os
import pandas as pd
from pyserini.search import FaissSearcher
from groq import Groq

GROQ_API_KEY = "make a groq api key and put it here.... groQ not groK"

DATA_DIR = "../../data"
COLLECTION = DATA_DIR + "/collection.csv"
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'castorini/tct_colbert-v2-hnp-msmarco'

print("Loading local FAISS index...")
searcher = FaissSearcher(INDEX, QUERY_ENCODER)
collection_df = pd.read_csv(COLLECTION)
client = Groq(api_key=GROQ_API_KEY)

id_col = 'passage_id' if 'passage_id' in collection_df.columns else collection_df.columns[0]
text_col = 'passage' if 'passage' in collection_df.columns else collection_df.columns[1]

def retrieve(query, top_k=3):
    hits = searcher.search(query, top_k)
    passages = []
    for hit in hits:
        matched = collection_df[collection_df[id_col].astype(str) == str(hit.docid)]
        if not matched.empty:
            passages.append(matched.iloc[0][text_col])
    return passages

def generate(query, passages):
    context = "\n\n".join([f"[Doc {i+1}]: {p}" for i, p in enumerate(passages)])
    
    prompt = f"""You are a helpful Australian Food Safety Assistant.
Answer the question based strictly on the retrieved passages below. If the information isn't present, state clearly that the knowledge base doesn't cover it.

Context:
{context}

Question: {query}
Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=0.1,
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Food Safety RAG System Ready (Type 'exit' to quit)")
    print("="*50)

    while True:
        q = input("\nAsk a food safety question: ").strip()
        if q.lower() in ['exit', 'quit', 'q']:
            break
        if not q:
            continue

        docs = retrieve(q, top_k=3)
        ans = generate(q, docs)

        print("\n--- Answer ---")
        print(ans)
        print("\n--- Sources ---")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d[:160]}...")