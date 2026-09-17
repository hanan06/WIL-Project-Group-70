import pandas as pd
from pyserini.search import FaissSearcher

DATA_DIR = "../../data"
COLLECTION = DATA_DIR + "/collection.csv"
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'castorini/tct_colbert-v2-hnp-msmarco'

print("Loading FAISS Searcher...")
searcher = FaissSearcher(INDEX, QUERY_ENCODER)
collection_df = pd.read_csv(COLLECTION)

# Normalize column names in case collection uses 'id'/'contents' or 'passage_id'/'passage'
id_col = 'passage_id' if 'passage_id' in collection_df.columns else collection_df.columns[0]
text_col = 'passage' if 'passage' in collection_df.columns else collection_df.columns[1]

def get_context_passages(question, top_k=3):
    hits = searcher.search(question, top_k)
    passages = []
    for hit in hits:
        matched = collection_df[collection_df[id_col].astype(str) == str(hit.docid)]
        if not matched.empty:
            passages.append(matched.iloc[0][text_col])
    return passages

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Food Safety RAG Retrieval Test (Type 'exit' to quit)")
    print("="*50)

    while True:
        query = input("\nAsk a food safety question: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if not query:
            continue

        retrieved = get_context_passages(query, top_k=3)
        print(f"\n--- Top Retrieved Documents for: '{query}' ---")
        for i, p in enumerate(retrieved, 1):
            print(f"\n[Document {i}]:\n{p[:350]}...")