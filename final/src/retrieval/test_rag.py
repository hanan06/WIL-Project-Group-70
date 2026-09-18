from rag_core import retrieve


def main():
    print("\n" + "=" * 50)
    print("Food Safety RAG Retrieval Test (Type 'exit' to quit)")
    print("=" * 50)

    while True:
        query = input("\nAsk a food safety question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        retrieved = retrieve(query, top_k=3)
        print(f"\n--- Top Retrieved Documents for: '{query}' ---")
        for i, p in enumerate(retrieved, 1):
            print(f"\n[Document {i}] ({p['passage_id']}, score={p['score']:.3f}):\n{p['text'][:350]}...")


if __name__ == "__main__":
    main()
