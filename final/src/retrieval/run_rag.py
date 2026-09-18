from rag_core import answer_question, retrieve  # noqa: F401 (retrieve kept for clarity)


def main():
    print("\n" + "=" * 50)
    print("Food Safety RAG System Ready (Type 'exit' to quit)")
    print("=" * 50)

    while True:
        q = input("\nAsk a food safety question: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue

        answer, passages = answer_question(q, top_k=3)

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for i, p in enumerate(passages, 1):
            print(f"[{i}] ({p['passage_id']}) {p['text'][:160]}...")


if __name__ == "__main__":
    main()
