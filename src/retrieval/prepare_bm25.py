# Copies the corpus into a folder Pyserini's Lucene indexer accepts, using the field names it requires (id, contents).
import json, os

SRC = "../../data/collection.jsonl"
OUT_DIR = "../../data/bm25_collection"
os.makedirs(OUT_DIR, exist_ok=True)

n = 0
with open(SRC, encoding="utf-8") as f, open(os.path.join(OUT_DIR, "collection.jsonl"), "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        text = d.get("contents") or d.get("text")  # accept either field name
        out.write(json.dumps({"id": d["id"], "contents": text}) + "\n")
        n += 1
print(f"Wrote {n} passages to {OUT_DIR}")