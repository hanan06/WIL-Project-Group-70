# Random-ranking sanity baseline: real metrics must beat this by a wide margin.
import random
import pandas as pd

TOPICS = "../../data/topics.csv"
COLLECTION = "../../data/collection.csv"
OUTPUT_PATH = "../../target/runs/random.txt"

topics = pd.read_csv(TOPICS)
ids = pd.read_csv(COLLECTION).iloc[:, 0].astype(str).tolist()
rng = random.Random(42)

with open(OUTPUT_PATH, "w") as out:
    for qid in topics["question_id"]:
        for rank, pid in enumerate(rng.sample(ids, 100), 1):
            out.write(f"{qid} Q0 {pid} {rank} {100 - rank} random\n")
print(f"Saved {OUTPUT_PATH}")