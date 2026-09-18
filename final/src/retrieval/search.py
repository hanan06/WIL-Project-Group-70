import os
from pyserini.search import FaissSearcher
from pyserini import trectools
from pyserini.output_writer import OutputFormat, get_output_writer
import pandas as pd

DATA_DIR = "../../data"
# NOTE: this used to point at data/topics.csv, which never existed for
# this project (it was left over from Walert). Our actual test questions
# live in data/eval/topics.csv - see data/eval/README.md.
TOPICS = DATA_DIR + "/eval/topics.csv"

# Dense Retrieval
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'castorini/tct_colbert-v2-hnp-msmarco'
OUTPUT_PATH = '../../target/runs/rag-dense-faiss.txt'
RUN = "dense-faiss"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
topics = pd.read_csv(TOPICS)

searcher = FaissSearcher(
    INDEX,
    QUERY_ENCODER
)

tag = "walert.rag." + RUN
num_hits = 100
output_filename = OUTPUT_PATH

output_writer = get_output_writer(
    output_filename, 
    OutputFormat('trec'), 
    'w',
    max_hits=num_hits, 
    tag=tag, 
    topics=topics
)

with output_writer:
    for question_id, question in topics[['question_id', 'question']].values:
        hits = searcher.search(str(question), num_hits)
        output_writer.write(question_id, hits)
