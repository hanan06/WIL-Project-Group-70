"""
app.py
------
A minimal browser-based front end for the Food Safety RAG Assistant,
using Streamlit. This is what turns your CLI tool into "a prototype of a
browser-based interactive app", as suggested in the assignment brief -
and it makes for a much better demo video than a terminal window.

Setup (same as run_rag.py):
    1. Copy .env.example (project root) to .env and add your Groq API key.
    2. pip install -r requirements.txt   (or use the conda environment)

Run it:
    streamlit run app.py

This opens a page in your browser at http://localhost:8501
"""

import streamlit as st
from rag_core import answer_question

st.set_page_config(page_title="Food Safety Assistant", page_icon="🥗")

st.title("🥗 Food Safety & Allergen Assistant")
st.caption(
    "Answers are grounded in official Australian sources (FSANZ, Better "
    "Health Channel, SA Health, NSW Food Authority). This is a student "
    "prototype, not medical advice."
)

# st.session_state keeps the chat history across reruns (Streamlit reruns
# the whole script on every interaction, so without this the chat would
# reset every time you asked a new question).
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, sources)

question = st.chat_input("Ask a food safety or allergen question...")

if question:
    with st.spinner("Searching the knowledge base and generating an answer..."):
        try:
            answer, passages = answer_question(question, top_k=3)
            st.session_state.history.append((question, answer, passages))
        except RuntimeError as e:
            st.error(str(e))  # e.g. missing GROQ_API_KEY

for q, a, sources in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("Show sources"):
            for i, p in enumerate(sources, 1):
                st.markdown(f"**[{i}] passage `{p['passage_id']}`**")
                st.caption(p["text"])
