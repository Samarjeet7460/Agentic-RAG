import streamlit as st
from main import Agentic_RAG
import os

st.title("Agentic RAG Chatbot")

BASE_DOCS_DIR = "docs"
os.makedirs(BASE_DOCS_DIR, exist_ok=True)


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    save_path = os.path.join(BASE_DOCS_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved at {save_path}")

    if "rag" not in st.session_state:
        rag = Agentic_RAG(path=save_path)
        rag.ingest_documents()
        rag.load_vectorstore()
        st.session_state.rag = rag
        st.success("Vector store ready!")

if "rag" in st.session_state:
    chat_input = st.chat_input("Ask a question from the document...")

    if chat_input:
        with st.chat_message("user"):
            st.write(chat_input)

        with st.chat_message("assistant"):
            answer = st.session_state.rag.output_result(chat_input)
            st.write(answer)
