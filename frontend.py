import streamlit as st
from main import Agentic_RAG
import os

st.set_page_config(page_title="Document QNA Chatbot!", page_icon="🤖")
st.title("Document QNA Chatbot!")

BASE_DOCS_DIR = "docs"
os.makedirs(BASE_DOCS_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = None


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    save_path = os.path.join(BASE_DOCS_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Document uploaded successfully!")

    if st.session_state.rag is None:
        with st.spinner("Indexing document..."):
            rag = Agentic_RAG(path=save_path)
            rag.ingest_documents()
            st.session_state.rag = rag


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if st.session_state.rag:
    user_input = st.chat_input("Ask a question from the document...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)


        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag.output_result(user_input)
                st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
