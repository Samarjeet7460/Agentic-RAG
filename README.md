# Agentic RAG using LangGraph

This project is an Agentic Retrieval-Augmented Generation (RAG) system that enables users to upload documents and ask intelligent, context-aware questions about their content. Unlike traditional RAG pipelines, this system uses an agent-driven iterative retrieval approach to determine whether the retrieved context is sufficient before generating a final answer.

The backend is built using LangGraph to orchestrate agent workflows, LangChain for LLM integration, and FAISS for efficient vector-based semantic search. A Streamlit-based frontend provides an interactive chat interface for document upload and conversational querying.

## Architecture Overview

Document Ingestion

PDF → Chunking → Embedding → FAISS Vector Store

Agentic Retrieval Loop

Retrieve relevant chunks

LLM checks context sufficiency

Refine query and retry if required

Answer Generation

Final response generated strictly from retrieved document context

Frontend

Streamlit UI for file upload and chat-based interaction

## Tech Stack

Python

LangChain

LangGraph

Groq (LLaMA Models)

FAISS

HuggingFace Sentence Transformers

Streamlit

dotenv

## Install Dependencies : 
### pip install uv
### uv init --name "agentic-rag"
### uv venv
### uv pip install -r pyproject.toml


## How to run : 
### streamlit run frontend.py

## NOTE: Add your groq api key in .env file
## how to get groq api key for free: https://console.groq.com/home 