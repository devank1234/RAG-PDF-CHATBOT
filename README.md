# 📚 Chat with Multiple PDFs — RAG Q&A

A Retrieval-Augmented Generation (RAG) based PDF Question Answering application that allows users to upload multiple PDF documents and ask questions about their content.

The application uses **Hugging Face Sentence Transformers** for generating embeddings, **FAISS** for semantic similarity search, and **Llama 3.2 3B through Ollama** for generating answers from the retrieved PDF context.

---

## 🚀 Features

- 📄 Upload multiple PDF files
- 🔍 Ask questions about uploaded documents
- ✂️ Automatic text extraction and chunking
- 🧠 Hugging Face-based text embeddings
- ⚡ FAISS vector similarity search
- 🤖 Local Llama 3.2 3B LLM using Ollama
- 💬 Interactive Streamlit chat interface
- 📝 Conversation history
- 📥 Download conversation history as CSV
- 🔄 Reset the current knowledge base
- 🔒 Local LLM inference without external LLM API credits

---

## 🏗️ Architecture

```text
                PDF Documents
                      │
                      ▼
                 PyPDF2
                      │
                      ▼
              Text Extraction
                      │
                      ▼
        Recursive Character Splitter
                      │
                      ▼
            Text Chunks
                      │
                      ▼
       Hugging Face Embeddings
       all-MiniLM-L6-v2
                      │
                      ▼
                  FAISS
             Vector Store
                      │
                      │
                User Question
                      │
                      ▼
            Query Embedding
                      │
                      ▼
          Similarity Search
             Top 4 Chunks
                      │
                      ▼
             Retrieved Context
                      │
                      ▼
          RAG Prompt Template
                      │
                      ▼
          Llama 3.2 3B
             via Ollama
                      │
                      ▼
                  Answer

