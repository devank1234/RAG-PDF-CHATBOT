import os   
from datetime import datetime

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Multiple PDFs")
st.write(
    "Upload PDF files and ask questions using "
    "Hugging Face embeddings, FAISS and a local Ollama LLM."
)


# =========================================================
# SESSION STATE
# =========================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# =========================================================
# HUGGING FACE EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# OLLAMA LLM
# =========================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0.3
    )


# Load models
try:

    embedding_model = load_embedding_model()
    llm = load_llm()

except Exception as e:

    st.error(f"Error loading models: {e}")
    st.stop()


# =========================================================
# EXTRACT TEXT FROM PDFs
# =========================================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        except Exception as e:

            st.warning(
                f"Could not read {pdf.name}: {e}"
            )

    return text


# =========================================================
# TEXT CHUNKING
# =========================================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)


# =========================================================
# CREATE FAISS VECTOR STORE
# =========================================================

def get_vector_store(text_chunks):

    if not text_chunks:
        return None

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embedding_model
    )

    return vector_store


# =========================================================
# RAG PROMPT
# =========================================================

prompt_template = """
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information provided
in the context.

If the answer is not available in the context, say:

"Answer is not available in the provided PDF context."

Do not make up information.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question):

    vector_store = st.session_state.vector_store

    if vector_store is None:

        return "Please upload and process PDF files first."

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=4
    )

    if not docs:

        return (
            "Answer is not available in the "
            "provided PDF context."
        )

    # Combine retrieved documents
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Create prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # Ask Ollama
    response = llm.invoke(
        formatted_prompt
    )

    if hasattr(response, "content"):

        return response.content

    return str(response)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown(
        """
        **Embedding Model**

        `all-MiniLM-L6-v2`

        **LLM**

        `Llama 3.2 3B`

        **Vector Store**

        `FAISS`

        **Inference**

        `Ollama (Local)`
        """
    )

    st.divider()

    st.subheader("📂 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🚀 Process PDFs",
        use_container_width=True
    )

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )


# =========================================================
# RESET
# =========================================================

if reset_button:

    st.session_state.conversation_history = []
    st.session_state.vector_store = None
    st.session_state.pdf_names = []
    st.session_state.processed = False

    st.rerun()


# =========================================================
# PROCESS PDFs
# =========================================================

if process_button:

    if not pdf_docs:

        st.warning(
            "Please upload at least one PDF file."
        )

    else:

        # ---------------------------------------------
        # Extract text
        # ---------------------------------------------

        with st.spinner(
            "📖 Reading PDF files..."
        ):

            raw_text = get_pdf_text(
                pdf_docs
            )

        if not raw_text.strip():

            st.error(
                "No readable text was found in the PDFs."
            )

        else:

            # -----------------------------------------
            # Chunking
            # -----------------------------------------

            with st.spinner(
                "✂️ Splitting documents into chunks..."
            ):

                text_chunks = get_text_chunks(
                    raw_text
                )

            st.info(
                f"Created {len(text_chunks)} text chunks."
            )

            # -----------------------------------------
            # Embeddings + FAISS
            # -----------------------------------------

            with st.spinner(
                "🧠 Creating embeddings and FAISS index..."
            ):

                vector_store = get_vector_store(
                    text_chunks
                )

            if vector_store:

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.pdf_names = [
                    pdf.name
                    for pdf in pdf_docs
                ]

                st.session_state.processed = True

                st.session_state.conversation_history = []

                st.success(
                    "✅ PDFs processed successfully!"
                )


# =========================================================
# PROCESSING STATUS
# =========================================================

if st.session_state.processed:

    st.success(
        "🟢 PDF knowledge base is ready."
    )

    st.caption(
        "Processed PDFs: "
        + ", ".join(
            st.session_state.pdf_names
        )
    )


# =========================================================
# CHAT INPUT
# =========================================================

st.subheader("💬 Ask a Question")

user_question = st.chat_input(
    "Ask something about your PDF..."
)


# =========================================================
# ANSWER QUESTION
# =========================================================

if user_question:

    if not st.session_state.processed:

        st.warning(
            "Please upload and process your PDFs first."
        )

    else:

        # User message
        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        # Assistant response
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching PDF and generating answer..."
            ):

                try:

                    answer = generate_answer(
                        user_question
                    )

                    st.markdown(
                        answer
                    )

                except Exception as e:

                    answer = (
                        f"An error occurred: {e}"
                    )

                    st.error(
                        answer
                    )

        # Save history
        st.session_state.conversation_history.append(
            (
                user_question,
                answer,
                "Ollama - Llama 3.2 3B",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                ", ".join(
                    st.session_state.pdf_names
                )
            )
        )


# =========================================================
# CONVERSATION HISTORY
# =========================================================

if st.session_state.conversation_history:

    st.divider()

    st.subheader("📝 Conversation History")

    for (
        question,
        answer,
        model_name,
        timestamp,
        pdf_name
    ) in reversed(
        st.session_state.conversation_history
    ):

        with st.expander(
            f"❓ {question}"
        ):

            st.markdown(
                f"**Answer:** {answer}"
            )

            st.caption(
                f"Model: {model_name}"
            )

            st.caption(
                f"Time: {timestamp}"
            )

            st.caption(
                f"PDF: {pdf_name}"
            )


# =========================================================
# DOWNLOAD CHAT HISTORY
# =========================================================

if st.session_state.conversation_history:

    df = pd.DataFrame(
        st.session_state.conversation_history,
        columns=[
            "Question",
            "Answer",
            "Model",
            "Timestamp",
            "PDF Name"
        ]
    )

    csv = df.to_csv(
        index=False
    )

    st.sidebar.download_button(
        label="⬇️ Download Chat History",
        data=csv,
        file_name="conversation_history.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Streamlit • LangChain • Hugging Face • FAISS • Ollama"
)     