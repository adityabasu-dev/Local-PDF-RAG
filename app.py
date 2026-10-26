
import hashlib
import streamlit as st
TOP_K = 3
from rag import (
    process_uploaded_pdf,
    retrieve,
    generate_answer
)

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions about it")


# =====================================
# INITIALIZE SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        accept_multiple_files=False
    )

    st.divider()


    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("🧠 LLM: Llama 3.2 (Local)")
    st.caption("🔎 Embeddings: EmbeddingGemma (Local)")


# =====================================
# PROCESS UPLOADED PDF
# =====================================

if uploaded_file is not None:

    # Hash the file to detect changes
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:

        # Clear the old document state
        st.session_state.embeddings = None
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.pdf_hash = None

        try:
            with st.spinner("Processing and embedding PDF..."):

                uploaded_file.seek(0)

                embeddings, chunks = process_uploaded_pdf(
                    uploaded_file,
                    doc_id=current_hash[:16]
                )

            st.session_state.embeddings = embeddings
            st.session_state.chunks = chunks
            st.session_state.pdf_hash = current_hash

            st.success(
                f"PDF processed! Created {len(chunks)} chunks."
            )

        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            st.stop()

else:
    # Clear document when uploader is emptied
    st.session_state.embeddings = None
    st.session_state.chunks = []
    st.session_state.pdf_hash = None
    st.session_state.messages = []


# =====================================
# DOCUMENT STATUS
# =====================================

if st.session_state.embeddings is None:
    st.info("👈 Upload a PDF from the sidebar to begin.")
    st.stop()

st.caption(
    f"📄 {uploaded_file.name} | "
    f"{len(st.session_state.chunks)} chunks indexed"
)


# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
        message["role"] == "assistant"
        and message.get("sources")
        ):
        
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.write(
                        f"Page {source['page']} | "
                        f"Score: {source['score']:.4f}"
                    )
                    st.write(source["text"])


# =====================================
# CHAT INPUT
# =====================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


# =====================================
# RETRIEVAL + GENERATION
# =====================================

if question and question.strip():

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        try:
            # Step 1: Retrieve relevant chunks
            with st.spinner("Searching document..."):
                results = retrieve(
                    question=question,
                    embeddings=st.session_state.embeddings,
                    chunks=st.session_state.chunks,
                    top_k=TOP_K
                )

            # Step 2: Generate answer using local Ollama
            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    question=question,
                    retrieved_chunks=results
                )

            # Step 3: Display answer
            st.markdown(answer)

            # Step 4: Display sources
            if results:
                with st.expander("📖 Sources"):
                    for source in results:
                        st.markdown(
                        f"**Page {source['page']}** "
                        f"— Similarity: {source['score']:.4f}"
                        )
                        st.write(source["text"])
                        st.divider()
            # Step 5: Save answer to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": results
            })

        except Exception as e:
            st.error(f"Error: {e}")