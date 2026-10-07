import gc
import shutil
import uuid
from pathlib import Path

import streamlit as st
import yaml

from src.chunking import Chunking
from src.embedding import EmbeddingManager
from src.generation import RAGRetriever
from src.ingestion import Ingest, SUPPORTED_EXTENSIONS
from src.vectore_store import VectorStore


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config" / "config.yml"
SESSION_ROOT = APP_DIR / "temp"


st.set_page_config(
    page_title="QPilot",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
        <style>
        .stApp,
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background: #0b1020 !important;
            color: #e5e7eb;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.95);
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #263244;
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        [data-testid="stChatMessage"] {
            background: #151e2e;
            border: 1px solid #263244;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }

        /* Style both the chat input and its inner textarea wrapper. */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: #151e2e !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"] {
            overflow: hidden;
        }

        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            caret-color: #60a5fa;
            outline: none !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1;
        }

        [data-testid="stChatInput"]:focus-within,
        [data-testid="stChatInput"]:focus-within [data-baseweb="textarea"] {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }

        [data-testid="stFileUploader"] {
            background: #151e2e;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 0.5rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #151e2e !important;
            border-color: #475569 !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #e5e7eb !important;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background: #202a3b !important;
            color: #f8fafc !important;
            border-radius: 8px;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
            color: #f8fafc !important;
        }

        [data-testid="stMetric"] {
            background: #151e2e;
            border: 1px solid #263244;
            border-radius: 12px;
            padding: 0.8rem;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
            color: #e5e7eb !important;
        }

        h1, h2, h3, p, label {
            color: #e5e7eb;
        }

        .stCaption {
            color: #9ca3af;
        }

        .stButton > button {
            background: #2563eb;
            color: #ffffff;
            border: 0;
            border-radius: 10px;
            min-height: 2.6rem;
        }

        .stButton > button:hover {
            background: #1d4ed8;
            color: #ffffff;
            border: 0;
        }

        .qpilot-credit {
            position: fixed;
            top: 4rem;
            right: 1.5rem;
            z-index: 1001;
            display: flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.65rem 1rem;
            border: 1px solid #60a5fa;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
            color: #ffffff !important;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }

        .qpilot-credit strong {
            color: #ffffff !important;
            font-weight: 700;
        }

        @media (max-width: 640px) {
            .qpilot-credit {
                top: 3.5rem;
                right: 0.75rem;
                padding: 0.55rem 0.8rem;
                font-size: 0.78rem;
            }
        }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="qpilot-credit">A Product by <strong>Hariharan</strong></div>',
    unsafe_allow_html=True,
)


with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_manager(model_name: str) -> EmbeddingManager:
    return EmbeddingManager(model_name)


def start_session() -> None:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    session_id = uuid.uuid4().hex
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir()

    uploads_dir = session_dir / "uploads"
    uploads_dir.mkdir()

    vector_store = VectorStore(
        collection_name=CONFIG["vector_store"]["collection_name"],
        persistant_dir=str(session_dir / "vector_store"),
    )
    embedding_manager = load_embedding_manager(CONFIG["model_name"])

    st.session_state.session_id = session_id
    st.session_state.session_dir = str(session_dir)
    st.session_state.uploads_dir = str(uploads_dir)
    st.session_state.vector_store = vector_store
    st.session_state.embedding_manager = embedding_manager
    st.session_state.retriever = RAGRetriever(vector_store, embedding_manager)
    st.session_state.messages = []
    st.session_state.indexed_chunks = 0


def delete_session_data() -> None:
    session_dir = Path(st.session_state.session_dir)
    vector_store = st.session_state.vector_store
    client = vector_store.client

    # The whole session directory is being removed, so close Chroma first
    # to release its file handles before deleting it.
    vector_store.collection = None
    client.close()
    vector_store.client = None

    st.session_state.clear()
    del vector_store, client
    gc.collect()

    shutil.rmtree(session_dir)


if "session_id" not in st.session_state:
    try:
        start_session()
    except Exception as error:
        st.error(f"Could not start a QPilot session: {error}")
        st.stop()


with st.sidebar:
    st.title("🤖 QPilot")
    st.caption("Upload documents and click Finish upload to start asking questions.")
    st.caption("Supported formats: PDF, TXT, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV, and JSON.")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="The files are stored temporarily for this session.",
        key=f"uploaded_files_{st.session_state.session_id}",
    )
    if uploaded_files:
        st.markdown("**Selected files**")
        for uploaded_file in uploaded_files:
            size_kb = uploaded_file.size / 1024
            st.caption(f"📄 {uploaded_file.name} · {size_kb:.1f} KB")
    if st.button("Finish upload", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one supported document first.")
        else:
            try:
                with st.spinner("Reading and indexing your documents..."):
                    uploads_dir = Path(st.session_state.uploads_dir)
                    documents = Ingest().process_uploaded_files(
                        uploaded_files,
                        uploads_dir,
                    )

                    if not documents:
                        st.warning("No readable text was found in those files.")
                    else:
                        chunks = Chunking().split(documents)

                        if not chunks:
                            st.warning("No text chunks were produced.")
                        else:
                            texts = [chunk.page_content for chunk in chunks]
                            embeddings = (
                                st.session_state.embedding_manager
                                .generate_embeddings(texts)
                            )

                            vector_store = st.session_state.vector_store
                            vector_store.client.delete_collection(
                                name=vector_store.collection_name
                            )
                            vector_store.collection = (
                                vector_store.client.get_or_create_collection(
                                    name=vector_store.collection_name,
                                    metadata={
                                        "description": "QPilot session documents"
                                    },
                                )
                            )
                            vector_store.add_documents(chunks, embeddings)
                            st.session_state.indexed_chunks = (
                                vector_store.collection.count()
                            )

                if st.session_state.indexed_chunks:
                    st.success("Documents are ready. You can now ask questions.")

            except Exception as error:
                st.error(f"Could not index the uploaded documents: {error}")

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Session files and vector data are stored temporarily.")

    if st.button("End session & delete data", use_container_width=True):
        try:
            delete_session_data()
        except Exception as error:
            st.error(f"Could not fully delete session data: {error}")
            st.stop()

        st.rerun()


st.title("🤖 QPilot")
st.caption("Your private document Q&A assistant")

if st.session_state.indexed_chunks == 0:
    with st.chat_message("assistant"):
        st.markdown(
            "Upload a supported document in the sidebar and click **Finish upload** to get started."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=st.session_state.indexed_chunks == 0,
):
    if st.session_state.indexed_chunks == 0:
        st.warning("Upload and index documents before asking a question.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    answer = st.session_state.retriever.generate(
                        prompt,
                        st.session_state.retriever,
                        top_k=3,
                    )
                except Exception as error:
                    answer = f"Sorry, I couldn't generate an answer: {error}"

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )