import gc
import shutil
import uuid
from pathlib import Path

import streamlit as st
import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.chunking import Chunking
from src.embedding import EmbeddingManager
from src.generation import RAGRetriever
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
    </style>
    """,
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

    vector_store.client.delete_collection(name=vector_store.collection_name)

    st.session_state.clear()
    del vector_store
    gc.collect()

    shutil.rmtree(session_dir)


def load_uploaded_documents(uploaded_files, uploads_dir: Path):
    shutil.rmtree(uploads_dir, ignore_errors=True)
    uploads_dir.mkdir(parents=True)

    documents = []

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        file_path = uploads_dir / file_name
        file_path.write_bytes(uploaded_file.getvalue())

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(file_path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(
                str(file_path),
                autodetect_encoding=True,
            ).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        for document in loaded:
            document.metadata["source_file"] = file_name
            document.metadata["file_type"] = suffix.lstrip(".")

        documents.extend(loaded)

    return documents


if "session_id" not in st.session_state:
    try:
        start_session()
    except Exception as error:
        st.error(f"Could not start a QPilot session: {error}")
        st.stop()


with st.sidebar:
    st.title("🤖 QPilot")
    st.caption("Upload documents and click Index uploaded documents, then ask questions about them.")

    uploaded_files = st.file_uploader(
        "Add PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="The files are stored temporarily for this session.",
    )
    if uploaded_files:
        st.markdown("**Selected files**")
        for uploaded_file in uploaded_files:
            size_kb = uploaded_file.size / 1024
            st.caption(f"📄 {uploaded_file.name} · {size_kb:.1f} KB")
    if st.button("Index uploaded documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            try:
                with st.spinner("Reading and indexing your documents..."):
                    uploads_dir = Path(st.session_state.uploads_dir)
                    documents = load_uploaded_documents(
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
                    st.success(
                        f"Indexed {st.session_state.indexed_chunks} chunks."
                    )

            except Exception as error:
                st.error(f"Could not index the uploaded documents: {error}")

    st.metric("Indexed chunks", st.session_state.indexed_chunks)

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Session files and vector data are stored temporarily.")

    if st.button("End session & delete data", use_container_width=True):
        try:
            delete_session_data()
            st.success("Session files and vector data were deleted.")
            st.stop()
        except Exception as error:
            st.error(f"Could not fully delete session data: {error}")


st.title("🤖 QPilot")
st.caption("Your private document Q&A assistant")

if st.session_state.indexed_chunks == 0:
    with st.chat_message("assistant"):
        st.markdown(
            "Upload PDF or TXT files in the sidebar and index them to get started."
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