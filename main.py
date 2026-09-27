import os
import streamlit as st
from llama_index.readers.file import PyMuPDFReader, ImageReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Medical Reference AI Chatbot", page_icon="🩺", layout="centered")

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- INITIALIZE GLOBAL CHAT HISTORY EARLY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am Chiti, your medical reference assistant. How can I help you navigate your documentation today?"}
    ]


# --- INITIALIZATION FUNCTION --
@st.cache_resource
def initialize_rag_system():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        return None

    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".png": ImageReader(),
        ".jpg": ImageReader(),
        ".jpeg": ImageReader()
    }

    # Load local documents
    documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()

    if "GROQ_API_KEY" not in st.secrets:
        st.error("❌ Missing GROQ_API_KEY in Streamlit Secrets!")
        return None

    secret_key = st.secrets["GROQ_API_KEY"]

    # Configure Groq Cloud LLM (Using a valid Groq model name)
    Settings.llm = Groq(model="llama3-8b-8192", api_key=secret_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512
    Settings.context_window = 2048

    index = VectorStoreIndex.from_documents(documents)

    system_prompt = (
        "You are Chiti, a professional medical reference assistant. Your job is to answer queries strictly "
        "using the provided reference documentation. Always provide accurate, structured information. "
        "Maintain a factual, calm tone. Always include a brief standard reminder to consult a healthcare provider."
    )
    Settings.system_prompt = system_prompt

    # Returns the chat engine (handles conversation history natively)
    return index.as_chat_engine(chat_mode="context", system_prompt=system_prompt, similarity_top_k=2)


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("📋 Document Management")
    st.caption("Manage your configuration parameters securely.")
    st.markdown("---")
    st.header("⚙️ Chat Actions")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am Chiti, your medical reference assistant. How can I help you navigate your documentation today?"}
        ]
        st.rerun()

    if len(st.session_state.messages) > 1:
        chat_text = "# Medical Reference Chat History\n\n"
        for msg in st.session_state.messages:
            role_name = "Chiti" if msg["role"] == "assistant" else "User"
            chat_text += f"**{role_name}:** {msg['content']}\n\n"

        st.download_button(
            label="💾 Export Chat Log (.md)",
            data=chat_text,
            file_name="medical_chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- MAIN WEB UI ---
st.title("🩺 Chiti - Medical Reference Assistant")
st.subheader("RAG Chatbot Pipeline (LlamaIndex + Groq Cloud)")

# Load the RAG engine
chat_engine = initialize_rag_system()

# Render persistent historical conversation bubbles
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if chat_engine is None:
    st.info("👋 Welcome! Please upload some reference documents using the ➕ attachment bar below to start chatting.")

# --- FLOATING CONTROLS FOOTER ---
input_container = st.container()
with input_container:
    btn_col, chat_col = st.columns([1, 12])

    with btn_col:
        with st.popover("➕", help="Click to add attachments"):
            uploaded_files = st.file_uploader(
                "Upload files:",
                type=["pdf", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if uploaded_files:
                new_file_added = False
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    if not os.path.exists(file_path):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.toast(f"📥 Indexed: {uploaded_file.name}", icon="✅")
                        new_file_added = True
                if new_file_added:
                    st.cache_resource.clear()
                    st.rerun()

    with chat_col:
        # Single execution entry point for user text submission
        user_query = st.chat_input("Type your question about medications...", disabled=(chat_engine is None))

# --- CORE RESPONSE EXECUTION LAYER ---
if user_query and chat_engine:
    # 1. Immediately append and display the user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Call the cloud model engine safely
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use LlamaIndex ChatEngine's native chat feature (remembers context)
                response = chat_engine.chat(user_query)
                response_text = str(response)

                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    st.rerun()
