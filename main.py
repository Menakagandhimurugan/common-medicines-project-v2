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


# --- INITIALIZATION FUNCTION --
@st.cache_resource
def initialize_rag_system():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        return None
    # Instruct Chitti to utilize PyMuPDF's scanning layer for any PDF files 👇
    file_extractor = file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".png": ImageReader(),
        ".jpg": ImageReader(),
        ".jpeg": ImageReader()
    }

    # Load your local data folder files using the custom file extractor framework
    documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()

    if "GROQ_API_KEY" not in st.secrets:
        st.error("❌ Missing GROQ_API_KEY in secrets configuration!")
        return None

    secret_key = st.secrets["GROQ_API_KEY"]
    # Configure Groq Cloud LLM and HuggingFace Embeddings
    Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=secret_key)


    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.chunk_size = 512  # Cuts document segments into smaller sizes
    Settings.context_window = 2048  # Limits total text memory requested from engine

    index = VectorStoreIndex.from_documents(documents)

    # System prompt: Configures Chitti's custom medical assistant persona
    system_prompt = (
        "You are Chitti, a professional medical reference assistant. Your job is to answer queries strictly "
        "using the provided reference documentation. Always provide accurate, structured information "
        "including generic and brand names when available. Maintain a factual, calm tone. Always "
        "include a brief standard reminder to consult a healthcare provider for any actual clinical decisions."
    )
    Settings.system_prompt = system_prompt

    # Returns conversational Chat Engine tracking your identity prompt details
    return index.as_chat_engine(chat_mode="context", system_prompt=system_prompt, similarity_top_k=2)


# --- SIDEBAR: CLEAN MANAGEMENT & RESET CONTROLS ---
with st.sidebar:
    st.header("📋 Document Management")
    st.caption("Manage your configuration parameters securely.")

    st.markdown("---")
    st.header("⚙️ Chat Actions")

    # 1. Clear Chat History Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am Chitti, your local medical reference assistant. How can I help you navigate your documentation today?"}
        ]
        st.rerun()

    # 2. Export Chat History to Markdown File
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        chat_text = "# Medical Reference Chat History\n\n"
        for msg in st.session_state.messages:
            role_name = "Chitti" if msg["role"] == "assistant" else "User"
            chat_text += f"**{role_name}:** {msg['content']}\n\n"

        st.download_button(
            label="💾 Export Chat Log (.md)",
            data=chat_text,
            file_name="medical_chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- MAIN WEB UI INTERFACE ---
st.title("🩺 Chitti - Medical Reference Assistant")
st.subheader("RAG Chatbot Pipeline (LlamaIndex + Groq Cloud)")
st.caption("Analyzing your local medical databases completely offline.")

# Initialize or grab the conversational engine
query_engine = initialize_rag_system()

if query_engine is None:
    st.info("👋 Welcome! Please upload some text or PDF documents using the attachment button below to start chatting.")
else:
    # Initialize message session state history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am Chitti, your local medical reference assistant. How can I help you navigate your documentation today?"}
        ]

    # Render chat history visual bubbles
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- CONVERSATIONAL ATTACHMENT ACTION BAR (LIKE CHATGPT/GEMINI) ---
# Creates an expandable container for uploads right above the active chat timeline


# --- CHAT INPUT & CORE RESPONSE SYSTEM ---
# --- CHAT INPUT & CORE RESPONSE SYSTEM ---
# 1. Create a container locked to the bottom layout
input_container = st.container()

with input_container:
    # 2. Divide the horizontal row layout space: 1 part for the '+' button, 12 parts for the text bar
    btn_col, chat_col = st.columns([1, 12])

    user_query = None

    # 3. Left Column: The standalone plus utility toggle button
    with btn_col:
        with st.popover("➕", help="Click to add attachments"):
            uploaded_files = st.file_uploader(
                "Upload medical reference files to train Chitti:",
                type=["pdf", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True
            )
            if uploaded_files:
                new_file_added = False
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    if not os.path.exists(file_path):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.toast(f"📥 Successfully indexed: {uploaded_file.name}", icon="✅")
                        new_file_added = True
                if new_file_added:
                    st.cache_resource.clear()
                    st.rerun()

    # 4. Right Column: The clean typing field running parallel to the button
    with chat_col:
        user_query = st.chat_input("Type your question about medications...")

# 5. Process the conversational pipeline if input text is captured
if user_query:
    # Display user input bubble
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate response bubble
    with st.chat_message("assistant"):
        response_stream = query_engine.stream_chat(user_query)
        bot_response = st.write_stream(response_stream.response_gen)

    # Save final single clean response instance to session storage history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Rerun to cleanly synchronize history actions
    st.rerun()

