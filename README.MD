PDF RAG Chatbot using LlamaIndex and Ollama
Project Overview
it is a upgrade of common-medicines-project-v1

This project is an initial-stage Retrieval-Augmented Generation (RAG) application that answers questions based on information available in a PDF document.

The project uses LlamaIndex for document processing, chunking, embeddings, indexing, and retrieval.

Ollama is used to run the Llama 3.2 language model and the Nomic embedding model locally.



RAG Workflow
PDF Document
     ↓
PDF Reader
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embeddings
     ↓
Vector Store Index
     ↓
Similarity Search
     ↓
Llama 3.2
     ↓
   Answer 


Technologies Used

Python
LlamaIndex
Ollama
Llama 3.2
Nomic Embed Text
PDFReader
SentenceSplitter
VectorStoreIndex
Retrieval-Augmented Generation (RAG)

Added practical features to your sidebar layout so you don't have to manage files or logs inside your Windows file explorer

Drag-and-Drop Uploader (st.file_uploader): 
Added a block that lets you upload a new PDF right from your browser. 
The code reads the raw file bytes, saves it automatically to your local data/ folder, 
clears the engine's old text memory cache, and updates your app.


Clear Chat History Action Button (st.button): 
Added a reset trigger that wipes out the session memory arrays and cleanly sets Chitti's greeting phrase 
back to its baseline default configuration.

Export Chat Log Data Button (st.download_button): 
Built a text compiler script that loops through your active conversation history,
organizes it into standard readable Markdown (.md) format, and lets you download the file directly to your PC.

"Successfully personalized assistant identity to Chitti across UI, prompts, and data exporters"