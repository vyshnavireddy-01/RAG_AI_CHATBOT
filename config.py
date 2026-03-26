MONGO_URI = "mongodb://localhost:27017"

DB_NAME = "rag_database"

# OLD collection (keep temporarily so app does not break)
DOCUMENT_COLLECTION = "documents"

# NEW collections for correct RAG architecture
PAGES_COLLECTION = "pages"
CHUNKS_COLLECTION = "chunks"
CHAT_COLLECTION = "chat_logs"

FAISS_INDEX_PATH = "vector_store/index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"

CLAUDE_MODEL = "claude-3-haiku-20240307"