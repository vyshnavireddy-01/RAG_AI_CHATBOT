import os
import faiss
import numpy as np
import config
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =========================
# Load embedding model once
# =========================
model = SentenceTransformer(config.MODEL_NAME)

# =========================
# FAISS L2 distance threshold
# Lower L2 distance = more similar
# Chunks with distance > this value are ignored
# =========================
L2_DISTANCE_THRESHOLD = 1.8   # tune: lower = stricter, higher = more results

# =========================
# Load FAISS index once
# =========================
index = None

def load_index():
    global index

    if index is None:
        if not os.path.exists(config.FAISS_INDEX_PATH):
            raise ValueError("FAISS index not found. Please run ingestion first.")
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        logger.info("[Retriever] FAISS index loaded — total vectors: %d", index.ntotal)

    return index
def reload_index():
    """
    Force-reload the FAISS index from disk.
    Called automatically by auto_updater after re-ingestion completes.
    Bot immediately uses new index without restart.
    """
    global index
    index = None
    load_index()
    logger.info("[Retriever] ✅ FAISS index hot-reloaded successfully.")
 

# =========================
# Query expansion
# =========================
def expand_query(question):
    q = question.lower()

    ABBREVIATIONS = {
        "gmb"  : "google my business",
        "seo"  : "search engine optimization",
        "smm"  : "social media marketing",
        "ppc"  : "pay per click advertising",
        "ai"   : "artificial intelligence",
        "ml"   : "machine learning",
        "ceo"  : "chief executive officer founder",
        "cfo"  : "chief financial officer",
        "cto"  : "chief technology officer",
    }

    for short, full in ABBREVIATIONS.items():
        if short in q:
            question += " " + full

    return question


# =========================
# Search — returns filtered chunk_id list
# =========================
def search(question, k=20):
    idx = load_index()

    question = expand_query(question)

    question_embedding = np.array(
        model.encode([question])
    ).astype("float32")

    D, I = idx.search(question_embedding, k)

    distances = D[0]
    indexes   = I[0]

    # ── Log every score so you can debug threshold issues ──
    logger.info("[Retriever] Query: '%s'", question)
    for rank, (chunk_id, dist) in enumerate(zip(indexes, distances), start=1):
        logger.info(
            "  Rank %d | chunk_id: %d | L2 dist: %.4f | pass: %s",
            rank, chunk_id, dist, dist <= L2_DISTANCE_THRESHOLD
        )

    # ── Filter ──
    filtered = [
        int(chunk_id)
        for chunk_id, dist in zip(indexes, distances)
        if chunk_id != -1 and dist <= L2_DISTANCE_THRESHOLD
    ]

    logger.info("[Retriever] %d/%d chunks passed for query: '%s'",
                len(filtered), k, question)

    return filtered