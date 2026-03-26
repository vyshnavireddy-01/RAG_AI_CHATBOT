"""
auto_updater.py
Ealkay Consulting — Auto Sitemap Change Detector & Re-Ingestion Engine

HOW IT WORKS:
1. On Flask startup → immediately checks sitemap vs DB
2. Every 24 hours → compares sitemap URLs with what's stored in DB
3. If URLs changed (new pages added / old pages removed) → triggers full re-ingestion
4. Bot immediately answers with updated data — zero manual work needed

INSTALL REQUIRED:
    pip install apscheduler
"""

import logging
import hashlib
import requests
import xml.etree.ElementTree as ET
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler

import config

logger = logging.getLogger(__name__)

# ─── Sitemap URL ──────────────────────────────────────────────────────────────
SITEMAP_URL = "https://www.ealkay.com/sitemap.xml"

# ─── MongoDB ──────────────────────────────────────────────────────────────────
_mongo_client = MongoClient(config.MONGO_URI)
_db           = _mongo_client[config.DB_NAME]
_meta_col     = _db["sitemap_meta"]     # stores last known sitemap hash
_chunks_col   = _db["chunks"]


# ─── STEP 1: Fetch current sitemap URLs ───────────────────────────────────────
def fetch_sitemap_urls(sitemap_url: str) -> list:
    """Fetch all URLs from the sitemap. Returns sorted list of URLs."""
    try:
        response = requests.get(sitemap_url, timeout=15)
        response.encoding = "utf-8"
        root      = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        urls = []
        for url in root.findall(".//ns:loc", namespace):
            link = url.text.strip()
            if any(x in link for x in ["tag", "author", "feed", "wp-json"]):
                continue
            urls.append(link)

        return sorted(urls)

    except Exception as e:
        logger.error("[AutoUpdater] Failed to fetch sitemap: %s", e)
        return []


# ─── STEP 2: Generate a fingerprint hash of the URL list ──────────────────────
def compute_hash(urls: list) -> str:
    """Create a unique hash from the sorted URL list.
    If sitemap changes (new page added/removed), hash will be different."""
    content = "\n".join(urls)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ─── STEP 3: Get last stored hash from MongoDB ────────────────────────────────
def get_stored_hash() -> str:
    """Retrieve the last saved sitemap hash from MongoDB."""
    doc = _meta_col.find_one({"_id": "sitemap_hash"})
    return doc["hash"] if doc else None


# ─── STEP 4: Save new hash to MongoDB ─────────────────────────────────────────
def save_hash(new_hash: str):
    """Save the new sitemap hash to MongoDB."""
    _meta_col.update_one(
        {"_id": "sitemap_hash"},
        {"$set": {"hash": new_hash}},
        upsert=True
    )
    logger.info("[AutoUpdater] Sitemap hash saved: %s", new_hash)


# ─── STEP 5: Run full re-ingestion ────────────────────────────────────────────
def run_ingestion():
    """
    Full re-ingestion pipeline:
    Crawl → Chunk → Embed → Store in MongoDB → Rebuild FAISS index
    """
    import numpy as np
    import faiss
    from crawler  import crawl_website
    from chunker  import chunk_text
    from embedder import create_embeddings

    logger.info("[AutoUpdater] 🔵 Starting full re-ingestion...")

    try:
        # 1. Crawl
        pages = crawl_website(SITEMAP_URL)
        logger.info("[AutoUpdater] 🟢 Crawled %d pages", len(pages))

        # 2. Chunk
        all_chunks = []
        for page in pages:
            for chunk in chunk_text(page["text"]):
                all_chunks.append({"text": chunk, "page_url": page["url"]})
        logger.info("[AutoUpdater] 🟡 Created %d chunks", len(all_chunks))

        # 3. Embed
        embeddings = create_embeddings([c["text"] for c in all_chunks])
        logger.info("[AutoUpdater] 🟣 Embeddings created")

        # 4. Store in MongoDB
        _chunks_col.delete_many({})
        for i, chunk in enumerate(all_chunks):
            _chunks_col.insert_one({
                "chunk_id": i,
                "text":     chunk["text"],
                "page_url": chunk["page_url"]
            })
        logger.info("[AutoUpdater] 🟡 MongoDB updated — %d chunks stored", len(all_chunks))

        # 5. Rebuild FAISS index
        dimension = len(embeddings[0])
        idx = faiss.IndexFlatL2(dimension)
        idx.add(np.array(embeddings).astype("float32"))
        faiss.write_index(idx, config.FAISS_INDEX_PATH)
        logger.info("[AutoUpdater] 🔴 FAISS index rebuilt at: %s", config.FAISS_INDEX_PATH)

        # 6. Hot-reload FAISS in retriever so bot uses new index immediately
        from retriever import reload_index
        reload_index()
        logger.info("[AutoUpdater] ✅ Re-ingestion complete — bot is now using updated data!")

    except Exception as e:
        logger.error("[AutoUpdater] ❌ Re-ingestion failed: %s", e)


# ─── MAIN CHECK FUNCTION (runs every 24 hours) ────────────────────────────────
def check_and_update():
    """
    Core logic — runs on startup and every 24 hours:
    1. Fetch current sitemap URLs
    2. Compute hash
    3. Compare with stored hash
    4. If different → re-ingest. If same → do nothing.
    """
    logger.info("[AutoUpdater] 🔍 Checking sitemap for changes...")

    current_urls = fetch_sitemap_urls(SITEMAP_URL)

    if not current_urls:
        logger.warning("[AutoUpdater] Sitemap fetch returned 0 URLs — skipping check.")
        return

    current_hash = compute_hash(current_urls)
    stored_hash  = get_stored_hash()

    logger.info("[AutoUpdater] Current hash : %s", current_hash)
    logger.info("[AutoUpdater] Stored hash  : %s", stored_hash or "None (first run)")

    if current_hash == stored_hash:
        logger.info("[AutoUpdater] ✅ No changes detected — DB is up to date.")
        return

    # Sitemap changed — find what's new or removed
    stored_urls = []
    if stored_hash:
        stored_doc = _meta_col.find_one({"_id": "sitemap_urls"})
        stored_urls = stored_doc["urls"] if stored_doc else []

    new_urls     = set(current_urls) - set(stored_urls)
    removed_urls = set(stored_urls)  - set(current_urls)

    if new_urls:
        logger.info("[AutoUpdater] 🆕 New pages detected (%d): %s", len(new_urls), list(new_urls)[:5])
    if removed_urls:
        logger.info("[AutoUpdater] 🗑️  Removed pages detected (%d): %s", len(removed_urls), list(removed_urls)[:5])

    logger.info("[AutoUpdater] 🔄 Sitemap changed — triggering re-ingestion...")

    # Save new hash + URL list before ingestion
    save_hash(current_hash)
    _meta_col.update_one(
        {"_id": "sitemap_urls"},
        {"$set": {"urls": current_urls}},
        upsert=True
    )

    # Run full re-ingestion
    run_ingestion()


# ─── SCHEDULER STARTER (called from app.py once) ─────────────────────────────
def start_auto_updater():
    """
    Call this once in app.py at startup.
    - Runs check_and_update() immediately on startup
    - Then schedules it every 24 hours automatically
    """
    logger.info("[AutoUpdater] 🚀 Starting auto-updater — checks every 24 hours")

    # Run immediately on startup
    check_and_update()

    # Schedule every 24 hours
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func             = check_and_update,
        trigger          = "interval",
        hours            = 24,
        id               = "sitemap_check",
        replace_existing = True
    )
    scheduler.start()
    logger.info("[AutoUpdater] ⏰ Scheduler running — next check in 24 hours")