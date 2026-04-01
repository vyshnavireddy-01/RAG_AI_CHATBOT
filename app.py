from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import numpy as np
import faiss
import config
from crawler import crawl_website
from chunker import chunk_text
from embedder import create_embeddings
from retriever import search
import anthropic
import os
import logging
from dotenv import load_dotenv
import re
import threading
from auto_updater import start_auto_updater
import math 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

claude = anthropic.Anthropic(api_key=api_key)
app    = Flask(__name__)

client            = MongoClient(config.MONGO_URI)
db                = client[config.DB_NAME]
chunks_collection = db["chunks"]

if os.path.exists(config.FAISS_INDEX_PATH):
    index = faiss.read_index(config.FAISS_INDEX_PATH)
else:
    index = None


# ─── INGEST ───────────────────────────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "URL is required"}), 400
        url = data["url"]
        print("🔵 Crawling website...")
        pages = crawl_website(url)
        print("🟢 Chunking content...")
        all_chunks = []
        for page in pages:
            for chunk in chunk_text(page["text"]):
                all_chunks.append({"text": chunk, "page_url": page["url"]})
        print("🟣 Creating embeddings...")
        embeddings = create_embeddings([c["text"] for c in all_chunks])
        print("🟡 Storing in MongoDB...")
        chunks_collection.delete_many({})
        for i, chunk in enumerate(all_chunks):
            chunks_collection.insert_one({"chunk_id": i, "text": chunk["text"], "page_url": chunk["page_url"]})
        print("🔴 Building FAISS index...")
        dimension = len(embeddings[0])
        idx = faiss.IndexFlatL2(dimension)
        idx.add(np.array(embeddings).astype("float32"))
        faiss.write_index(idx, config.FAISS_INDEX_PATH)
        print("✅ Ingestion completed!")
        return jsonify({"message": "Ingestion successful!"})
    except Exception as e:
        logger.error("INGEST ERROR: %s", e)
        return jsonify({"error": str(e)}), 500


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fallback_response():
    return {"answer": (
        "I appreciate your question! However, this information is best discussed directly with our team.<br><br>"
        "📞 <b>Phone:</b> +91 98669 62305<br>"
        "📧 <b>Email:</b> info@ealkay.com<br>"
        "🌐 <b>Website:</b> https://www.ealkay.com/<br><br>"
        "Our team is always ready to assist you! 😊"
    )}

def out_of_scope_response():
    return {"answer": (
        "I can only assist with questions related to <b>Ealkay Consulting</b> — "
        "our services, team, office, and more.<br><br>Feel free to ask anything about Ealkay! 😊"
    )}

def clean_answer(text):
    patterns = [
        r"^(according to|based on|as per|from) (the )?(provided )?(context|information|ealkay(\.com)? website)[,.]?\s*",
        r"^the (provided )?context (mentions|states|indicates|shows)[,.]?\s*",
        r"^the (ealkay )?website (mentions|states|indicates|shows)[,.]?\s*",
    ]
    t = text.strip()
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return (t[0].upper() + t[1:]).strip() if t else t

def get_chunks_by_regex(regex, url_filter=None, limit=6):
    query = {"text": {"$regex": regex, "$options": "i"}}
    if url_filter:
        query["page_url"] = {"$regex": url_filter, "$options": "i"}
    return list(chunks_collection.find(query).limit(limit))

def build_context(chunks):
    return "\n\n---\n\n".join([c["text"] for c in chunks if "text" in c])

def ask_claude(context, question, max_tokens=400, extra=""):
    try:
        prompt = f"""You are Ealkay AI — the official assistant for Ealkay Consulting.

STRICT RULES:
1. Answer ONLY using the context below. Never use outside knowledge.
2. If the answer is not in the context, respond EXACTLY: FALLBACK
3. NEVER use any of these phrases anywhere in your answer — not at start, not in middle, not at end:
   "According to", "Based on", "The context", "The website", "Provided", 
   "From the website", "As per", "The information", "It is mentioned",
   "The Ealkay website", "From the context", "As mentioned".
4. Always give the SAME answer for the same question.
5. Keep answers SHORT — max 4 sentences or a clean bullet list.
6. Never answer questions unrelated to Ealkay — respond EXACTLY: OUT_OF_SCOPE
7. Be professional, warm, and direct.
{extra}

Context:
{context}

Question: {question}

Answer:"""

        response = claude.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        if answer == "FALLBACK":      return None
        if answer == "OUT_OF_SCOPE":  return "OUT_OF_SCOPE"
        answer = clean_answer(answer)
        bad = ["i could not find", "could not find", "not available", "no information"]
        if any(p in answer.lower() for p in bad):
            return None
        return re.sub(r'\n{3,}', '\n\n', answer).strip()
    except Exception as e:
        logger.error("Claude error: %s", e)
        return None


# ─── HOME ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("chat.html")


# ─── NORMALIZER ───────────────────────────────────────────────────────────────
def normalize(q):
    replacements = {
        r"\b4t\b":               "4T promise trust transparency timely transformation",
        r"\bgtm\b":              "go to market strategy",
        r"\bceo\b":              "chief executive officer founder Baba Kishore",
        r"\bcfo\b":              "chief financial officer",
        r"\bcto\b":              "chief technology officer",
        r"\bai\b":               "artificial intelligence",
        r"\bseo\b":              "search engine optimization",
        r"\bsmm\b":              "social media marketing",
        r"\bgmb\b":              "google my business",
        r"\bcgtmse\b":           "government backed loan MSME",
        r"\besop\b":             "employee stock ownership plan",
        r"\bmission\b":          "mission exist clarity complex business",
        r"\bvision\b":           "vision trusted partner",
        r"\bmilestone\w*\b":     "milestones journey growth innovation 2020 2021 2022",
        r"\bsuccess stor\w*\b":  "success story Prowessoft Skyshade Viruj",
        r"\btestimonial\w*\b":   "client testimonial Aravind Sekhar Mahesh",
        r"\boffice hour\w*\b":   "office hours monday friday 9.30 AM",
        r"\bworking hour\w*\b":  "office hours monday friday 9.30 AM",
        r"\baddress\b":          "office address Kukatpally Hyderabad",
        r"\blocation\b":         "office address Kukatpally Hyderabad",
        r"\bco.?founder\b":      "co-founder Lalitha Deepthi Mutta director legal",
    }
    result = q
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


# ─── INTENT HELPERS ───────────────────────────────────────────────────────────
def has(q, *keywords):
    """Return True if ANY keyword found in q."""
    return any(k in q for k in keywords)


# ─── CHAT ─────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    raw_question = data["question"].strip()
    if not raw_question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # FIX 5: Normalize — strip punctuation BUT keep question meaning intact
    q = raw_question.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)   # replace punctuation with space, not empty
    q = re.sub(r'\s+', ' ', q).strip()

    default_suggestions = [
        "Who is the CEO of Ealkay?",
        "What services does Ealkay provide?",
        "How can I contact Ealkay?",
        "What are Ealkay office hours?"
    ]

    # ── BLOGS (MOVED TO TOP FOR PRIORITY) ────────────────────────────────────
    blog_keywords = [
        "blog", "blogs",
        "article", "articles",
        "post", "posts",
        "blog on", "blog about",
        "article on", "article about",
        "do you have a blog",
        "any blog on",
        "latest blog",
        "write up"
    ]
    if has(q, *blog_keywords):
        indexes     = search(raw_question + " blog ealkay", k=20)
        blog_chunks = []
        seen_urls   = set()
        for i in indexes:
            doc = chunks_collection.find_one({"chunk_id": int(i)})
            if doc and "/Blogs/" in doc.get("page_url", ""):
                url = doc["page_url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    blog_chunks.append(doc)
            if len(blog_chunks) >= 3:
                break

        if not blog_chunks:
            return jsonify(fallback_response())

        context = "\n\n".join([f"Blog URL: {c['page_url']}\nContent: {c['text']}" for c in blog_chunks])
        answer  = ask_claude(context, raw_question, max_tokens=600,
            extra="""- Format EXACTLY like this for EACH blog:
📝 <b>Blog Title:</b> (title from content)
<b>Summary:</b> (2-3 lines about what the blog covers)
🔗 <b>Source:</b> (exact blog URL)

- List up to 3 blogs. No intro sentence.""")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": [
                "Show me more Ealkay blogs",
                "What services does Ealkay provide?",
                "Who is the CEO of Ealkay?"
            ]})
        return jsonify(fallback_response())

    # ── OUT OF SCOPE ──────────────────────────────────────────────────────────
    if has(q, "weather", "cricket", "sports", "movie", "song", "recipe",
              "stock price", "news today", "politics", "celebrity", "ipl"):
        return jsonify(out_of_scope_response())

    # ── WHAT IS EALKAY ────────────────────────────────────────────────────────
    if has(q, "what is ealkay", "about ealkay", "tell me about ealkay", "who is ealkay", "describe ealkay"):
        chunks  = get_chunks_by_regex(r"360.*business|co-engineer|strategy finance legal technology", url_filter="ealkay.com", limit=3)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=200,
            extra="- Give exactly 2-3 sentences: what Ealkay is, what they do, who they serve.")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── MISSION ───────────────────────────────────────────────────────────────
    if has(q, "mission") and not has(q, "vision"):
        chunks  = get_chunks_by_regex(r"our mission|we exist to|clarity to complex", url_filter="about", limit=3)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=200,
            extra="- State Ealkay's mission in 1-2 sentences. Start with 'Ealkay's mission is...'")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── VISION ────────────────────────────────────────────────────────────────
    if has(q, "vision") and not has(q, "mission"):
        chunks  = get_chunks_by_regex(r"our vision|most trusted|trusted partner.*businesses", url_filter="about", limit=3)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=200,
            extra="- State Ealkay's vision in 1-2 sentences. Start with 'Ealkay's vision is...'")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── MISSION + VISION ──────────────────────────────────────────────────────
    if has(q, "mission") and has(q, "vision"):
        chunks  = get_chunks_by_regex(r"our mission|our vision|we exist|most trusted", url_filter="about", limit=4)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=300,
            extra="- Give mission in 1-2 sentences then vision in 1-2 sentences. Label each clearly.")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── OUR STORY ─────────────────────────────────────────────────────────────
    if has(q, "our story", "ealkay story", "how ealkay started", "when was ealkay founded", "when founded", "founded in"):
        chunks  = get_chunks_by_regex(r"our story|founded|2020|started|began", url_filter="about", limit=4)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=250,
            extra="- Briefly describe how and when Ealkay was founded. Max 3 sentences.")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── 4T PROMISE ────────────────────────────────────────────────────────────
    if has(q, "4t", "4 t", "four t", "promise", "trust transparency", "timely delivery"):
        chunks  = get_chunks_by_regex(r"trust.*transparency.*timely.*transformation|4T|our promise|4 pillars", limit=4)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=250,
            extra="- Explain Ealkay's 4T: Trust, Transparency, Timely Delivery, Transformation. One line each.")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify({"answer": (
            "Ealkay's <b>4T Promise</b> — the four values behind every engagement:<br><br>"
            "✅ <b>Trust</b> — Advising from within, executing alongside clients<br>"
            "✅ <b>Transparency</b> — Strategy, finance, legal, and tech unified under one roof<br>"
            "✅ <b>Timely Delivery</b> — Every engagement closes with measurable business impact<br>"
            "✅ <b>Transformation</b> — Co-engineering your business future, not just consulting"
        ), "suggestions": default_suggestions})

    # ── MILESTONES ────────────────────────────────────────────────────────────
    # FIX 4: Hardcoded from website data — year-wise correct milestones
    if has(q, "milestone", "milestones", "journey", "achievements", "growth since", "since when"):
        return jsonify({"answer": (
            "🗓️ <b>Ealkay's Key Milestones:</b><br><br>"
            "📌 <b>2020</b> — Ealkay Consulting was founded, delivering strategy, finance, legal, and technology solutions<br>"
            "📌 <b>2021</b> — Began 8-year strategic partnership with Prowessoft, scaling revenue from ₹4 Cr to ₹100 Cr<br>"
            "📌 <b>2021</b> — Guided POPI through complex restructuring and business realignment<br>"
            "📌 <b>2022</b> — Delivered Skyshade's digital transformation, achieving 160% organic traffic growth<br>"
            "📌 <b>2023</b> — Executed Viruj Chematrix structured funding with 100% cost efficiency<br>"
            "📌 <b>Present</b> — $90 Million+ valuation strategy delivered across clients"
        ), "suggestions": [
            "What are Ealkay success stories?",
            "Who is the CEO of Ealkay?",
            "What services does Ealkay provide?"
        ]})

    # ── SUCCESS STORIES ───────────────────────────────────────────────────────
    if has(q, "success stor", "case stud", "client success", "results", "prowessoft", "skyshade", "viruj"):
        return jsonify({"answer": (
            "Here are Ealkay's 3 Success Stories:<br><br>"
            "🏆 <b>Prowessoft — 8-Year Strategic Partnership</b><br>"
            "Ealkay helped Prowessoft grow revenue from ₹4 Cr to ₹100 Cr through finance strategy, "
            "governance improvement, and strategic decision-making support.<br><br>"
            "🏆 <b>Skyshade — Digital Transformation</b><br>"
            "Ealkay built and branded Skyshade's social platforms from scratch, delivering a 160% boost "
            "in organic traffic and generating hundreds of leads.<br><br>"
            "🏆 <b>Viruj Chematrix — Structured Funding</b><br>"
            "Ealkay delivered structured funding with 100% cost efficiency and unmatched clarity "
            "in project execution."
        ), "suggestions": [
            "What are Ealkay client testimonials?",
            "Who is the CEO of Ealkay?",
            "What services does Ealkay provide?"
        ]})

    # ── TESTIMONIALS ──────────────────────────────────────────────────────────
    if has(q, "testimonial", "testimonials", "review", "reviews", "what clients say", "client feedback", "what do clients"):
        chunks = get_chunks_by_regex(
            r"Aravind Konda|Sekhar Noo?ri|Mahesh|Prashant.*Gowriraju|LR Lakshmi|Ajay Konda",
            url_filter="ealkay.com", limit=6
        )
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=600,
            extra="""- List actual client testimonials from the context.
- Format each as:
  "[full client quote]"
  — Client Name, Title, Company
- Include ALL clients found. Do NOT mention video testimonials or general advice.""")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": [
                "What are Ealkay success stories?",
                "Who is the CEO of Ealkay?",
                "What services does Ealkay provide?"
            ]})
        return jsonify(fallback_response())

    # ── CEO ───────────────────────────────────────────────────────────────────
    # FIX 3: Strict CEO check — only trigger for CEO/founder, NOT co-founder
    if has(q, "who is the ceo", "ceo of ealkay", "who is ceo", "baba kishore", "founder of ealkay") and not has(q, "co-founder", "cofounder", "co founder", "lalitha"):
        return jsonify({"answer": (
            "<b>Baba Kishore Mutta</b> is the Founder & CEO of Ealkay Consulting.<br><br>"
            "He is a qualified strategist from IIM Lucknow and IIM Nagpur with over two decades "
            "of experience in Corporate Finance and Business Strategy."
        ), "suggestions": default_suggestions})

    # ── CO-FOUNDER ────────────────────────────────────────────────────────────
    # FIX 3: Separate co-founder block — clearly distinct from CEO
    if has(q, "co-founder", "cofounder", "co founder", "lalitha", "lalitha deepthi", "director of ealkay"):
        return jsonify({"answer": (
            "<b>Lalitha Deepthi Mutta</b> is the Co-Founder & Director of Legal & Operations at Ealkay.<br><br>"
            "She is a qualified lawyer with over a decade of experience in Corporate Law and "
            "Real Estate across US and Indian geographies."
        ), "suggestions": default_suggestions})

    # ── TEAM ──────────────────────────────────────────────────────────────────
    if has(q, "team", "who works", "leadership", "management", "ajay konda", "ajay"):
        chunks  = get_chunks_by_regex(r"Baba Kishore|Lalitha|Ajay Konda|CGO|director|leadership", url_filter="about", limit=5)
        context = build_context(chunks)
        answer  = ask_claude(context, raw_question, max_tokens=350,
            extra="- List key Ealkay team members: name, title, one-line background each. Keep it concise.")
        if answer and answer != "OUT_OF_SCOPE":
            return jsonify({"answer": answer, "suggestions": default_suggestions})
        return jsonify(fallback_response())

    # ── CONTACT ───────────────────────────────────────────────────────────────
    if has(q, "contact", "email", "phone", "reach", "call", "whatsapp", "get in touch", "how to reach"):
        return jsonify({"answer": (
            "You can reach Ealkay through:<br><br>"
            "📧 <b>Email:</b> info@ealkay.com<br>"
            "📞 <b>Phone:</b> +91 98669 62305<br>"
            "🌐 <b>Website:</b> www.ealkay.com"
        ), "suggestions": [
            "Where is Ealkay office located?",
            "What are Ealkay office hours?",
            "What services does Ealkay provide?"
        ]})

    # ── ADDRESS ───────────────────────────────────────────────────────────────
    if has(q, "address", "location", "where is ealkay", "where are you located") and not has(q, "hour", "timing", "open", "close"):
        return jsonify({"answer": (
            "🏢 <b>Ealkay Consulting Office:</b><br><br>"
            "Suite No. 106, First Floor, Manjeera Trinity Corporate<br>"
            "JNTU – Hitech City Road, Beside Lulu Mall<br>"
            "Kukatpally, Hyderabad — Telangana 500072, India"
        ), "suggestions": [
            "What are Ealkay office hours?",
            "How can I contact Ealkay?"
        ]})

    # ── OFFICE ────────────────────────────────────────────────────────────────
    if has(q, "office hour", "working hour", "office timing", "when open", "when close",
              "open on saturday", "open on sunday", "office time", "business hour"):
        return jsonify({"answer": (
            "🕘 <b>Ealkay Office Hours:</b><br><br>"
            "📅 <b>Monday – Friday:</b> 9:30 AM – 6:00 PM<br>"
            "📅 <b>Saturday & Sunday:</b> Closed<br><br>"
            "For queries outside hours: 📧 info@ealkay.com | 📞 +91 98669 62305"
        ), "suggestions": [
            "Where is Ealkay office located?",
            "How can I contact Ealkay?"
        ]})

    # Prevent blog queries from going into services
    if not has(q, "blog", "article", "post"):

        # ── SERVICES TOP LEVEL ────────────────────────────────────────────────────
        # FIX 1: Removed "Architecture" — correct 4 services only
        if has(q, "what services", "services does ealkay", "services offered",
                  "what does ealkay offer", "ealkay services", "what do you offer",
                  "services in ealkay", "services of ealkay") and not has(q, "strategy", "finance", "legal", "technology", "tech", "digital"):
            return jsonify({"answer": (
                "Ealkay Consulting offers four core service areas:<br><br>"
                "1️⃣ <b>Strategy & GTM</b> — Growth, operational, and expansion strategies<br>"
                "2️⃣ <b>Finance & Capital Architecture</b> — Loans, CFO services, structured financing<br>"
                "3️⃣ <b>Legal & Governance</b> — Contracts, compliance, IP protection<br>"
                "4️⃣ <b>Technology & Digital Ecosystem</b> — Web development, AI chatbots, data analytics,managed Services,"
            ), "suggestions": [
                "What services does Ealkay provide in Strategy?",
                "What services does Ealkay provide in Finance?",
                "What services does Ealkay provide in Legal?",
                "What services does Ealkay provide in Technology?"
            ]})

        # ── SERVICES BY CATEGORY ─────────────────────────────────────────────────
        if has(q, "strategy", "gtm") and has(q, "service", "provide", "offer", "what", "tell", "about", "in"):
            return jsonify({"answer": (
                "<b>Ealkay — Strategy & GTM Services:</b><br><br>"
                "• <b>Growth Strategy</b> — Scaling and business growth planning<br>"
                "• <b>Operational Strategy</b> — Process efficiency and standardization<br>"
                "• <b>Expansion Strategy</b> — Market entry and business expansion<br>"
                "• <b>Business Model & Structuring</b> — Building and refining business models"
            ), "suggestions": ["What services does Ealkay provide in Finance?", "What services does Ealkay provide in Legal?"]})

        if has(q, "finance", "capital") and has(q, "service", "provide", "offer", "what", "tell", "about", "in"):
            return jsonify({"answer": (
                "<b>Ealkay — Finance & Capital Architecture Services:</b><br><br>"
                "• <b>Unsecured Business Loans</b> — Quick funding without collateral<br>"
                "• <b>CGTMSE</b> — Government-backed loans for MSMEs<br>"
                "• <b>Loan Against Property</b> — Funding secured against property<br>"
                "• <b>Term Loan</b> — Fixed tenure business loans<br>"
                "• <b>Structured Loan</b> — Customized financing solutions<br>"
                "• <b>Project Finance</b> — Funding for large-scale projects<br>"
                "• <b>Working Capital</b> — Managing daily business operations<br>"
                "• <b>Bridge Loan</b> — Short-term financing for immediate needs"
            ), "suggestions": ["What services does Ealkay provide in Legal?", "What services does Ealkay provide in Technology?"]})

        if has(q, "legal", "governance") and has(q, "service", "provide", "offer", "what", "tell", "about", "in"):
            return jsonify({"answer": (
                "<b>Ealkay — Legal & Governance Services:</b><br><br>"
                "• <b>Contracts & Commercial Agreements</b> — Drafting and managing business contracts<br>"
                "• <b>Entity Formation & Structuring</b> — Company formation and legal structuring<br>"
                "• <b>IP & Brand Protection</b> — Trademark and intellectual property protection<br>"
                "• <b>Corporate Governance & Compliance</b> — Regulatory compliance advisory<br>"
                "• <b>Real Estate Due Diligence</b> — Legal support for property transactions<br>"
                "• <b>ESOP & Equity Structuring</b> — Employee stock ownership plan advisory"
            ), "suggestions": ["What services does Ealkay provide in Technology?", "What services does Ealkay provide in Strategy?"]})

        if has(q, "technology", "tech", "digital") and has(q, "service", "services", "provide", "offer", "what"):
            return jsonify({"answer": (
                "<b>Ealkay — Technology & Digital Ecosystem Services:</b><br><br>"
                "• <b>Web Development</b> — Dynamic and responsive website development<br>"
                "• <b>AI Chatbots</b> — Intelligent conversational bots for businesses<br>"
                "• <b>Data Analytics</b> — Data-driven insights for business decisions<br>"
                "• <b>Managed Services</b> — End-to-end technology management"
            ), "suggestions": ["What services does Ealkay provide in Strategy?", "What services does Ealkay provide in Finance?"]})

        # ── FIX 2: SPECIFIC TECH TOPIC QUESTIONS (web dev, data analytics etc) ───
        if has(q, "web development", "web dev", "website development,"):
            return jsonify({"answer": (
                "Ealkay's <b>Web Development</b> service focuses on building dynamic, responsive, "
                "and high-performance websites for businesses.<br><br>"
                "The team designs and develops websites that are optimized for conversions, mobile-friendly, "
                "and aligned with the client's brand identity.<br>"
                "From landing pages to full business portals, Ealkay delivers end-to-end web solutions "
                "tailored to each client's goals."
            ), "suggestions": ["What services does Ealkay provide in Technology?", "What are AI Chatbot services?"]})

        if has(q, "data analytics", "data analysis", "analytics service"):
            return jsonify({"answer": (
                "Ealkay's <b>Data Analytics</b> service helps businesses make smarter decisions "
                "by turning raw data into actionable insights.<br><br>"
                "The team uses advanced analytics tools to track performance, identify trends, "
                "and optimize business strategies.<br>"
                "From MIS reporting to business intelligence dashboards, Ealkay provides end-to-end "
                "data solutions for growth-focused businesses."
            ), "suggestions": ["What services does Ealkay provide in Technology?", "What services does Ealkay provide in Finance?"]})

        if has(q, "ai chatbot", "chatbot service", "chatbot", "artificial intelligence chatbot","tell me about chatbot"):
            return jsonify({"answer": (
                "Ealkay's <b>AI Chatbot</b> service involves building intelligent, conversational bots "
                "that automate customer interactions for businesses.<br><br>"
                "These chatbots are trained on business-specific data to answer queries, generate leads, "
                "and provide 24/7 support without human intervention.<br>"
                "Ealkay designs and deploys custom AI chatbots tailored to each business's needs and workflow."
            ), "suggestions": ["What services does Ealkay provide in Technology?", "What is web development?"]})

        if has(q, "managed service", "managed services"):
            return jsonify({"answer": (
                "Ealkay's <b>Managed Services</b> provide end-to-end technology management for businesses, "
                "handling everything from infrastructure to digital operations.<br><br>"
                "This service allows businesses to focus on growth while Ealkay manages their tech stack, "
                "systems, and digital tools.<br>"
                "It includes monitoring, maintenance, support, and continuous optimization of all technology assets."
            ), "suggestions": ["What services does Ealkay provide in Technology?", "How can I contact Ealkay?"]})

    # ── GENERAL RAG ───────────────────────────────────────────────────────────
    try:
        expanded = normalize(q)
        indexes  = search(expanded, k=15)
    except Exception as e:
        logger.error("Search error: %s", e)
        return jsonify({"error": "Search failed"}), 500

    if not indexes:
        return jsonify(fallback_response())

    context = ""
    for i in indexes:
        doc = chunks_collection.find_one({"chunk_id": int(i)})
        if doc and "text" in doc:
            text = re.sub(r'\n?\s*\d+\.\s*', '\n', doc["text"])
            context += f"\nSource: {doc.get('page_url','')}\n{text}\n"

    if not context.strip():
        return jsonify(fallback_response())

    answer = ask_claude(context, raw_question)
    if not answer:
        return jsonify(fallback_response())
    if answer == "OUT_OF_SCOPE":
        return jsonify(out_of_scope_response())

    return jsonify({"answer": answer, "suggestions": default_suggestions})

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        threading.Thread(target=start_auto_updater, daemon=True).start()
    app.run(debug=True)