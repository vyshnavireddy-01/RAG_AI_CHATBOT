# 🤖 RAG AI Chatbot for Ealkay

An AI-powered Retrieval-Augmented Generation (RAG) chatbot developed for Ealkay. This system retrieves relevant information from company data sources and generates accurate, context-aware responses in real-time.

---

## 🚀 Features

- 🔍 Retrieval-based response system using vector search
- 🧠 Context-aware answers using LLM integration
- 🌐 Website data ingestion using crawler
- 📄 Text chunking and embedding generation
- ⚡ Real-time chatbot interaction
- 🔄 Automated scheduled data updates

---

## 🛠️ Tech Stack

- Python
- Flask
- MongoDB
- Vector Embeddings
- HTML, CSS
- LLM API (Anthropic / OpenAI)

---

## 📁 Project Structure
RAG_AI_CHATBOT/
│── app.py
│── crawler.py
│── chunker.py
│── embedder.py
│── retriever.py
│── auto_updater.py
│── config.py
│── requirements.txt
│── templates/
│ └── chat.html
│── static/
│── .gitignore---

## ⚙️ Installation & Setup

### 1. Clone the repository


git clone https://github.com/yourusername/RAG_AI_CHATBOT.git

cd RAG_AI_CHATBOT


---

### 2. Create virtual environment


python -m venv venv


Activate it:

- Windows:

venv\Scripts\activate


- Mac/Linux:

source venv/bin/activate
---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Setup environment variables

Create a `.env` file and add:


API_KEY=your_api_key_here
MONGO_URI=your_mongodb_connection


---

### 5. Run the application


python app.py


---

## 🔄 How It Works

1. Crawl website data using `crawler.py`
2. Split content into chunks using `chunker.py`
3. Generate embeddings using `embedder.py`
4. Store embeddings in MongoDB
5. Retrieve relevant data using `retriever.py`
6. Generate responses using LLM
7. Display results via Flask UI

---

## 🔐 Security Note

- Do NOT upload `.env` file to GitHub
- Keep API keys secure
- Always use environment variables

---

## 📌 Future Improvements

- User authentication
- UI/UX improvements
- Multi-language support
- Performance optimization

---

## 👩‍💻 Author

Vyshnavi

---

## 📄 License

This project is for educational and internal use.
