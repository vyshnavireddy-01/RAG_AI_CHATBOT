RAG Powered AI Chatbot
This project implements a Retrieval Augmented Generation (RAG) AI Chatbot that can answer questions based on website content.

The system crawls a website, converts the content into embeddings, stores them in a vector database, retrieves relevant information, and generates answers using an AI model.

Technologies Used
Python
Flask
MongoDB
FAISS (Vector Database)
Sentence Transformers
BeautifulSoup
Anthropic Claude API
Project Architecture
User Question
↓
Convert Question → Embedding
↓
FAISS Vector Search
↓
Retrieve Relevant Text Chunks
↓
Send Context to Claude AI
↓
Generate Final Answer

Project Structure
RAG Project
│
├── app.py
├── crawler.py
├── chunker.py
├── embedder.py
├── retriever.py
├── config.py
│
├── templates
│   └── chat.html
│
├── vector_store
│
├── README.md
└── .gitignore
Prerequisites
Before running the project, install:

Python 3.10+
MongoDB
pip
Installation
Clone the repository

git clone https://github.com/vyshnavireddy-01/rag-ai-chatbot.git
Go to project directory

cd rag-ai-chatbot
Install dependencies

pip install flask pymongo faiss-cpu sentence-transformers requests beautifulsoup4 anthropic python-dotenv
Set API Key
This project requires an Anthropic Claude API key.

Set it as an environment variable.

Linux / Mac

export ANTHROPIC_API_KEY="your_api_key"
Windows (Command Prompt)

set ANTHROPIC_API_KEY=your_api_key
Windows (PowerShell)

$env:ANTHROPIC_API_KEY="your_api_key"
Start MongoDB
Start MongoDB before running the application.

Example commands:

mongod
or start MongoDB service if installed as a service.

Run the Application
Start the Flask server:

python app.py
Open the browser:

http://127.0.0.1:5000
You will see the chatbot interface.

Ingest Website Data
Before asking questions, the chatbot must load website content.

Run the following command in a new terminal:

curl -X POST http://127.0.0.1:5000/ingest -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
This step will:

Crawl the website
Split content into chunks
Generate embeddings
Store vectors in FAISS
Store text in MongoDB
Example Questions
Try asking questions such as:

What services does the company provide?
What are the main offerings?
What technologies are used?
What is the company's mission?
Learning Outcomes
Students will learn:

Retrieval Augmented Generation (RAG)
Vector embeddings
Semantic search
AI chatbot development
Integrating LLM APIs
Building AI-powered web applications
License
