# WeBot - Website Chatbot 

## Project Overview
**WeBot** is an AI-powered chatbot that allows users to interact with any website by asking questions **strictly based on the website's content**.  
Built for the **Humanli.ai AI/ML Engineer assessment**, this application demonstrates modern AI techniques including web scraping, text embeddings, vector search, and natural language processing.

---

## Key Features
- **Website Crawling**: Extract clean content from any website  
- **Smart Chunking**: Split content into semantic chunks for processing  
- **AI-Powered Q&A**: Answer questions using OpenAI GPT or simple text matching  
- **Strict Content Grounding**: Answers are based **ONLY** on website content  
- **Conversation Memory**: Maintains context across multiple questions  
- **Responsive UI**: Works seamlessly across all devices  
- **Real-time Processing**: Instant answers to user queries  

---

## Technology Stack

| Component | Technology | Purpose |
|--------|------------|---------|
| Frontend | Streamlit | Interactive web interface |
| Web Scraping | BeautifulSoup4 | Extract website content |
| Text Processing | Custom chunking | Split content into manageable pieces |
| Embeddings | Bag-of-Words / OpenAI | Create searchable text representations |
| Vector Search | Cosine similarity | Find relevant content chunks |
| LLM | OpenAI GPT-3.5-turbo | Generate human-like answers |
| Memory | Session-based storage | Maintain conversation context |

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher  
- OpenAI API key *(optional, for better answers)*  
- Internet connection  

---

### Step-by-Step Setup

#### 1️⃣ Clone the project
```bash
https://github.com/ishubtripathi/WeBot.git

```
#### 2️⃣ Create virtual environment
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install dependencies
```bash
pip install streamlit beautifulsoup4 requests validators openai numpy
```

#### 4️⃣ Create environment file (optional)
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```
#### 5️⃣ Run the application
```bash
streamlit run app.py
```

**App runs at: http://localhost:8501**

---
## Step-by-Step Setup

#### 1️⃣ Website Indexing Process
 Input URL → Validate → Crawl → Clean → Chunk → Create Embeddings → Store

### Steps
- URL Validation
- Content Extraction (BeautifulSoup)
- Content Cleaning
- Text Chunking (1000 chars + 200 overlap)
- Embedding Creation
- In-memory Storage

### 2️⃣ Question Answering Process
 User Question → Embedding → Similarity Search → Context Retrieval → Answer Generation

### Steps
- Question embedding
- Cosine similarity search
- Top-3 chunk selection
- Answer generation (with or without OpenAI)
- Conversation memory update

---

## Usage Guide
```bash
streamlit run app.py
```
- Enter OpenAI API key (optional)
- Enter website URL
- Click Load Website
- Ask questions in chat


## Future Improvements
- Multi-page crawling
- Vector DB integration
- Semantic embeddings
- Citation tracking
- Cloud deployment
- Browser extension

## Security & Privacy
- No data persistence
- API key stored in session only
- No third-party sharing (except OpenAI)
- Input validation & rate limiting


## Developer

**Name:** Shubhrant Tripathi  
**Project:** WeBot – Website Chatbot  
**Purpose:** Humanli.ai AI/ML Engineer Assessment  
**Year:** 2026  

### Profiles
- GitHub: https://github.com/ishubtripathi  
- LinkedIn: https://www.linkedin.com/in/ishubtripathi/  

📩 *For queries please reach out via GitHub or LinkedIn.*
