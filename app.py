import streamlit as st
import requests
from bs4 import BeautifulSoup
import validators
import os
import json
from datetime import datetime
import re
from collections import Counter
import numpy as np

# Try to import OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    st.warning("OpenAI package not installed. Using simple keyword matching.")

# ========== SIMPLE EMBEDDING FIX ==========
class SimpleEmbeddings:
    """Fixed simple embeddings that always returns same dimension"""
    
    def __init__(self):
        self.vocabulary = set()
        self.word_to_idx = {}
        self.embedding_dim = 0
        
    def build_vocabulary(self, texts):
        """Build vocabulary from all texts"""
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        self.vocabulary = all_words
        self.word_to_idx = {word: i for i, word in enumerate(all_words)}
        self.embedding_dim = len(all_words)
        return self.embedding_dim
    
    def text_to_vector(self, text):
        """Convert text to vector using bag-of-words"""
        words = text.lower().split()
        vector = np.zeros(self.embedding_dim)
        
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                vector[idx] = count
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

# ========== SESSION STATE ==========
st.set_page_config(
    page_title="WeBot by Shubhrant Tripathi",
    page_icon="🌐",
    layout="wide"
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'website_data' not in st.session_state:
    st.session_state.website_data = {}
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []
if 'current_url' not in st.session_state:
    st.session_state.current_url = None

# Initialize embedding system
if 'embedder' not in st.session_state:
    st.session_state.embedder = SimpleEmbeddings()

# ========== HELPER FUNCTIONS ==========
def validate_url(url):
    """Simple URL validation"""
    if not url or not url.strip():
        return False, "URL cannot be empty"
    if not validators.url(url):
        return False, "Invalid URL format"
    return True, ""

def extract_website_content(url):
    """Extract text content from website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else url
        
        # Get main content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean text - remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove very short lines (likely navigation)
        lines = text.split('. ')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 30]
        cleaned_text = '. '.join(cleaned_lines)
        
        return {
            'url': url,
            'title': title,
            'content': cleaned_text[:20000],  # Limit size
            'success': True
        }
        
    except Exception as e:
        return {
            'url': url,
            'error': str(e),
            'success': False
        }

def chunk_text(text, chunk_size=1000, overlap=200):
    """Simple text chunking"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Only add non-empty chunks
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 100:  # Minimum chunk size
                chunks.append(chunk)
        i += chunk_size - overlap
    
    # Remove exact duplicates
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    return unique_chunks

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        # If dimensions don't match, pad with zeros
        max_dim = max(len(vec1), len(vec2))
        vec1 = np.pad(vec1, (0, max_dim - len(vec1)))
        vec2 = np.pad(vec2, (0, max_dim - len(vec2)))
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    similarity = dot_product / (norm1 * norm2)
    return max(0, min(1, similarity))  # Ensure between 0 and 1

def find_relevant_chunks(query, chunks, embedder, top_k=3):
    """Find most relevant chunks for a query"""
    if not chunks:
        return []
    
    # Create query embedding
    query_vector = embedder.text_to_vector(query)
    
    # Calculate similarities
    results = []
    for i, chunk in enumerate(chunks):
        chunk_vector = embedder.text_to_vector(chunk)
        similarity = cosine_similarity(query_vector, chunk_vector)
        results.append((similarity, chunk, i))
    
    # Sort by similarity and return top-k
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]

def get_openai_answer(query, context_chunks, api_key):
    """Get answer using OpenAI"""
    if not HAS_OPENAI or not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare context
        context = "\n\n".join([chunk[:500] for chunk in context_chunks])
        
        # Get conversation history
        history = ""
        if st.session_state.conversation_memory:
            history = "Previous conversation:\n"
            for q, a in st.session_state.conversation_memory[-2:]:
                history += f"Q: {q}\nA: {a}\n\n"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant that answers questions based ONLY on the provided website content.
                    
                    WEBSITE CONTEXT:
                    {context}
                    
                    {history}
                    
                    IMPORTANT RULES:
                    1. Answer using ONLY the context above.
                    2. Do not use any outside knowledge.
                    3. If the answer is not in the context, say EXACTLY: "The answer is not available on the provided website."
                    4. Do not apologize or explain when answer is not available.
                    5. Keep answers concise."""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}"
                }
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Ensure exact response for unavailable answers
        unavailable_phrases = [
            "not available on the provided website",
            "cannot be found in the context",
            "not present in the website",
            "i don't have enough information",
            "the context does not contain"
        ]
        
        if any(phrase in answer.lower() for phrase in unavailable_phrases):
            answer = "The answer is not available on the provided website."
        
        return answer
        
    except Exception as e:
        st.error(f"OpenAI Error: {str(e)}")
        return None

def get_simple_answer(query, relevant_chunks):
    """Get simple answer without OpenAI"""
    if not relevant_chunks:
        return "The answer is not available on the provided website."
    
    # Use the most relevant chunk
    best_chunk = relevant_chunks[0][1]  # (similarity, chunk, index)
    
    # Extract sentences containing query words
    query_words = set(query.lower().split())
    sentences = re.split(r'[.!?]+', best_chunk)
    
    relevant_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        if query_words.intersection(sentence_words):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return ' '.join(relevant_sentences[:3])[:300] + "..."
    else:
        # Return beginning of best chunk
        return best_chunk[:200] + "..."

# ========== STREAMLIT UI ==========
st.title("Website Chatbot")
st.markdown("Ask questions about any website content")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key (optional):",
        type="password",
        help="For better answers. Get from platform.openai.com",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    st.divider()
    
    # Website URL
    st.subheader("1. Enter Website")
    url = st.text_input(
        "Website URL:",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
        key="url_input",
        value=st.session_state.get('url_input', '')
    )
    
    # Load Button
    if st.button("Load Website", type="primary", use_container_width=True):
        if not url:
            st.error("Please enter a website URL")
        else:
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                st.error(f"Invalid URL: {error_msg}")
            else:
                with st.spinner("Loading and processing website..."):
                    # Extract content
                    content_data = extract_website_content(url)
                    
                    if content_data['success']:
                        # Chunk text
                        chunks = chunk_text(content_data['content'])
                        
                        if chunks:
                            # Build embeddings vocabulary
                            embedder = SimpleEmbeddings()
                            embedder.build_vocabulary(chunks)
                            
                            # Store everything
                            st.session_state.website_data = {
                                'url': content_data['url'],
                                'title': content_data['title'],
                                'chunks': chunks,
                                'embedder': embedder
                            }
                            st.session_state.current_url = url
                            
                            # Clear chat
                            st.session_state.chat_history = []
                            st.session_state.conversation_memory = []
                            
                            st.success(f"Loaded: {content_data['title']}")
                            st.success(f"Created {len(chunks)} text chunks")
                        else:
                            st.error("Could not extract meaningful content")
                    else:
                        st.error(f"Failed to load website")
    
    st.divider()
    
    # Status
    st.subheader("Status")
    if st.session_state.current_url:
        data = st.session_state.website_data
        st.success("Website loaded")
        st.caption(f"Title: {data.get('title', 'Unknown')}")
        st.caption(f"Chunks: {len(data.get('chunks', []))}")
    else:
        st.info("No website loaded")
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.rerun()
    
    with col2:
        if st.button("Clear Website", use_container_width=True):
            st.session_state.website_data = {}
            st.session_state.current_url = None
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Quick test URLs
    st.caption("**Try these:**")
    if st.button("Wikipedia - AI", use_container_width=True):
        st.session_state.url_input = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        st.rerun()
    
    if st.button("Python Docs", use_container_width=True):
        st.session_state.url_input = "https://docs.python.org/3/tutorial/"
        st.rerun()

# Main Chat Interface
st.header("Chat")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        
        if chat.get("confidence"):
            st.caption(f"Confidence: {chat['confidence']:.0%}")

# Chat input
if st.session_state.current_url:
    website_title = st.session_state.website_data.get('title', 'the website')
    
    if prompt := st.chat_input(f"Ask about {website_title}..."):
        if not st.session_state.website_data.get('chunks'):
            st.warning("No content loaded. Please load a website first.")
        else:
            # Add user question to chat
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": "",
                "timestamp": datetime.now().isoformat()
            })
            
            # Show thinking spinner
            with st.spinner("Searching for answer..."):
                try:
                    # Get website data
                    chunks = st.session_state.website_data['chunks']
                    embedder = st.session_state.website_data['embedder']
                    
                    # Find relevant chunks
                    relevant = find_relevant_chunks(prompt, chunks, embedder, top_k=3)
                    
                    if relevant:
                        # Extract just the chunk texts
                        context_chunks = [chunk for _, chunk, _ in relevant]
                        
                        # Try OpenAI first if available
                        answer = None
                        if api_key and HAS_OPENAI:
                            answer = get_openai_answer(prompt, context_chunks, api_key)
                        
                        # Fallback to simple method
                        if not answer:
                            answer = get_simple_answer(prompt, relevant)
                        
                        # Get confidence from best match
                        confidence = relevant[0][0] if relevant else 0
                        
                        # Update chat
                        st.session_state.chat_history[-1]["answer"] = answer
                        st.session_state.chat_history[-1]["confidence"] = confidence
                        
                        # Add to conversation memory (last 3)
                        st.session_state.conversation_memory.append((prompt, answer))
                        if len(st.session_state.conversation_memory) > 3:
                            st.session_state.conversation_memory = st.session_state.conversation_memory[-3:]
                        
                    else:
                        # No relevant chunks found
                        answer = "The answer is not available on the provided website."
                        st.session_state.chat_history[-1]["answer"] = answer
                    
                    # Rerun to show answer
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.chat_history[-1]["answer"] = error_msg
                    st.error(f"An error occurred: {str(e)}")
                    st.rerun()
else:
    # Welcome screen
    st.info("Enter a website URL in the sidebar to get started")
    
    st.markdown("""
    ### Example workflow:
    1. Enter a website URL (e.g., https://en.wikipedia.org/wiki/Artificial_intelligence)
    2. Click "Load Website"
    3. Ask questions about the content
    4. Get answers based only on that website
    
    ### Try asking:
    - "What is this website about?"
    - "What are the main topics discussed?"
    - "Explain the key concepts mentioned"
    """)
    
    # Example questions
    st.divider()
    st.subheader("Quick Test (if website loaded):")
    
    test_questions = [
        "What is the main topic of this website?",
        "Can you summarize the content?",
        "What are the key points mentioned?"
    ]
    
    for q in test_questions:
        if st.button(q, key=f"test_{q}"):
            if st.session_state.current_url:
                # Simulate asking this question
                st.session_state.chat_history.append({
                    "question": q,
                    "answer": "Click 'Load Website' first, then ask questions!",
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

# Footer
st.divider()
st.caption("WeBot | Developed by Shubhrant Tripathi | © 2026")