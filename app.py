import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
import torch
from concurrent.futures import ThreadPoolExecutor

# Logging setup for debugging
logging.basicConfig(level=logging.INFO)

# Set Cohere API key 
API_KEY = "nEBF2bqrVJBFGWHKhQxqPU49n7ompcmuNZs5IXCr"
COHERE_API_URL = "https://api.cohere.ai/v1/generate"
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Load models for summarization, NER, and semantic chunking
summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the embedding model for chunking
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cohere prompt limit
MAX_TOKENS = 2048  # Token limit for Cohere API
MAX_CHUNK_TOKENS = 1200  # Allow room for the question

# Clear the previous question when a new dataset is uploaded
def clear_previous_question():
    st.session_state["user_query"] = ""

# Function to make a Cohere API request for text generation
def ask_cohere(question, context):
    logging.info(f"Asking Cohere: {question[:50]} with context: {context[:50]}...")
    payload = {
        "model": "command-xlarge-nightly",
        "prompt": f"Question: {question}\nContext: {context}",
        "max_tokens": 200  # Limit response size
    }
    response = requests.post(COHERE_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["generations"][0]["text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function for semantic chunking using sentence embeddings
def semantic_chunking(text, chunk_size=500):
    logging.info("Performing semantic chunking...")
    sentences = text.split('. ')
    chunks = []
    batch = ""
    for sentence in sentences:
        if len(batch.split()) + len(sentence.split()) <= chunk_size:
            batch += sentence + '. '
        else:
            chunks.append(batch.strip())
            batch = sentence + '. '
    if batch:
        chunks.append(batch.strip())
    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

# Function to find the most relevant chunk using semantic similarity with batching
def find_relevant_chunk(question, chunks, batch_size=8):
    logging.info(f"Finding relevant chunk for question: {question[:50]}")
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    
    # Encode all chunks in parallel
    with ThreadPoolExecutor() as executor:
        chunk_embeddings = list(executor.map(lambda chunk: embedding_model.encode(chunk, convert_to_tensor=True), chunks))

    # Calculate cosine similarity for all chunks at once
    similarities = util.pytorch_cos_sim(question_embedding, torch.stack(chunk_embeddings))
    
    # Find the best chunk by identifying the highest similarity score
    best_score = torch.max(similarities).item()
    best_chunk_idx = torch.argmax(similarities).item()

    best_chunk = chunks[best_chunk_idx]
    logging.info(f"Best chunk score: {best_score}")
    return best_chunk

# Truncate chunk to fit within Cohere API token limit
def truncate_chunk(chunk, max_chunk_tokens=MAX_CHUNK_TOKENS):
    tokens = chunk.split()  # Simple tokenization
    if len(tokens) > max_chunk_tokens:
        truncated_chunk = ' '.join(tokens[:max_chunk_tokens])
        logging.info("Truncating chunk to fit token limit")
        return truncated_chunk
    return chunk

# Streamlit app UI
st.title('Advanced Data Question Answering with Cohere API (NER and Summarization)')

# Clear previous query when a new file is uploaded
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

uploaded_file = st.file_uploader("Upload your data file (CSV, Excel, TXT)", type=["csv", "xlsx", "txt"], on_change=clear_previous_question)

if uploaded_file:
    with st.spinner('Processing uploaded file...'):
        if uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            st.write("CSV File Uploaded Successfully:")
            st.write(df.head())
            context = df.to_string(index=False)
            enriched_chunks = [context]  # Treat the whole CSV content as a single chunk

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
            st.write("Excel File Uploaded Successfully:")
            st.write(df.head())
            context = df.to_string(index=False)
            enriched_chunks = [context]  # No chunking needed for Excel files

        elif uploaded_file.type == 'text/plain':  # Handling text files
            text_data = uploaded_file.read().decode('utf-8')
            st.write("Text File Content Uploaded:")
            context = text_data

            # Apply semantic chunking
            with st.spinner('Applying semantic chunking...'):
                semantic_chunks = semantic_chunking(text_data, chunk_size=500)

            enriched_chunks = semantic_chunks

            st.success('Text successfully chunked!')

        else:
            st.error("Unsupported file type.")

    # Get user query
    user_query = st.text_input("Ask a question about the uploaded file", key="user_query")

    if user_query and enriched_chunks:
        with st.spinner('Processing your query...'):
            relevant_chunk = find_relevant_chunk(user_query, enriched_chunks)

            # Truncate the chunk to fit within token limits
            truncated_chunk = truncate_chunk(relevant_chunk)

            # Use the Cohere API to get the answer
            answer = ask_cohere(user_query, truncated_chunk)

        st.write("### Answer")
        st.write(answer)
