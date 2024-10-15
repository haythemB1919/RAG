import streamlit as st
import pandas as pd
import requests
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pyttsx3
import PyPDF2
import gc  # For memory management
import logging
import json  # For storing vectors

# Logging setup for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Set Cohere API key
API_KEY = ""
COHERE_API_URL = "https://api.cohere.ai/v1/generate"
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

# Model Initialization
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Caching embeddings and storing them on disk
VECTOR_CACHE_PATH = "vector_cache.json"

# Helper function to load cached vectors from disk
def load_vector_cache():
    if os.path.exists(VECTOR_CACHE_PATH):
        with open(VECTOR_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}

# Helper function to save vector cache to disk
def save_vector_cache(vector_cache):
    with open(VECTOR_CACHE_PATH, 'w') as f:
        json.dump(vector_cache, f)

# Load cached embeddings (persistent cache)
vector_cache = load_vector_cache()

# Function to compute and cache embeddings
def cache_embeddings(doc_name, text):
    if doc_name not in vector_cache:
        logging.info(f"Calculating embeddings for {doc_name}...")
        chunks = semantic_chunking(text)
        embeddings = [embedding_model.encode(chunk, convert_to_tensor=True).tolist() for chunk in chunks]
        vector_cache[doc_name] = {'chunks': chunks, 'embeddings': embeddings}
        save_vector_cache(vector_cache)  # Save to disk

# Semantic Chunking Function
def semantic_chunking(text, chunk_size=500):
    sentences = text.split('. ')
    chunks, batch = [], ""
    for sentence in sentences:
        if len(batch.split()) + len(sentence.split()) <= chunk_size:
            batch += sentence + '. '
        else:
            chunks.append(batch.strip())
            batch = sentence + '. '
    if batch:
        chunks.append(batch.strip())
    return chunks

# Find the most relevant chunk using cached embeddings
def find_relevant_chunk(question, doc_name):
    if doc_name not in vector_cache:
        return "Document not found in cache."

    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    chunk_embeddings = torch.tensor(vector_cache[doc_name]['embeddings'])

    # Calculate similarities between question and document embeddings
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
    best_chunk_idx = torch.argmax(similarities).item()
    return vector_cache[doc_name]['chunks'][best_chunk_idx]

# Function to process documents and cache their embeddings
def process_and_cache_files(files):
    for uploaded_file in files:
        content = ""
        if uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            content = df.to_string(index=False)
        elif uploaded_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = "".join([page.extract_text() for page in pdf_reader.pages])
        elif uploaded_file.type in ['text/plain']:
            content = uploaded_file.read().decode('utf-8')

        # Cache the document's embeddings
        cache_embeddings(uploaded_file.name, content)

# Cohere API Request Function
def ask_cohere(question, context):
    payload = {"model": "command-xlarge-nightly", "prompt": f"{question}\nContext: {context}", "max_tokens": 200}
    try:
        response = requests.post(COHERE_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()["generations"][0]["text"]
    except Exception as e:
        logging.error(f"Error in Cohere API: {e}")
        return f"Error: {e}"

# Streamlit UI Setup
st.title("QA System")

# Upload multiple files for overnight processing
uploaded_files = st.file_uploader("Upload Files for Processing", type=["csv", "pdf", "txt"], accept_multiple_files=True)

if st.button("Process Files and Cache Embeddings"):
    if uploaded_files:
        with st.spinner("Processing and caching files..."):
            process_and_cache_files(uploaded_files)
        st.success("Files processed and embeddings cached successfully!")

# Get user query
doc_name = st.selectbox("Select a Document", options=list(vector_cache.keys()))
user_question = st.text_input("Ask a question about the selected document")

if st.button("Get Answer"):
    with st.spinner("Fetching the answer..."):
        relevant_chunk = find_relevant_chunk(user_question, doc_name)
        answer = ask_cohere(user_question, relevant_chunk)
        st.write("### Answer")
        st.write(answer)

# Cleanup resources after processing
def cleanup_resources():
    gc.collect()

cleanup_resources()
