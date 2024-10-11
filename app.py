import streamlit as st  
import pandas as pd  
import requests  # This is used to send HTTP requests, here to the Cohere API
from sentence_transformers import SentenceTransformer, util  # Used for text embeddings (converting text into numbers that a model can understand)
from transformers import BartForConditionalGeneration, BartTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer  # Different models and tools for summarization and image captioning
import torch  # PyTorch, a deep learning library, used here for handling data in tensor format
from concurrent.futures import ThreadPoolExecutor  # This allows running tasks in parallel (multi-threading)
import pyttsx3  # A library to convert text to speech (used for audio summaries)
import PyPDF2  # A library used to read PDF files
import gc  # Garbage Collector, used to clean up unused memory
import logging  # Logging is used for debugging and keeping track of events in the program

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",  # Defines the format of the log messages
    handlers=[logging.StreamHandler()]  # Send the log messages to the console
)

# Setting up the Cohere API (for asking questions and getting text summaries)
API_KEY = "nEBF2bqrVJBFGWHKhQxqPU49n7ompcmuNZs5IXCr" 
COHERE_API_URL = "https://api.cohere.ai/v1/generate"  # The URL endpoint of the Cohere API
HEADERS = {  
    'Authorization': f'Bearer {API_KEY}',  # Authorization token required to authenticate requests
    'Content-Type': 'application/json'  # The content type of the request is JSON
}


summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")  # A pre-trained model for summarizing text
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")  # Tokenizer to convert text into tokens for the BART model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # A pre-trained model used to create sentence embeddings (convert text to numbers)


# Caching the embeddings (to avoid recalculating them each time)
chunk_embedding_cache = {}  # Dictionary to store the embeddings for chunks of text, so we don't need to calculate them again

# Clears the previous question when a new file is uploaded
def clear_previous_question():
    st.session_state["user_query"] = ""  # Resets the user's question

# Function to ask a question to the Cohere API and get a text summary as the answer
def ask_cohere(question, context, length="Brief"):
    # Sets the number of tokens (words) in the response based on the length (Brief, Detailed, Comprehensive)
    max_tokens = 100 if length == "Brief" else 300 if length == "Detailed" else 800  
    # The data we send to the Cohere API
    payload = {
        "model": "command-xlarge-nightly",  
        "prompt": f"Question: {question}\nContext: {context}",  
        "max_tokens": max_tokens  
    }
    try:
        response = requests.post(COHERE_API_URL, headers=HEADERS, json=payload)  # Sending the data to the API
        response.raise_for_status()  # This will raise an error if the request fails
        return response.json()["generations"][0]["text"]  # Extracting the generated text from the API response
    except Exception as e:
        logging.error(f"Error in Cohere API request: {e}")  # Logs an error message if something goes wrong
        return f"Error: {e}"  # Returns an error message to display

# Function to divide large text into smaller chunks 
def semantic_chunking(text, chunk_size=500):
    logging.info("Performing semantic chunking...")  
    sentences = text.split('. ')  # Splits the text into sentences (based on ". ")
    chunks = []  # List to store chunks of text
    batch = ""  # This will hold the current chunk we are working on
    for sentence in sentences:  # Loop through each sentence
        # Add the sentence to the current batch if it's small enough
        if len(batch.split()) + len(sentence.split()) <= chunk_size:
            batch += sentence + '. '  # Add the sentence and a period to the batch
        else:
            chunks.append(batch.strip())  # If the batch is full, add it to the chunks list
            batch = sentence + '. '  # Start a new batch with the current sentence
    if batch:  # If there's still some leftover batch after the loop, add it to the chunks
        chunks.append(batch.strip())
    logging.info(f"Total chunks created: {len(chunks)}")  # Log how many chunks were created
    return chunks  # Return the list of chunks

# Function to find the chunk of text most relevant to the user's question
def find_relevant_chunk_with_cache(question, chunks):
    logging.info(f"Finding relevant chunk for question: {question[:50]}...")  # Log the start of this process
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)  # Convert the question into an embedding (vector of numbers)

    chunk_embeddings = []  # List to store embeddings for each chunk of text
    for chunk in chunks:  # Loop through each chunk
        if chunk in chunk_embedding_cache:  # Check if we already have the embedding for this chunk in the cache
            chunk_embeddings.append(chunk_embedding_cache[chunk])  # If so, use the cached embedding
        else:
            chunk_embedding = embedding_model.encode(chunk, convert_to_tensor=True)  # Otherwise, create a new embedding for the chunk
            chunk_embedding_cache[chunk] = chunk_embedding  # Save the new embedding to the cache
            chunk_embeddings.append(chunk_embedding)  # Add it to the list

    # Calculate the similarity between the question and each chunk, and return the most similar chunk
    similarities = util.pytorch_cos_sim(question_embedding, torch.stack(chunk_embeddings))
    best_chunk_idx = torch.argmax(similarities).item()  # Find the index of the most similar chunk
    best_chunk = chunks[best_chunk_idx]  # Get the corresponding chunk of text
    return best_chunk  # Return the best-matching chunk

# Function to make sure the chunk isn't too long for the Cohere API
def truncate_chunk(chunk, max_chunk_tokens=1200):
    tokens = chunk.split()  # Split the chunk into words (tokens)
    if len(tokens) > max_chunk_tokens:  # If it's too long, trim it down
        truncated_chunk = ' '.join(tokens[:max_chunk_tokens])  # Join the first 1200 tokens back into a string
        logging.info("Truncating chunk to fit token limit")  # Log that we truncated the chunk
        return truncated_chunk  # Return the truncated chunk
    return chunk  # If the chunk is already short enough, return it as-is



# Function to clean up resources (free up memory)
def cleanup_resources():
    logging.info("Cleaning up resources...")  # Log that cleanup is happening
    gc.collect()  # Manually run garbage collection (clean up unused memory)

# This is where the app starts, allowing users to upload a file
uploaded_file = st.file_uploader("Upload your data file (CSV, Excel, TXT, PDF)", type=["csv", "xlsx", "txt", "pdf"], on_change=clear_previous_question)

if uploaded_file:
    try:
        dataset_name = uploaded_file.name
        context = ""

        # Process based on file type
        if uploaded_file.type in ['text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            df = pd.read_csv(uploaded_file) if uploaded_file.type == 'text/csv' else pd.read_excel(uploaded_file)
            st.write(df.head())  # Show the first few rows of the data
            context = df.to_string(index=False)  # Convert the data to a string for processing
        elif uploaded_file.type == 'text/plain':
            context = uploaded_file.read().decode('utf-8')  # Read and decode the text file
            st.write(context[:500])  # Show the first 500 characters of the text
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)  # Initialize the PDF reader
            context = "".join([page.extract_text() for page in pdf_reader.pages])  # Extract text from all pages
            st.write(context[:500])  # Show the first 500 characters of the PDF text

        # Get user input for question and response length
        response_length = st.sidebar.selectbox("Select Response Length", ["Brief", "Detailed", "Comprehensive"])
        user_question = st.text_input("Ask a question about the uploaded file", key="user_query")

        if user_question:
            with st.spinner('Processing your query...'):
                # Split text into smaller chunks and find the most relevant one
                relevant_chunk = find_relevant_chunk_with_cache(user_question, semantic_chunking(context))
                truncated_chunk = truncate_chunk(relevant_chunk)
                # Get a summary from Cohere API
                summary = ask_cohere(user_question, truncated_chunk, response_length)
                st.write("### Answer")
                st.write(summary)

             

            # Clean up resources after processing
            cleanup_resources()

    except Exception as e:
        st.error(f"Error processing file: {e}")  # Show an error message if something goes wrong
