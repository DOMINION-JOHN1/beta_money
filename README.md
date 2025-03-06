# Financial Advisor RAG System with Multimedia Interaction

This project implements a Retrieval-Augmented Generation (RAG) system for financial advice focused on stocks such as NVIDIA (NVDA), Tesla (TSLA), and Alphabet (GOOG). The system processes web-scraped financial news and discussions, splits the content into manageable chunks, and stores them as embeddings in a Pinecone vector store. It leverages Gemini models (via LangChain integrations) for both text embeddings and LLM-based reasoning, and integrates Whisper for speech-to-text conversion. Additionally, the system offers text-to-speech (TTS) and video generation to create multimedia outputs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Ingestion & Vectorization](#data-ingestion--vectorization)
  - [RAG System Functions](#rag-system-functions)
  - [Example Workflow](#example-workflow)
- [Scheduled Deletion of Data](#scheduled-deletion-of-data)
- [Project Structure](#project-structure)

## Overview

The system is designed with the following components:

1. **Data Ingestion & Splitting:**  
   Web-scraped financial data is first split into smaller chunks using LangChain’s `RecursiveCharacterTextSplitter`. This allows more efficient embedding generation and retrieval.

2. **Vector Store Integration:**  
   Processed text chunks are embedded using a Gemini embedding model and stored in a Pinecone index named **"beta"**. This ensures that the context required for generating responses is available in an efficient vector database.

3. **RAG System:**  
   The core RAG engine uses the Gemini LLM (via LangChain integration) for personalized financial advice. The system prompt instructs the model to analyze user queries (which may be transcribed from audio using Whisper) and generate detailed responses that include advantages and disadvantages based on inferred risk appetite.

4. **Multimedia Output:**  
   Generated responses can be converted to speech using gTTS and further rendered as simple videos with MoviePy, offering multimodal interaction.

5. **Data Lifecycle Management:**  
   The system includes functionality to automatically delete web-scraped data and embeddings from Pinecone every 24 hours.

## Features

- **Web-Scraped Data Splitting:**  
  Uses a recursive text splitter with configurable chunk size and overlap.

- **Pinecone Vector Store:**  
  Efficiently stores text embeddings using a Gemini embedding model.

- **RAG Engine:**  
  Uses a Gemini LLM for personalized financial advice generation.

- **Speech-to-Text:**  
  Converts audio queries to text using OpenAI's Whisper model.

- **Text-to-Speech & Video Generation:**  
  Converts generated text responses into audio files (via gTTS) and video explanations (via MoviePy).

- **Automated Data Deletion:**  
  Includes a deletion function (and a sample GitHub Actions YAML) to remove data from the vector store every 24 hours.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DOMINION-JOHN1/beta_money.git
   cd beta_money
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8+ installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes all necessary packages such as `pinecone-client`, `langchain`, `gTTS`, `moviepy`, `whisper`, and others.

3. **Environment Variables:**

   Create a `.env` file in the root directory with the following variables:

   ```ini
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

4. **Start App Api:**

   ```bash
    fastapi dev src
   ```

   This runs a simple api for the system, with endpoints:

   - `/chat/text` for text queries
   - `/chat/audio` for audio queries
   - `/chat/video` for video queries

## Configuration

- **Embedding Model & LLM:**  
  The project uses Gemini models via the LangChain integrations:

  - **Embedding:** `GoogleGenerativeAIEmbeddings` with model `"models/embedding-001"`.
  - **LLM:** `GoogleGenerativeAI` with model `"gemini-1.5-flash"`.

- **Pinecone Index:**  
  The index is named **"beta"**. If it doesn't exist, it will be created with a dimension matching the Gemini embedding model.

- **Text Splitting:**  
  The text splitter uses a `chunk_size` of 2000 and a `chunk_overlap` of 400, with separators: `['\n\n', '\n', '.', ' ']`.

## Usage

### Data Ingestion & Vectorization

To ingest and store web-scraped data, use the function `add_webscraped_data_to_pinecone`. This function:

- Registers/creates the Pinecone index ("beta").
- Splits each document into smaller chunks.
- Converts the chunks into embeddings and adds them to the vector store.

**Example:**

```python
# Assume embedding_model is instantiated, e.g.,
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

webscraped_data = [
    {"id": "nvda_doc1", "text": "Latest news on NVIDIA showing strong performance in AI-driven sectors."},
    {"id": "tsla_doc1", "text": "Tesla's new model release and updates have boosted investor confidence."},
    {"id": "goog_doc1", "text": "Alphabet experiences significant growth in cloud business, increasing revenue prospects."}
]

vector_store = add_webscraped_data_to_pinecone(webscraped_data, embedding_model)
```

### RAG System Functions

The `RAGSystem` class encapsulates the following functions:

- **`speech_to_text(audio_filename)`**  
  Expects: Path to an audio file (e.g., "user_query.wav").  
  Returns: Transcribed text string using Whisper.

- **`retrieve_documents(query, top_k=5)`**  
  Expects: A text query (either typed or transcribed) and an optional `top_k` parameter specifying how many documents to retrieve.  
  Returns: A list of relevant documents from the vector store.

- **`generate_response(query, retrieved_docs)`**  
  Expects:

  - `query`: The user's query in text.
  - `retrieved_docs`: The documents retrieved from the vector store.  
    The function constructs a prompt (instructing the LLM to infer the user's risk appetite and provide detailed financial advice) and returns the generated response text.

- **`text_to_speech(text, filename="response.mp3")`**  
  Expects:

  - `text`: The generated response text.
  - `filename`: Optional output file name.  
    Returns: The path to the saved audio file.

- **`generate_video(text, output_filename="response_video.mp4")`**  
  Expects:
  - `text`: The generated response text.
  - `output_filename`: Optional video output file name.  
    Returns: The path to the saved video file with text overlay.

### Example Workflow

Below is an example usage script that demonstrates how to use the `RAGSystem` class in a real-life application:

```python
# Import your necessary modules and instantiate your models before this script.
# For instance:
# from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# import whisper
# ... (other necessary imports)

# Initialize Gemini models for LLM and embeddings.
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Create or update the Pinecone vector store using web-scraped data.
vector_store = add_webscraped_data_to_pinecone(webscraped_data, embedding_model)

# Load the Whisper model for speech-to-text.
whisper_model = whisper.load_model("base")

# Instantiate the RAG system.
rag_system = RAGSystem(vector_store=vector_store, llm=llm, whisper_model=whisper_model)

# Process an audio query (assuming an audio file "user_query.wav" exists).
audio_file = "user_query.wav"  # Path to the audio input file
try:
    user_query = rag_system.speech_to_text(audio_file)
except Exception as e:
    print("Error transcribing audio, defaulting to text input.")
    user_query = "Should I buy Tesla stock now?"

print("User Query (transcribed):", user_query)

# Retrieve relevant documents from the vector store.
retrieved_docs = rag_system.retrieve_documents(user_query)

# Generate a personalized financial advice response.
response_text = rag_system.generate_response(user_query, retrieved_docs)
print("Generated Response:", response_text)

# Optionally, convert the response to audio (TTS) and generate a video explanation.
audio_response_file = rag_system.text_to_speech(response_text)
print("Audio response saved to:", audio_response_file)

video_response_file = rag_system.generate_video(response_text)
print("Video response saved to:", video_response_file)
```

### Scheduled Data Deletion

To ensure that web-scraped data is stored only for 24 hours, the function `delete_webscraped_data_from_pinecone` is provided. You can schedule this function to run daily using a scheduler such as cron or GitHub Actions.

For example, a GitHub Actions workflow file (`.github/workflows/delete_cronjob.yaml`) can run the deletion script every 24 hours. The deletion script (`delete_data.py`) initializes Pinecone and calls the deletion function to clear all vectors from the "beta" index.

## Project Structure

```
financial-advisor-rag/
├── .env                  # Environment variables file
├── README.md             # Project documentation (this file)
├── requirements.txt      # Required Python packages
├── main.py               # Main script for running the RAG system (this is yet to be added)
├── clear_database.py        # Script for scheduled deletion of data from Pinecone
├── modules/
│   ├── ai_app.py     # Contains the RAGSystem class and related functions
│   └── add_data.py # Contains the function to add webscraped data (with splitting) to Pinecone
└── .github/
    └── workflows/
        └── scheduler.yaml  # GitHub Actions workflow for scheduled deletion
```

---

For any issues or feature requests, please create an issue in this repository or contact the maintainer.

```

This `README.md` covers the purpose, installation, configuration, and detailed usage instructions, including descriptions of each function and how they integrate in a real-world workflow.
```
