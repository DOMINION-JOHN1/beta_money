import os
import json
from dotenv import load_dotenv
import pinecone
import whisper
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from gtts import gTTS
# import moviepy.editor as mpy

# Load environment variables 
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize model and embeddings
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

# Initialize vector store
index_name = "beta"
pinecone_vectorStore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# Load the Whisper model for speech-to-text conversion
# TODO: Reactivate when properly understood
# whisper_model = whisper.load_model("base")

class RAGSystem:
    def __init__(self):
        self.vector_store = pinecone_vectorStore
        self.llm = llm
        # self.whisper_model = whisper_model

    def retrieve_documents(self, query, top_k=5):
        # Use the vector store's similarity search to retrieve relevant documents
        return self.vector_store.similarity_search(query, k=top_k)
    
    def generate_response(self, query, retrieved_docs):
        """
        Generate financial advice with personalized risk appetite analysis.
        The prompt instructs the LLM to analyze the user query to infer risk tolerance,
        and to provide clear advantages and disadvantages of buying the specified stock.
        """
        # Combine retrieved documents (assumed to have a .page_content attribute) into context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
        You are a financial advisor. Your task is to generate a very detailed financial advice with personalized risk appetite analysis inferred from the user's query. In your analysis the advantages and disadvantages of buying the specified stock, based on the context provided.

        User Query: {query}
        Context: {context}

        Provide your detailed advice below:
        """
        response = self.llm.invoke(prompt)
        return response

    def speech_to_text(self, audio_filename):
        # Transcribe audio input using the Whisper model
        result = self.whisper_model.transcribe(audio_filename)
        return result["text"]

    def text_to_speech(self, text, filename="response.mp3"):
        # Convert text to speech using gTTS
        tts = gTTS(text)
        tts.save(filename)
        return filename

    # def generate_video(self, text, output_filename="response_video.mp4"):
    #     # Create a simple video with text overlay using MoviePy
    #     clip = mpy.ColorClip(size=(640, 480), color=(255, 255, 255), duration=10)
    #     txt_clip = mpy.TextClip(text, fontsize=24, color='black', method='caption', size=(600, 400))
    #     txt_clip = txt_clip.set_position('center').set_duration(10)
    #     video = mpy.CompositeVideoClip([clip, txt_clip])
    #     video.write_videofile(output_filename, fps=24)
    #     return output_filename

