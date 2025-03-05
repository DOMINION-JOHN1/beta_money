import os
from dotenv import load_dotenv
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables 
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
API_KEY = os.getenv("GEMINI_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)


def add_webscraped_data_to_pinecone(webscraped_data, embedding_model):
    """
    Splits the web-scraped data into smaller chunks, vectorizes them, and adds them to the Pinecone index.
    
    Args:
        webscraped_data (list of dict): Each dictionary should have keys 'id' and 'text'.
        embedding_model: An embedding model instance with:
                         - embed_text: function to compute embeddings.
                         - embedding_dimension: dimensionality of embeddings.
    
    Returns:
        PineconeVectorStore: The vector store instance with the added documents.
    """
    # Define text splitter parameters
    chunk_size = 2000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '.', ' ']
    )
    
    index_name = "beta"
    # Check if the index exists; if not, create it using the embedding model's dimension.
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embedding_model.embedding_dimension)
    
    # Create the vector store using the Pinecone index and embedding function.
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding_function=embedding_model.embed_text
    )
    
    # Split each document into chunks.
    splitted_documents = []
    for doc in webscraped_data:
        # Use the text splitter to split the document text.
        chunks = text_splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            splitted_documents.append({"id": f"{doc['id']}_{i}", "text": chunk})
    
    # Extract texts and IDs from the splitted documents.
    texts = [doc["text"] for doc in splitted_documents]
    ids = [doc["id"] for doc in splitted_documents]
    
    # Add the splitted documents to the vector store.
    vector_store.add_texts(texts=texts, ids=ids)
    
    return vector_store


if __name__ == "__main__":
    add_webscraped_data_to_pinecone(webscraped_data, embedding_model)
    
