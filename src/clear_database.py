import os
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

def delete_webscraped_data_from_pinecone():
    index_name = "beta"
    if index_name in pinecone.list_indexes():
        index = pinecone.Index(index_name)
        index.delete(delete_all=True)
        print(f"All data in the index '{index_name}' has been deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")

if __name__ == "__main__":
    delete_webscraped_data_from_pinecone()
