from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

def create_searchable_db(text):
    """
    Create a vector database from text for semantic search
    
    Args:
        text: The transcript text to process
        
    Returns:
        vector_db: ChromaDB vector store
    """
    try:
        # 1. Break long text into smaller chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        print(f"✓ Created {len(chunks)} chunks")
        
        # 2. Convert text chunks into "Numbers" (Embeddings)
        print("Loading embedding model...")
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("✓ Embedding model loaded")
        
        # 3. Create the database
        print("Creating vector database...")
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("✓ Vector database created")
        
        return vector_db
        
    except Exception as e:
        print(f"Error creating searchable database: {e}")
        raise


def search_database(vector_db, query, k=3):
    """
    Search the vector database for relevant information
    
    Args:
        vector_db: ChromaDB vector store
        query: Question to search for
        k: Number of results to return
        
    Returns:
        List of relevant text chunks
    """
    try:
        results = vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error searching database: {e}")
        raise


def load_existing_db():
    """Load an existing vector database"""
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        return vector_db
    except Exception as e:
        print(f"Error loading database: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    enabling computers to learn from data without being explicitly programmed. 
    Deep learning, a subset of machine learning, uses neural networks with 
    multiple layers to learn complex patterns in data.
    """
    
    # Create database
    db = create_searchable_db(sample_text)
    
    # Search database
    query = "What is deep learning?"
    results = search_database(db, query)
    
    print(f"\nQuery: {query}")
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}\n")