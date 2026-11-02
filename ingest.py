import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

def get_embeddings():
    """Initialize open-source local embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def process_and_store_document(file_path: str, original_filename: str):
    """
    Parses a PDF or DOCX file, chunks it with legal clause awareness,
    and stores embeddings into ChromaDB.
    """
    # 1. Load Document based on extension
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format for: {original_filename}")
    
    docs = loader.load()

    # 2. Split text intelligently into clause-sized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\nSECTION", "\nClause", "\nARTICLE", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(docs)

    # Attach rich metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = original_filename

    # 3. Store into ChromaDB vector store
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    
    vector_store.add_documents(chunks)
    vector_store.persist()
    
    return len(chunks)

def get_db_stats():
    """Retrieve summary stats of stored precedents in ChromaDB."""
    try:
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        collection = vector_store._collection
        count = collection.count()
        
        # Extract unique source files from metadata
        data = collection.get(include=["metadatas"])
        sources = set(meta.get("source_file", "Unknown") for meta in data["metadatas"]) if data["metadatas"] else set()
        
        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "documents": list(sources)
        }
    except Exception:
        return {"total_chunks": 0, "total_documents": 0, "documents": []}
