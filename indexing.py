import os
import json
from glob import glob
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from config import embeddings

def load_documents_from_folder(folder_path: str):
    """Loads JSON files from a folder and extracts the 'content' field as a Document."""
    file_paths = glob(os.path.join(folder_path, '*.json'))
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

# Load documents from folders
inheritance_docs = load_documents_from_folder('laws/inheritance')
divorce_docs = load_documents_from_folder('laws/divorce')

# Create two separate Chroma vector stores
inheritance_db = Chroma.from_documents(inheritance_docs, embeddings, collection_name="inheritance")
divorce_db = Chroma.from_documents(divorce_docs, embeddings, collection_name="divorce")

# Create retrievers for each vector store
inheritance_retriever = inheritance_db.as_retriever()
divorce_retriever = divorce_db.as_retriever()

# Expose a dictionary mapping for retrieval
retriever_dict = {
    "INHERITANCE": inheritance_retriever,
    "DIVORCE": divorce_retriever,
}
