import os
import json
from glob import glob
import pandas as pd
from typing import List
from langchain.docstore.document import Document
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

# Constants
LLM_MODEL = "llama3.3"

# Load documents
def load_documents_from_folder(folder_path: str) -> List[Document]:
    file_paths = glob(os.path.join(folder_path, '*.json'))
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

# Generate a single question based on a law document
def generate_question(llm, doc: Document, split: str) -> str:
    prompt = (
        f"Given the following italian law about {split}, generate ONE clear and helpful question "
        "a user might ask after reading it:\n\n"
        f"{doc.page_content}\n\nQuestion:"
    )
    response = llm.invoke(prompt)
    question = response.strip().split("\n")[0].strip("- ").strip()
    return question

# Answer a question based on the document
def generate_answer(llm, doc: Document, question: str) -> str:
    prompt = (
        f"Answer the following question using only the context provided below.\n\n"
        f"Context:\n{doc.page_content}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.strip()

# Build testset with document, question, and answer
def build_testset(llm, documents: List[Document], split: str) -> List[dict]:
    entries = []
    for doc in tqdm(documents, desc=f"Generating QA pairs for {split}"):
        try:
            question = generate_question(llm, doc, split)
            answer = generate_answer(llm, doc, question)
            entries.append({
                "document": doc.page_content,
                "question": question,
                "answer": answer,
                "metadata": doc.metadata
            })
        except Exception as e:
            print(f"Skipping document due to error: {e}")
    return entries

# Setup
llm = OllamaLLM(model=LLM_MODEL)
inheritance_docs = load_documents_from_folder('laws/inheritance')
divorce_docs = load_documents_from_folder('laws/divorce')

# Generate test data
inheritance_data = build_testset(llm, inheritance_docs, split="inheritance")
divorce_data = build_testset(llm, divorce_docs, split="divorce")

# Save to CSV
pd.DataFrame(inheritance_data).to_csv("evaluation/inheritance_testset.csv", index=False)
pd.DataFrame(divorce_data).to_csv("evaluation/divorce_testset.csv", index=False)
