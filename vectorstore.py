import os
import re
import faiss
import numpy as np
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import JinaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

VECTOR_DB_PATH = Config.VECTOR_DB_PATH

# Initialize Jina Embeddings
text_embeddings = JinaEmbeddings(
    jina_api_key=Config.JINA_API_KEY, 
    model_name=Config.EMBEDDING_MODEL
)

def split_by_headings(text):
    """Splits text into sections under headings, then applies a recursive text splitter."""
    sections = []
    current_heading = " "
    current_content = []

    heading_pattern = re.compile(r"^(?:[A-Z][A-Z0-9 ]{2,}|[\d\.]+\s+[A-Z].*)$")

    for line in text.split("\n"):
        line = line.strip()
        if heading_pattern.match(line):  
            if current_content:  
                sections.append((current_heading, "\n".join(current_content)))
                current_content = []
            current_heading = line  
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_heading, "\n".join(current_content)))

    # Apply RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_sections = []
    
    for heading, content in sections:
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            split_sections.append((heading, chunk))

    return split_sections

def process_and_store(file_path):
    """Loads document, splits text using headings and recursive text splitter, and stores embeddings in FAISS."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    full_text = "\n".join([doc.page_content for doc in documents])
    
    sections = split_by_headings(full_text)

    texts = [f"{heading}\n\n{content}" for heading, content in sections]
    embeddings = text_embeddings.embed_documents(texts)

    # Ensure embeddings match text count
    assert len(texts) == len(embeddings), "Mismatch between texts and embeddings!"

    # Initialize FAISS index
    embedding_dim = len(embeddings[0])  
    index = faiss.IndexFlatL2(embedding_dim)  

    # Initialize FAISS vector store
    vector_store = FAISS(
        embedding_function=text_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add embeddings
    text_embedding_pairs = list(zip(texts, embeddings))
    vector_store.add_embeddings(text_embedding_pairs)

    # Save FAISS index
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)
    vector_store.save_local(VECTOR_DB_PATH)

def load_vectorstore():
    """Loads the FAISS vector store."""
    if not os.path.exists(f"{VECTOR_DB_PATH}/index.faiss"):
        raise FileNotFoundError(f"FAISS index not found in {VECTOR_DB_PATH}/index.faiss. Did you run process_and_store()?")

    return FAISS.load_local(VECTOR_DB_PATH, text_embeddings, allow_dangerous_deserialization=True)
