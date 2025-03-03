from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from vectorstore import load_vectorstore
from config import Config

def get_rag_chain():
    """Creates a RAG pipeline with Ollama DeepSeek"""
    llm = OllamaLLM(model=Config.OLLAMA_MODEL)
    retriever = load_vectorstore().as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever
    )
    return qa_chain

def chat_with_document(query):
    """Processes user query through RAG pipeline"""
    chain = get_rag_chain()
    return chain.run(query)
