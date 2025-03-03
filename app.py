import streamlit as st
import os
from vectorstore import process_and_store
from rag_chain import chat_with_document
from config import Config

# Set up the app
st.set_page_config(page_title="Chat with Your Document", page_icon="📄")
st.title("📄 Chat with Your Document (RAG + LangChain)")

# Upload the document
uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_file.name)
    
    # Check if a new document is uploaded
    if "current_doc" not in st.session_state or st.session_state.current_doc != uploaded_file.name:
        st.session_state.current_doc = uploaded_file.name  # Update current document
        st.session_state.messages = []  # Reset chat history

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded! Processing...")
    process_and_store(file_path)
    st.success("Document indexed. Start chatting!")

# Initialize chat history if not set
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask a question..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response from RAG model
    response = chat_with_document(query)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
