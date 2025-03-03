# 📄 Chat with Your Document (RAG + LangChain)

This project enables users to **upload a document and chat with it** using a **Retrieval-Augmented Generation (RAG) pipeline**. It leverages:

- **FAISS** for vector storage  
- **Ollama (DeepSeek-R1:1.5B)** for LLM-based question answering  
- **LangChain** for integrating embeddings and retrieval  
- **Streamlit** for an interactive UI  

---

## **🛠️ Setup Guide**

### **1️⃣ Install Prerequisites**

Ensure you have **Python 3.10+** installed. Then, install **Miniconda** (if not already installed):

👉 **Mac/Linux:**  
```bash
brew install miniforge
```

👉 **Windows:**  
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

---

### **2️⃣ Create and Activate a Conda Environment**
```bash
conda create -n doc-rag
conda activate doc-rag
```

---

### **3️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/doc-rag.git
cd doc-rag
```

---

### **4️⃣ Install Required Packages**
```bash
pip install -r requirements.txt
```

📌 **Ensure Ollama is installed:**  
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

---

### **5️⃣ Set Up Environment Variables**

Create a `.env` file in the root directory and add:

```ini
# API Keys (if using cloud embeddings)
JINA_API_KEY="your_jina_api_key"

# Ollama Model
OLLAMA_MODEL="deepseek-r1:7b"

# Vector Store Path
VECTOR_DB_PATH="faiss_index"
```

---

### **6️⃣ Download the Ollama Model**
Before running the app, pull the **DeepSeek model**:
```bash
ollama pull deepseek-r1:1.5b
```

---

### **7️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

---

## **🖥️ How It Works**

### **📤 Upload a Document**
1. Launch the Streamlit app.  
2. Upload a **PDF document** via the file uploader.  

### **🧠 Processing**
- The app **extracts text** from the document.  
- Text is **converted into embeddings** using a **local embedding model** (or Jina).  
- Embeddings are stored in a **FAISS vector database**.  

### **💬 Ask Questions**
- Type a question in the chat box.  
- The system retrieves relevant document sections and generates a response using **DeepSeek**.  

---

## **🛠️ Troubleshooting**

### **🔴 Conda Environment Not Found**
**Error:**  
```
conda: command not found
```
✅ **Solution:**  
Install Miniconda and restart your terminal.

---

### **🔴 Ollama Model Not Found**
**Error:**  
```
ValueError: model "deepseek" not found, try pulling it first
```
✅ **Solution:**  
Run:
```bash
ollama pull deepseek-r1:1.5b
```

---

### **🔴 FAISS Index Not Found**
**Error:**  
```
FileNotFoundError: No FAISS index found
```
✅ **Solution:**  
Re-upload a document to **rebuild the FAISS index**.

---

### **🔴 Streamlit Not Found**
**Error:**  
```
streamlit: command not found
```
✅ **Solution:**  
Activate your virtual environment:
```bash
conda activate doc-rag
```
Then reinstall Streamlit:
```bash
pip install streamlit
```

---

## **🎯 Future Enhancements**
- **Multi-document support**  
- **Improved UI with chatbot memory**  
- **Integration with GPT-4 or Claude for better responses**  

---

🎉 **You’re all set!** Start chatting with your documents now. 🚀

