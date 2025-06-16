
#📚 Context-Aware RAG using LLaMA-3 and Groq API

This project demonstrates a **context-aware Retrieval-Augmented Generation (RAG)** system using:

- **LLaMA-3 (70B)** via **Groq API**
- **FAISS** vector store
- **LangChain**
- **HuggingFace Embeddings**
- **PDF Loader** for document ingestion

---

#🚀 Features

✅ Loads and processes PDF files  
✅ Splits documents into chunks  
✅ Creates embeddings using HuggingFace's `all-MiniLM-L6-v2`  
✅ Stores vector embeddings in FAISS  
✅ Uses LLaMA-3 via Groq to answer questions  
✅ Displays both answers and source references

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/your-username/context-aware-rag-groq-llama3.git
cd context-aware-rag-groq-llama3


### 2. Install Dependencies

Make sure you’re using **Python 3.10+** and a virtual environment.

bash
pip install -r requirements.txt


#3. Place Your PDF

Place your `.pdf` file in the project directory and edit the script to point to it:

```python
pdf_path = "your_pdf_file.pdf"
```

#4. Add Your Groq API Key

In `rag_llama3_pdf_qa.py`, replace:

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

with your actual [Groq API Key](https://console.groq.com/keys).

---

#🧠 Run the RAG Pipeline

```bash
python rag_llama3_pdf_qa.py
```

It will:

* Load the PDF
* Chunk and embed it
* Store embeddings
* Use LLaMA-3 to answer a query
* Print the answer and source document chunks

---

#🧪 Example Query

```python
query = "What are the main functions discussed in this document?"
```

**Output:**

```
Answer:
The document explains X, Y, and Z functions...

Sources:
• Page 3, your_pdf_file.pdf
• Page 7, your_pdf_file.pdf
```

---

#📦 File Structure

```
context-aware-rag-groq-llama3/
├── rag_llama3_pdf_qa.py           # Main RAG pipeline code
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```

---

#🤖 LLM Details

* **Model:** `llama3-70b-8192`
* **Provider:** [Groq](https://console.groq.com/)
* **Speed:** Ultra-fast inference (<1ms/token with Groq)

---

#🔍 Credits

* [LangChain](https://www.langchain.com/)
* [Groq](https://groq.com/)
* [Meta LLaMA-3](https://ai.meta.com/llama/)
* [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)

---

## 📬 License

MIT License - Free to use and modify.

