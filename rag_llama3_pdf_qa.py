import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Set Groq API Key (no .env needed)
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# Load PDF
pdf_path = r"C:\Users\rapol\Downloads\DATA SCIENCE\6. FEB 25\31st - RAG Models\python_inbuildfunction.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split PDF text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Embed with HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Initialize LLaMA-3 with Groq
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

# Wrap in RetrievalQA for context-aware RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# Run a sample query
query = "What are the functions mentioned in this document?"
response = qa_chain.invoke({"query": query})

# Print answer and sources
print("\nAnswer:\n", response["result"])
print("\nSources:")
for doc in response.get("source_documents", []):
    print("•", doc.metadata.get("source", "Unknown source"))
