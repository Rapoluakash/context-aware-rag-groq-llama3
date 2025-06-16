from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ✅ Replace this with either a public URL or repo-relative file path
pdf_path = "https://arxiv.org/pdf/2306.17806.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

llm = ChatGroq(model="llama3-70b-8192", api_key="your_groq_api_key")  # 👈 Replace with your actual API key

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

query = "List all the functions in the PDF."
response = qa.invoke({"query": query})
print(response["result"])
