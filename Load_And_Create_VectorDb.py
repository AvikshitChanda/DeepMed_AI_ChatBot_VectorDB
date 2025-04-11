
import glob
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter


pdf_paths = glob.glob("medical/*.pdf")  


all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    all_docs.extend(chunks)

print(f"✅ Total chunks created: {len(all_docs)}")


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectordb = FAISS.from_documents(all_docs, embeddings)


vectordb.save_local("medical_vector_db")

print("✅ Vector DB created and saved successfully!")
