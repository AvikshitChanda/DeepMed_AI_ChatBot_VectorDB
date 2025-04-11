# STEP 1: Load PDFs and create chunks
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# List of PDF files (use exact names from your folder)
pdf_paths = [
    "medical/10519-Abdominal-Pain.pdf",
    "medical/dengue1.pdf",
    "medical/dengue2.pdf",
    "medical/Interventions for the management of abdominal pain in Crohn's disease and inflammatory bowel disease (Review).pdf",
    "medical/the relationship between abdominal pain and emotional wellbeing in children and adolescents in the Raine Study.pdf",
]

# Load and split all PDFs into chunks
all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Split into 500 character chunks with overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    all_docs.extend(chunks)

print(f"✅ Total chunks created: {len(all_docs)}")


# STEP 2: Create vector embeddings and save in FAISS DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector DB from chunks
vectordb = FAISS.from_documents(all_docs, embeddings)

# Save vector DB locally
vectordb.save_local("medical_vector_db")

print("✅ Vector DB created and saved successfully!")
