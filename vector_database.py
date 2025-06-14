import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_SAVE_DIR = "pdfs"
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_SAVE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_and_split_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def process_pdf_to_vectorstore(uploaded_file):
    file_path = save_uploaded_file(uploaded_file)
    chunks = load_and_split_pdf(file_path)
    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore
