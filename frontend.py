import streamlit as st
from rag_pipeline import answer_query
from vector_database import process_pdf_to_vectorstore

st.set_page_config(page_title="AI Lawyer", layout="centered")

st.title("⚖️ AI Lawyer using RAG + DeepSeek ")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query = st.text_area("Ask your legal question", height=150, placeholder="e.g. What is Article 21?")

if st.button("Ask AI Lawyer"):
    if uploaded_file:
        st.chat_message("user").write(user_query)

        vectorstore = process_pdf_to_vectorstore(uploaded_file)
        retrieved_docs = vectorstore.similarity_search(user_query)

        response = answer_query(documents=retrieved_docs, query=user_query)
        st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Please upload a PDF file before asking.")
