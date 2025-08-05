import streamlit as st
import openai
import tempfile
import os
import fitz  # PyMuPDF
import docx
import pandas as pd
import base64
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import numpy as np
import openpyxl
from dotenv import load_dotenv
load_dotenv()



# === Azure OpenAI Configuration ===
openai.api_type = "azure"
openai.api_base = "https://arlotest.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  

embedding_deployment = "text-embedding-ada-002-test1"
chat_deployment = "gpt-4-cedo"

# === Helper Functions ===
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text(file):
    text = ""
    filename = file.name
    if filename.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file)
        text = df.to_string(index=False)
    return text

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        deployment_id=embedding_deployment
    )
    return response["data"][0]["embedding"]

def find_top_chunks(question, embedded_chunks, top_k=3):
    question_embedding = get_embedding(question)
    similarities = []
    for item in embedded_chunks:
        sim = cosine_similarity([question_embedding], [item["embedding"]])[0][0]
        similarities.append((item["chunk"], item.get("filename", "Unknown"), item.get("page", "N/A"), sim))
    top_chunks = sorted(similarities, key=lambda x: x[3], reverse=True)[:top_k]
    return top_chunks

def ask_question(question, embedded_chunks):
    top_chunks = find_top_chunks(question, embedded_chunks)
    context = "\n\n".join([c[0] for c in top_chunks])
    sources = [(c[1], c[2]) for c in top_chunks]
    prompt = f"""You are an assistant. Answer the question using only the following content:

{context}

Question: {question}
Answer:"""
    response = openai.ChatCompletion.create(
        deployment_id=chat_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful  assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )
    return response["choices"][0]["message"]["content"], sources

# === Streamlit UI ===
st.set_page_config(page_title="📁 Document summary Chatbot", layout="centered")
st.title("⚖️  Document Assistant")
st.markdown("Ask questions based on uploaded  documents (PDF, Word, Excel).")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = set()

uploaded_files = st.file_uploader("📄 Upload documents", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.files_processed]
    if new_files:
        with st.spinner("Processing uploaded files..."):
            for file in new_files:
                text = extract_text(file)
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    st.session_state.embedded_chunks.append({
                        "chunk": chunk,
                        "embedding": embedding,
                        "filename": file.name,
                        "page": i + 1
                    })
                st.session_state.files_processed.add(file.name)
        st.success("✅ New files processed successfully!")

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("💬 Ask your question:", key="question_input")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    if not st.session_state.embedded_chunks:
        st.warning("⚠️ Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_question(question, st.session_state.embedded_chunks)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")

# === Clear chat ===
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.embedded_chunks = []
    st.session_state.files_processed = set()
    st.success("🧹 Chat history cleared!")

# === Show history ===
for idx, entry in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"### 🧠 Q{len(st.session_state.chat_history)-idx}: {entry['question']}")
    st.write(entry["answer"])
    st.markdown("📚 **Sources:**")
    for filename, page in entry["sources"]:
        st.markdown(f"- **{filename}**, Page {page}")
    st.markdown("---")