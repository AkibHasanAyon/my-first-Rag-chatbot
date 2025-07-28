import streamlit as st
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import google.api_core.exceptions
from gemini_setup import model, system_prompt

# Title
st.title("📄 RAG PDF Chatbot")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

@st.cache_data(show_spinner=False)
def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text("text")
    doc.close()
    return all_text

@st.cache_data(show_spinner=False)
def chunk_text(text, max_tokens=300, overlap=50):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, chunk = [], []
    total_words = 0
    for sentence in sentences:
        words = sentence.split()
        if total_words + len(words) > max_tokens:
            chunks.append(' '.join(chunk))
            if overlap > 0:
                chunk = ' '.join(chunk[-overlap:]).split()
                total_words = len(chunk)
            else:
                chunk, total_words = [], 0
        chunk.extend(words)
        total_words += len(words)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

@st.cache_resource(show_spinner=False)
def embed_chunks(chunks):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model, embedding_model.encode(chunks)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, embeddings, embedding_model, k=2):
    query_embedding = embedding_model.encode([query])[0]
    scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(embeddings)]
    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return [chunks[i] for i, _ in top_indices]

def generate_response(system_prompt, top_chunks, query, max_chunk_chars=1200, max_retries=3):
    truncated_chunks = [c[:max_chunk_chars] for c in top_chunks]
    user_prompt = "\n".join([f"Context {i+1}:\n{c}\n=====================" for i, c in enumerate(truncated_chunks)])
    user_prompt += f"\n\nQuestion: {query}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    for attempt in range(1, max_retries+1):
        try:
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except google.api_core.exceptions.ResourceExhausted:
            wait = 10 * attempt
            st.warning(f"Rate limit hit, retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            return f"❌ ERROR: {str(e)}"

    return "❌ ERROR: Failed after multiple retries"

# Once PDF is uploaded
if pdf_file:
    st.success("✅ PDF uploaded. Processing...")
    text = extract_pdf_text(pdf_file)
    chunks = chunk_text(text)
    embedding_model, embeddings = embed_chunks(chunks)
    st.success(f"✅ Processed PDF into {len(chunks)} chunks.")

    # Chat interface
    st.subheader("💬 Ask me anything from your PDF")
    user_query = st.text_input("Type your question:", key="query")

    if user_query:
        with st.spinner("Generating answer..."):
            top_chunks = semantic_search(user_query, chunks, embeddings, embedding_model, k=2)
            answer = generate_response(system_prompt, top_chunks, user_query)
        st.markdown("---")
        st.markdown(f"**🔍 Question:** {user_query}")
        st.markdown(f"**🤖 Answer:** {answer}")
