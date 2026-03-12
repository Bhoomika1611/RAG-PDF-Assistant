import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Assistant")
st.markdown("Ask questions from your research PDF using AI.")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sidebar
st.sidebar.header("Settings")
pdf_path = st.sidebar.text_input(
    "PDF Path",
    "research_pdf.pdf"
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# Load and process PDF
@st.cache_resource
def load_vector_store(path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# Load database
try:
    vector_store = load_vector_store(pdf_path, chunk_size, chunk_overlap)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about the PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("Built with LangChain, FAISS, Groq, and Streamlit.")