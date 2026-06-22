# RAG PDF Assistant

A simple Streamlit application that turns PDF documents into a retrieval-augmented generation (RAG) assistant. Upload or point to a research PDF, then ask natural language questions and receive answers based on the PDF content.

## Features

- PDF ingestion using `PyPDFLoader`
- Text chunking with `RecursiveCharacterTextSplitter`
- Vector embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector store for fast retrieval
- Conversational question answering powered by Groq and `llama-3.1-8b-instant`
- Interactive Streamlit UI with chat-style input and response display

## Requirements

- Python 3.10+ (recommended)
- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-huggingface`
- `sentence-transformers`
- `faiss-cpu`
- `groq`
- `pypdf`
- `python-dotenv`

>The repository includes a `requirements.txt` file. Please verify that `stramlit` is corrected to `streamlit` if needed.

## Setup

1. Clone or open the project folder.
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

4. Create a `.env` file in the project root and add your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Place the target PDF file in the project folder, or set the full path in the sidebar.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the browser link shown by Streamlit.
4. Enter the PDF path and choose chunk size / overlap settings in the sidebar.
5. Ask questions about the PDF in the chat input.

## Configuration

- `PDF Path`: Local path to the PDF file to index.
- `Chunk Size`: Maximum number of characters per text chunk.
- `Chunk Overlap`: Number of overlapping characters between chunks.

## Notes

- The app caches the FAISS vector store using Streamlit resource caching.
- Questions are answered using retrieved PDF context and Groq's chat completion model.
- If the app cannot load the PDF, an error message is displayed.

## Files

- `app.py`: Main Streamlit application.
- `requirements.txt`: Python dependencies.
- `rag_demo.ipynb`: Notebook demonstration (optional).

## License

This project is provided as-is. Add a license file if you want to share it publicly.
