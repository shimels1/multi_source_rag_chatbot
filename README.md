# Multi-Source RAG Chatbot

A **conversational AI assistant** built with Streamlit, LangChain, FAISS, and Groq LLM (Llama 3) that answers questions based on document context.

---

## Features

- Conversational AI with memory for context-aware responses.
- Retrieves answers from uploaded documents using **RAG (Retrieval-Augmented Generation)**.
- Embeddings created with `sentence-transformers/all-MiniLM-L6-v2`.
- Modern, responsive chat interface in Streamlit.
- Typing indicator and user/assistant message bubbles for enhanced UX.
- No persistent FAISS storage; all embeddings are in-memory for simplicity.

---

## Technologies

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq LLM](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)
- [dotenv](https://pypi.org/project/python-dotenv/)

---

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/shimels1/multi_source_rag_chatbot.git
cd multi_source_rag_chatbot
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Environment variables (.env):**

Create a `.env` file in the root directory:

```env
LANGCHAIN_API_KEY=<your_langchain_api_key>
GROQ_API_KEY=<your_groq_api_key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=RAG_QA_Project
```

4. **Add your documents:**

Place your text files in `data/data.txt` or update the path in the code.

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

- Type your questions in the chat box.
- The AI responds using document context.
- Memory keeps track of recent interactions.

---

## Project Structure

```
├── app.py                 # Main Streamlit app
├── data/
│   └── data.txt           # Document source
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Notes

- The FAISS index is **in-memory**, so data is lost when the app stops.
- The model only answers based on the provided document context; it will not hallucinate.
- Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` for large documents.

---

## License

This project is licensed under the MIT License. See the full license text below.

```
MIT License

Copyright (c) 2025 Shimels Alem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**GitHub Repository:** [https://github.com/shimels1/multi_source_rag_chatbot](https://github.com/shimels1/multi_source_rag_chatbot)
