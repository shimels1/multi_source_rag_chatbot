# Multi-Source RAG Chatbot

A conversational AI assistant built with FastAPI, LangChain, FAISS, and Groq LLM (Llama 3) that answers questions based on document context, featuring a modern web interface.

## Features

- **Conversational AI with Memory**: Provides context-aware responses using `ConversationBufferWindowMemory` along with a **history-aware retriever** to rewrite queries based on past interactions.
- **Multi-Source RAG**: Retrieves answers from uploaded documents using Retrieval-Augmented Generation (RAG).
- **Embeddings**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for high-quality text embeddings.
- **Modern Web UI**: Features a responsive interface with Tailwind CSS, glassmorphism effects, and Font Awesome icons.
- **Multi-Format Support**: Handles `.txt`, `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.eml`, `.msg`, `.rtf`, `.odt`, and `.html` files.
- **In-Memory FAISS**: Stores embeddings in memory for simplicity (data is lost on server restart).
- **File Management**: Supports file uploads and downloads via API endpoints.
- **No Hallucination**: Answers are based solely on document context; returns "I lack sufficient information to answer that" otherwise.
## Screenshots
  ![Chat Interface](https://raw.githubusercontent.com/shimels1/multi_source_rag_chatbot/refs/heads/main/screenshot/1.png)
  ![Upload Files](https://raw.githubusercontent.com/shimels1/multi_source_rag_chatbot/refs/heads/main/screenshot/2.png)
  ![File List](https://raw.githubusercontent.com/shimels1/multi_source_rag_chatbot/refs/heads/main/screenshot/3.png)
## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shimels1/multi_source_rag_chatbot.git
   cd multi_source_rag_chatbot
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   For non-text file support (e.g., `.pdf`, `.docx`), install additional dependencies:
   - **Ubuntu**: `sudo apt-get install tesseract-ocr poppler-utils`
   - **Mac**: `brew install tesseract poppler`

3. **Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```env
   GROQ_API_KEY=<your_groq_api_key>
   LANGCHAIN_API_KEY=<your_langchain_api_key>
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_PROJECT=RAG_QA_Project
   ```
   Obtain API keys from [Groq](https://console.groq.com/) and [LangChain](https://smith.langchain.com/).

4. **Prepare Documents**:
   Place your documents in the `data/` directory. Supported formats include `.txt`, `.pdf`, `.docx`, etc. The server will process these on startup.

## Usage

1. **Run the FastAPI Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the Web Interface**:
   - Open your browser and navigate to `http://localhost:8000`.
   - You’ll see a chat interface with sections for chatting, uploading files, and viewing files.

3. **Interact with the Chatbot**:
   - Type questions in the chat box and press "Send".
   - The AI responds based on uploaded document context.
   - Memory tracks recent interactions for context, and the **history-aware retriever** rewrites queries for better answers.

4. **Manage Files**:
   - Use the "Upload" section to add new documents.
   - View and download files in the "Files" section.

## Project Structure

```
├── main.py                 # FastAPI backend with document processing and API endpoints
├── index.html              # Modern web UI with Tailwind CSS and JavaScript
├── data/                   # Directory for uploaded documents
├── screenshot/             # Folder containing screenshots for README and UI
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not tracked by git)
└── README.md               # This file
```

## License

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

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.

## Contact

For questions or support, reach out to Shimels Alem via the GitHub repository: [https://github.com/shimels1/multi_source_rag_chatbot](https://github.com/shimels1/multi_source_rag_chatbot).

## Support

If you find this project useful, please give it a ⭐ on [GitHub](https://github.com/shimels1/multi_source_rag_chatbot)!  
Your support helps others discover the project and keeps development going.
