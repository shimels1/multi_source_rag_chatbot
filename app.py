from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_unstructured import UnstructuredLoader
import os
import logging
from dotenv import load_dotenv
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded.")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RAG_QA_Project")

logger.info("LangChain environment variables set.")

# Initialize components
vectorstore = None
retriever = None
qa = None

# Create data directory if it doesn't exist
data_dir = os.path.abspath("data")
os.makedirs(data_dir, exist_ok=True)
logger.info("Data directory created/verified at: %s", data_dir)

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
logger.info("Groq LLM initialized.")

# Initialize memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=7,
    input_key="input",
    output_key="answer"
)

# Query-rewrite prompt for history-aware retriever
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone query using the chat history. "
     "Do not answer the question. Only rewrite it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Answer prompt for the final response
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful and accurate conversational AI assistant. "
     "Use the conversation history to recall previous interactions. "
     "Answer questions based solely on the provided document context from multiple sources. "
     "Do NOT make up information. "
     "If the answer is not in the conversation history or document context, respond exactly: "
     "'I lack sufficient information to answer that.'\n\n"
     "Important: Synthesize information from all relevant sources in the context. "
     "Answer concisely and accurately based only on the above."),
    MessagesPlaceholder("chat_history"),
    ("human",
     "Document Context from multiple sources:\n{context}\n\n"
     "User Question: {input}\n\n"
     "Answer:")
])

# Function to process documents and update vectorstore
def process_documents():
    global vectorstore, retriever, qa
    try:
        docs = []
        supported_formats = ('.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.eml', '.msg', '.rtf', '.odt', '.html')
        file_count = 0
        
        logger.info("Scanning data directory: %s", data_dir)
        if not os.path.exists(data_dir):
            logger.error("Data directory does not exist at: %s", data_dir)
            return
        sorted_files = sorted(os.listdir(data_dir))
        logger.info("Files found: %s", sorted_files)
        for file in sorted_files:
            if file.lower().endswith(supported_formats):
                file_path = os.path.join(data_dir, file)
                logger.info("Attempting to process: %s", file_path)
                try:
                    loader = UnstructuredLoader(file_path, encoding="utf-8", mode="single")
                    file_docs = loader.load()
                    logger.info("Loaded %d documents from: %s", len(file_docs), file)
                    for doc in file_docs:
                        doc.metadata["source"] = file
                    docs.extend(file_docs)
                    file_count += 1
                except Exception as e:
                    logger.error("Error processing %s: %s", file, str(e))
        
        logger.info("Total %d documents loaded from %d files in data directory.", len(docs), file_count)
        
        if docs:
            # Split each document individually to preserve metadata
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=500,
                add_start_index=True
            )
            chunks = []
            for doc in docs:
                # Split each document and preserve its metadata
                doc_chunks = splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            logger.info("%d chunks created from %d documents.", len(chunks), len(docs))

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            logger.info("FAISS index created in memory.")
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            logger.info("Base retriever set up.")
            
            # Build history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=retriever,
                prompt=contextualize_q_prompt
            )
            logger.info("History-aware retriever created.")
            
            # Create document chain
            doc_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=answer_prompt
            )
            logger.info("Stuff documents chain created.")
            
            # Final RAG chain
            qa = create_retrieval_chain(
                history_aware_retriever,
                doc_chain
            )
            logger.info("Retrieval chain initialized (history-aware RAG).")
        else:
            logger.warning("No documents loaded. Vectorstore not initialized.")
            vectorstore = None
            retriever = None
            qa = None
    except Exception as e:
        logger.error("Error in process_documents: %s", str(e))
        vectorstore = None
        retriever = None
        qa = None

# FastAPI app
app = FastAPI()

# Pydantic model for ask endpoint
class AskRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return HTMLResponse(open("index.html").read())

# List files in data directory
@app.get("/list-files")
async def list_files():
    files = []
    supported_formats = ('.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.eml', '.msg', '.rtf', '.odt', '.html')
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith(supported_formats):
            files.append({
                "filename": file,
                "size": os.path.getsize(file_path)
            })
    return JSONResponse({"files": files})

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        supported_formats = ('.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.eml', '.msg', '.rtf', '.odt', '.html')
        if not file.filename.lower().endswith(supported_formats):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            )
        
        file_path = os.path.join(data_dir, file.filename)
        
        if os.path.exists(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} already exists in data directory."
            )
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info("File uploaded: %s", file.filename)
        process_documents()
        
        return JSONResponse({
            "message": f"File {file.filename} uploaded successfully",
            "filename": file.filename,
            "size": os.path.getsize(file_path)
        })
    except Exception as e:
        logger.error("Error uploading file: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# Ask endpoint
@app.post("/ask")
async def ask_question(request: AskRequest):
    if not qa:
        raise HTTPException(
            status_code=400,
            detail="QA system not ready. Add documents to the data directory and restart server."
        )
    
    try:
        # Load chat history from memory
        chat_vars = memory.load_memory_variables({})
        chat_history = chat_vars.get("chat_history", [])
        
        # Invoke the chain with input and chat_history
        result = qa.invoke({
            "input": request.question,
            "chat_history": chat_history
        })
        
        answer = result.get("answer", "I couldn't find an answer to that question.")
        context_docs = result.get("context", [])  # List[Document]
        
        # Summarize sources
        sources_used = {}
        for doc in context_docs:
            src = (doc.metadata or {}).get("source", "unknown")
            sources_used[src] = sources_used.get(src, 0) + 1
        
        logger.info("Q: %s", request.question)
        logger.info("A: %s", answer)
        logger.info("Sources used: %s", sources_used)
        
        # Save to memory
        memory.save_context({"input": request.question}, {"answer": answer})
        
        return JSONResponse({
            "question": request.question,
            "answer": answer,
            "sources_used": sources_used
        })
    except Exception as e:
        logger.error("Error during QA: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Serve files for download
@app.get("/data/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

# Process documents on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting document processing.")
    process_documents()