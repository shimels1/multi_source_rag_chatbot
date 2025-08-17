import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
import logging

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

# Load documents
try:
    loader = TextLoader("data/data.txt", encoding="utf-8")
    docs = loader.load()
    logger.info(f"{len(docs)} documents loaded.")
except Exception as e:
    logger.error(f"Error loading documents: {e}")
    docs = []

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logger.info(f"{len(chunks)} chunks created from documents.")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.info("Embeddings initialized.")

# FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
logger.info("FAISS index created in memory.")

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
logger.info("Retriever set up.")

# Set up Groq LLM - USING BEST AVAILABLE MODEL
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",  # Upgraded to best model
    temperature=0
)
logger.info("Groq LLM initialized.")

# Initialize session state variables
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=20,
        input_key="question",
        output_key="answer"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt template
prompt = PromptTemplate(
    template=(
        "You are a helpful and accurate conversational AI assistant. "
        "Use the conversation history to recall personal information about the user. "
        "Answer questions based solely on the provided document context. "
        "Do NOT make up information. If the answer is not in the context, respond clearly with: 'I lack sufficient information to answer that.'\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Document Context:\n{context}\n\n"
        "User Question: {question}\n\n"
        "Answer concisely and accurately:"
    ),
    input_variables=["chat_history", "context", "question"]
)

logger.info("Prompt template initialized.")

# Build RetrievalQA chain
try:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=st.session_state.memory,  # Use session state memory
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer"
    )
    logger.info("ConversationalRetrievalChain initialized.")
except Exception as e:
    logger.error(f"Error initializing ConversationalRetrievalChain: {e}")
    qa = None

# ============================ 
# Streamlit Chat Handling - Modernized & Fixed
# ============================
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern chat container */
    .stChatFloatingInputContainer {
        background-color: #f8f9fa;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Message bubbles */
    [data-testid="stChatMessage"] {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 18px;
        max-width: 85%;
    }
    
    /* User message bubble */
    [data-testid="stChatMessage"][aria-label="user"] > div {
        background-color: #3b82f6;
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    
    /* Assistant message bubble */
    [data-testid="stChatMessage"][aria-label="assistant"] > div {
        background-color: #f1f5f9;
        color: #1e293b;
        border-bottom-left-radius: 4px;
        margin-right: auto;
    }
    
    /* Typing animation */
    @keyframes typing {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    
    .typing-indicator {
        display: flex;
        padding: 10px 15px;
    }
    
    .typing-dot {
        height: 8px;
        width: 8px;
        background-color: #94a3b8;
        border-radius: 50%;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        padding: 1rem 0 1.5rem;
    }
    
    .header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header p {
        color: #64748b;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Modern header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown('<h1>AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p>Powered by Groq & Llama 3 - Ask me anything!</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! I'm your AI assistant. How can I help you today?"}
    ]

# Display chat messages (only once)
for message in st.session_state.chat_history:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input
if prompt_text := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_text)

    # Display typing indicator while processing
    with st.chat_message("assistant", avatar="🤖"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>', 
            unsafe_allow_html=True
        )
    
    # Process with QA chain
    answer = ""
    if qa:
        try:
            # Run the chain
            result = qa.invoke({"question": prompt_text})
            answer = result.get("answer", "I couldn't find an answer to that question.")
            
            # Update memory with full interaction
            st.session_state.memory.save_context(
                {"question": prompt_text},
                {"answer": answer}
            )
            
            # Log the full memory state
            logger.info(f"Current memory state: {st.session_state.memory.load_memory_variables({})}")
        except Exception as e:
            logger.error(f"Error during QA invoke: {e}")
            answer = "⚠️ Sorry, I encountered an error processing your request."
    else:
        answer = "🔧 QA system is not available. Please try again later."
    
    # Remove typing indicator and show response
    typing_placeholder.empty()
    
    # Add assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Display the new assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
    
    # Auto-scroll to bottom
    st.markdown(
        """
        <script>
            window.scrollTo(0, document.body.scrollHeight);
        </script>
        """,
        unsafe_allow_html=True
    )
