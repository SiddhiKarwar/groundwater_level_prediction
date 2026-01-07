"""
RAG Chatbot with PDF Upload Support
Conversational AI for Groundwater & Borewell Queries
"""

import os
# Avoid importing TensorFlow backend in Transformers to sidestep Keras 3 incompatibility
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Global storage
store = {}
vectorstore_global = None
conversational_chain_global = None

def initialize_llm():
    """Initialize Groq LLM"""
    groq_api_key = os.getenv("GROQ_API_TOKEN")
    if not groq_api_key:
        raise ValueError("GROQ_API_TOKEN not found in environment variables")
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",  # Updated to new supported model
        temperature=0.7
    )
    return llm

def process_pdf(pdf_file_path):
    """
    Process uploaded PDF file and create embeddings
    
    Args:
        pdf_file_path: Path to the PDF file
        
    Returns:
        vectorstore: Chroma vectorstore with document embeddings
    """
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_file_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content extracted from PDF")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings using HuggingFace (local, no server needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        return vectorstore, len(docs)
    
    except Exception as e:
        msg = str(e)
        if 'Keras is Keras 3' in msg or 'Keras 3' in msg:
            raise Exception(
                "Error processing PDF: Transformers detected Keras 3. To fix: either install the tf-keras shim (pip install tf-keras) or run with TensorFlow disabled by setting environment variables TRANSFORMERS_NO_TF=1 and USE_TF=0 before starting the app."
            )
        raise Exception(f"Error processing PDF: {msg}")

def create_conversational_chain(vectorstore):
    """
    Create conversational RAG chain
    
    Args:
        vectorstore: Chroma vectorstore
        
    Returns:
        conversational_rag_chain: Chain with chat history support
    """
    llm = initialize_llm()
    retriever = vectorstore.as_retriever()
    
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and latest user question "
        "which might reference context in the chat history, "
        "formulate the standalone question which can be understood "
        "without the chat history. Do not answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # QA system prompt - Enhanced for comprehensive project knowledge
    system_prompt = (
        "You are an expert AI assistant for the Groundwater Level Prediction System for Nashik District. "
        "You have access to comprehensive project knowledge including: "
        "1) NAQUIM water quality reports and uploaded PDFs, "
        "2) CGWB borewell database with 30+ documented borewells across Nashik, "
        "3) Historical groundwater, rainfall, river level, and temperature data, "
        "4) Trained ML models for predictions and recommendations, "
        "5) Master dataset with time-series data. "
        "\n\n"
        "Use the retrieved context below to answer questions about: "
        "- Groundwater levels and predictions "
        "- Borewell locations, depths, yields, and success rates "
        "- Water quality and NAQUIM reports "
        "- Best sites for new borewells "
        "- Historical data and trends "
        "- Model predictions and confidence levels "
        "- Specific locations in Nashik district (Nashik city, Malegaon, Sinnar, Igatpuri, etc.) "
        "\n\n"
        "Answer in a clear, detailed, and helpful manner. If asked about specific locations, "
        "provide data from the borewell database or master dataset. "
        "If asked about predictions or recommendations, explain the AI model's approach. "
        "If you don't have specific data, acknowledge it and provide related context. "
        "\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Session history management
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_rag_chain

def upload_and_process_pdf(file_storage, session_id="default", include_project_knowledge=True):
    """
    Handle PDF upload and processing, optionally including project knowledge
    
    Args:
        file_storage: Flask FileStorage object
        session_id: Session identifier for chat history
        include_project_knowledge: If True, load project datasets and models info
        
    Returns:
        dict: Status and info about processed PDF
    """
    global vectorstore_global, conversational_chain_global
    
    try:
        print(f"📄 RAG: Starting PDF processing for session {session_id}")
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, file_storage.filename)
        print(f"📄 RAG: Saving to temp path: {temp_pdf_path}")
        
        # Save uploaded file
        file_storage.save(temp_pdf_path)
        print(f"📄 RAG: File saved, starting PDF processing...")
        
        # Process PDF
        vectorstore, num_pages = process_pdf(temp_pdf_path)
        print(f"📄 RAG: PDF processed, {num_pages} pages")
        
        # Load project knowledge if requested
        all_docs = []
        if include_project_knowledge:
            print(f"📚 Loading project knowledge (datasets, models, borewells)...")
            try:
                from project_knowledge_loader import load_all_project_knowledge
                project_docs = load_all_project_knowledge()
                if project_docs:
                    # Add project docs to existing vectorstore
                    embeddings = vectorstore._embedding_function
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(project_docs)
                    vectorstore.add_documents(splits)
                    print(f"✅ Added {len(splits)} project knowledge chunks to vectorstore")
            except Exception as e:
                print(f"⚠️ Could not load project knowledge: {e}")
        
        vectorstore_global = vectorstore
        print(f"📄 RAG: Creating conversation chain...")
        
        # Create conversational chain
        conversational_chain_global = create_conversational_chain(vectorstore)
        print(f"📄 RAG: Conversation chain created")
        
        # Clear chat history for new document
        if session_id in store:
            store[session_id].clear()
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✅ RAG: PDF processing complete!")
        
        return {
            'success': True,
            'filename': file_storage.filename,
            'num_pages': num_pages,
            'message': f'PDF processed successfully! {num_pages} pages loaded. Project knowledge loaded.'
        }
    
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def ask_question(question, session_id="default"):
    """
    Ask question to RAG chatbot
    
    Args:
        question: User question
        session_id: Session identifier for chat history
        
    Returns:
        dict: Answer and status
    """
    global conversational_chain_global
    
    try:
        if conversational_chain_global is None:
            return {
                'success': False,
                'error': 'Please upload a PDF document first. No vectorstore is loaded yet.'
            }
        
        # Invoke conversational chain
        response = conversational_chain_global.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        return {
            'success': True,
            'answer': response['answer'],
            'question': question
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_chat_history(session_id="default"):
    """
    Get chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        list: Chat history messages
    """
    if session_id not in store:
        return []
    
    history = store[session_id].messages
    chat_list = []
    
    for msg in history:
        if isinstance(msg, HumanMessage):
            chat_list.append({'type': 'human', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            chat_list.append({'type': 'ai', 'content': msg.content})
    
    return chat_list

def clear_chat_history(session_id="default"):
    """Clear chat history for a session"""
    if session_id in store:
        store[session_id].clear()
    return {'success': True, 'message': 'Chat history cleared'}

def reset_system():
    """Reset the entire RAG system"""
    global vectorstore_global, conversational_chain_global, store
    
    vectorstore_global = None
    conversational_chain_global = None
    store = {}
    
    return {'success': True, 'message': 'System reset successfully'}
