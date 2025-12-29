from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tempfile
import shutil
from datetime import datetime, timedelta

load_dotenv()

app = FastAPI(title="RAG API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global storage - only chat history in memory now
session_stores = {}
session_last_activity = {}

# Initialize Qdrant client
qdrant_client = None
embeddings = None

def get_qdrant_client():
    """Initialize Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise HTTPException(
                status_code=500,
                detail="QDRANT_URL and QDRANT_API_KEY must be set in environment variables"
            )
        
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
    return qdrant_client

def get_embeddings():
    """Initialize HuggingFace embeddings (runs locally)"""
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
    return embeddings

class QueryRequest(BaseModel):
    question: str
    session_id: str
    groq_api_key: str

class URLRequest(BaseModel):
    urls: List[str]
    session_id: str = "default_session"

class ChatHistoryResponse(BaseModel):
    messages: List[dict]

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    content: str

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    session_last_activity[session_id] = datetime.now()
    return session_stores[session_id]

def get_collection_name(session_id: str) -> str:
    """Generate collection name for session"""
    # Qdrant collection names must start with letter and contain only letters, digits, hyphens, underscores
    return f"session_{session_id.replace('-', '_')}"

def ensure_collection_exists(session_id: str):
    """Create Qdrant collection if it doesn't exist"""
    client = get_qdrant_client()
    collection_name = get_collection_name(session_id)
    
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Changed to 384 for HuggingFace
        )

def add_documents_to_vectorstore(session_id: str, documents: List[Document]):
    """Add documents to Qdrant vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, 
        chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)
    
    # Filter out empty documents
    splits = [doc for doc in splits if doc.page_content.strip()]
    
    if not splits:
        raise HTTPException(
            status_code=400,
            detail="All document chunks are empty."
        )
    
    collection_name = get_collection_name(session_id)
    
    # Check if collection exists and delete if it has wrong dimensions
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    force_recreate = False
    if collection_name in collection_names:
        # Collection exists, check if we need to recreate
        try:
            collection_info = client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size
            if vector_size != 384:  # HuggingFace embedding size
                print(f"Collection has wrong dimensions ({vector_size}), recreating...")
                force_recreate = True
        except:
            force_recreate = True
    
    # Add documents to Qdrant
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
        force_recreate=force_recreate
    )
    
    session_last_activity[session_id] = datetime.now()
    
    return len(splits)

def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """Get vector store for a session"""
    collection_name = get_collection_name(session_id)
    
    # Check if collection exists
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name not in collection_names:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded for this session. Please upload PDFs or URLs first."
        )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings()
    )
    
    session_last_activity[session_id] = datetime.now()
    
    return vectorstore

def extract_citations(source_documents: List[Document]) -> List[dict]:
    """Extract citation information from source documents"""
    citations = []
    seen = set()
    
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page')
        
        citation_id = f"{source}_{page}" if page is not None else source
        
        if citation_id not in seen:
            seen.add(citation_id)
            
            if source.endswith('.pdf') or '/' in source or '\\' in source:
                source_name = os.path.basename(source)
            else:
                source_name = source
            
            citation = {
                "source": source_name,
                "page": page,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            citations.append(citation)
    
    return citations

def cleanup_old_sessions():
    """Remove chat history for sessions inactive for > 2 hours"""
    cutoff = datetime.now() - timedelta(hours=2)
    to_delete = [
        sid for sid, last_time in session_last_activity.items()
        if last_time < cutoff
    ]
    for sid in to_delete:
        if sid in session_stores:
            del session_stores[sid]
        del session_last_activity[sid]

@app.middleware("http")
async def cleanup_middleware(request, call_next):
    """Cleanup old sessions on each request"""
    cleanup_old_sessions()
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG API v2.0 is running (Qdrant Cloud + HuggingFace Embeddings)",
        "status": "healthy",
        "storage": "Qdrant Cloud (external)",
        "embeddings": "HuggingFace (local)",
        "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
        "embedding_dimensions": 384,
        "endpoints": {
            "health": "/health",
            "upload_pdfs": "/upload-pdfs",
            "upload_urls": "/upload-urls",
            "query": "/query",
            "chat_history": "/chat-history/{session_id}",
            "clear_session": "/session/{session_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        return {
            "status": "healthy",
            "service": "RAG API v2.0",
            "active_chat_sessions": len(session_stores),
            "qdrant_collections": len(collections.collections),
            "storage": "Qdrant Cloud",
            "embeddings": "HuggingFace Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: str = "default_session"
):
    """Upload and process PDF files"""
    try:
        documents = []
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata['source'] = file.filename
                
                documents.extend(docs)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail="No content extracted from PDFs. Please check if PDFs contain text."
            )
        
        chunks = add_documents_to_vectorstore(session_id, documents)
        
        return {
            "status": "success",
            "message": f"Processed {len(files)} PDF files",
            "chunks": chunks,
            "documents": len(documents),
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/upload-urls")
async def upload_urls(request: URLRequest):
    """Load and process content from URLs"""
    try:
        if not request.urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        documents = []
        successful_urls = []
        failed_urls = []
        
        for url in request.urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                if docs and docs[0].page_content.strip():
                    for doc in docs:
                        doc.metadata['source'] = url
                    documents.extend(docs)
                    successful_urls.append(url)
                else:
                    failed_urls.append({"url": url, "reason": "No content extracted"})
                    
            except Exception as e:
                failed_urls.append({"url": url, "reason": str(e)})
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No content extracted from any URLs. Failed: {failed_urls}"
            )
        
        chunks = add_documents_to_vectorstore(request.session_id, documents)
        
        response = {
            "status": "success",
            "message": f"Processed {len(successful_urls)} URLs successfully",
            "chunks": chunks,
            "successful_urls": successful_urls,
            "session_id": request.session_id
        }
        
        if failed_urls:
            response["failed_urls"] = failed_urls
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the documents with chat history and return citations"""
    try:
        session_id = request.session_id
        
        # Get vectorstore from Qdrant
        vectorstore = get_vectorstore(session_id)
        
        llm = ChatGroq(
            groq_api_key=request.groq_api_key, 
            model_name="llama-3.3-70b-versatile"
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        
        # Contextualize question based on chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
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
        
        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": session_id}}
        )
        
        citations = extract_citations(response.get('context', []))
        
        return {
            "answer": response['answer'],
            "session_id": session_id,
            "citations": citations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = get_session_history(session_id)
        messages = [
            {
                "type": msg.type,
                "content": msg.content
            }
            for msg in history.messages
        ]
        return {"messages": messages, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's data (chat history and vector store)"""
    try:
        # Clear chat history
        if session_id in session_stores:
            del session_stores[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]
        
        # Delete Qdrant collection
        try:
            client = get_qdrant_client()
            collection_name = get_collection_name(session_id)
            client.delete_collection(collection_name)
        except Exception as e:
            # Collection might not exist, that's okay
            pass
        
        return {
            "status": "success", 
            "message": "Session cleared (chat history and documents)",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
