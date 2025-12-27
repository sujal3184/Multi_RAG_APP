from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
import tempfile
import shutil

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
session_stores = {}
vectorstores = {}
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    return session_stores[session_id]

def add_documents_to_vectorstore(session_id: str, documents: List[Document]):
    """Add documents to existing or new vectorstore"""
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
    
    # Create or update vectorstore
    if session_id in vectorstores:
        # Add to existing vectorstore
        vectorstores[session_id].add_documents(splits)
    else:
        # Create new vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=f"collection_{session_id}"
        )
        vectorstores[session_id] = vectorstore
    
    return len(splits)

def extract_citations(source_documents: List[Document]) -> List[dict]:
    """Extract citation information from source documents"""
    citations = []
    seen = set()
    
    for doc in source_documents:
        # Get source information
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page')
        
        # Create unique identifier
        citation_id = f"{source}_{page}" if page is not None else source
        
        # Avoid duplicate citations
        if citation_id not in seen:
            seen.add(citation_id)
            
            # Extract file name from path for PDFs
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
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Add filename to metadata
            for doc in docs:
                doc.metadata['source'] = file.filename
            
            documents.extend(docs)
            
            # Clean up
            os.unlink(tmp_path)
        
        # Check if documents were loaded
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail="No content extracted from PDFs. Please check if PDFs contain text."
            )
        
        # Add documents to vectorstore
        chunks = add_documents_to_vectorstore(session_id, documents)
        
        return {
            "status": "success",
            "message": f"Processed {len(files)} PDF files",
            "chunks": chunks,
            "documents": len(documents)
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
                # Load web content
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                if docs and docs[0].page_content.strip():
                    # Ensure URL is in metadata
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
        
        # Add documents to vectorstore
        chunks = add_documents_to_vectorstore(request.session_id, documents)
        
        response = {
            "status": "success",
            "message": f"Processed {len(successful_urls)} URLs successfully",
            "chunks": chunks,
            "successful_urls": successful_urls
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
        
        if session_id not in vectorstores:
            raise HTTPException(
                status_code=400, 
                detail="No documents uploaded for this session. Please upload PDFs or URLs first."
            )
        
        # Initialize LLM
        llm = ChatGroq(groq_api_key=request.groq_api_key, model_name="llama-3.3-70b-versatile")
        
        # Get retriever with more results for better citations
        retriever = vectorstores[session_id].as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )
        
        # Contextualize question prompt
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
        
        # Answer prompt
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
        
        # Get response
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Extract citations from context documents
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
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's data"""
    try:
        if session_id in session_stores:
            del session_stores[session_id]
        if session_id in vectorstores:
            del vectorstores[session_id]
        return {"status": "success", "message": "Session cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)