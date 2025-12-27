import streamlit as st
import requests
from typing import List
import os

# Backend API URL - works for both local and production
API_URL = os.getenv("BACKEND_URL", "https://multi-rag-app.onrender.com")

# Page config
st.set_page_config(
    page_title="RAG Q&A with PDFs & URLs",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = "default_session"
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'content_uploaded' not in st.session_state:
    st.session_state.content_uploaded = False

def display_citations(citations: List[dict]):
    """Display citations in a clean format"""
    if not citations:
        return
    
    st.markdown("---")
    st.markdown("**📚 Sources:**")
    
    for i, citation in enumerate(citations, 1):
        source = citation.get('source', 'Unknown')
        page = citation.get('page')
        content = citation.get('content', '')
        
        # Format the citation
        if page is not None:
            citation_text = f"**[{i}]** {source} (Page {page + 1})"
        else:
            citation_text = f"**[{i}]** {source}"
        
        with st.expander(citation_text):
            st.caption(content)

# Title and description
st.title("📄 Conversational RAG with PDFs & URLs")
st.markdown("Upload PDFs or add website URLs and chat with their content using AI")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Groq API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Enter your Groq API key to use the chat. Get one at console.groq.com"
    )
    if api_key:
        st.session_state.api_key = api_key
    
    # Show link to get API key
    st.markdown("[🔑 Get your free Groq API key](https://console.groq.com)")
    
    # Session ID input
    session_id = st.text_input(
        "Session ID:",
        value=st.session_state.session_id,
        help="Unique identifier for your conversation"
    )
    st.session_state.session_id = session_id
    
    st.divider()
    
    # Create tabs for PDF and URL upload
    upload_tab, url_tab = st.tabs(["📤 Upload PDFs", "🌐 Add URLs"])
    
    with upload_tab:
        st.subheader("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if uploaded_files and st.button("Process PDFs", type="primary", key="process_pdfs"):
            with st.spinner("Processing PDFs..."):
                try:
                    # Reset file pointers
                    for file in uploaded_files:
                        file.seek(0)
                    
                    files = [
                        ("files", (file.name, file, "application/pdf"))
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(
                        f"{API_URL}/upload-pdfs",
                        files=files,
                        params={"session_id": st.session_state.session_id},
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"📄 Documents: {result.get('documents', 'N/A')} | 📦 Chunks: {result['chunks']}")
                        st.session_state.content_uploaded = True
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Error: {error_detail}")
                        
                        if "empty" in error_detail.lower():
                            st.warning("💡 Tip: Make sure your PDFs contain actual text, not just images.")
                        elif "image" in error_detail.lower():
                            st.warning("💡 Tip: This system works with text-based PDFs. Image-only PDFs require OCR.")
                            
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Try with smaller PDFs or fewer files.")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Cannot connect to backend at {API_URL}. Please check if the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error uploading files: {str(e)}")
    
    with url_tab:
        st.subheader("Add Website URLs")
        
        # URL input area
        url_input = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            help="Enter website URLs to extract and index their content",
            placeholder="https://example.com\nhttps://another-site.com"
        )
        
        if st.button("Process URLs", type="primary", key="process_urls"):
            if not url_input.strip():
                st.warning("⚠️ Please enter at least one URL")
            else:
                # Parse URLs
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                
                with st.spinner(f"Processing {len(urls)} URL(s)..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload-urls",
                            json={
                                "urls": urls,
                                "session_id": st.session_state.session_id
                            },
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.info(f"📦 Chunks created: {result['chunks']}")
                            
                            # Show successful URLs
                            if result.get('successful_urls'):
                                with st.expander("✅ Successfully processed URLs"):
                                    for url in result['successful_urls']:
                                        st.text(f"• {url}")
                            
                            # Show failed URLs if any
                            if result.get('failed_urls'):
                                with st.expander("❌ Failed URLs", expanded=True):
                                    for item in result['failed_urls']:
                                        st.error(f"• {item['url']}")
                                        st.caption(f"  Reason: {item['reason']}")
                            
                            st.session_state.content_uploaded = True
                        else:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"❌ Error: {error_detail}")
                            
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. Try with fewer URLs or simpler websites.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"🔌 Cannot connect to backend at {API_URL}. Please check if the backend is running.")
                    except Exception as e:
                        st.error(f"❌ Error processing URLs: {str(e)}")
    
    st.divider()
    
    # Clear session button
    if st.button("🗑️ Clear Session", help="Clear chat history and all documents"):
        try:
            response = requests.delete(
                f"{API_URL}/session/{st.session_state.session_id}"
            )
            if response.status_code == 200:
                st.session_state.messages = []
                st.session_state.content_uploaded = False
                st.success("Session cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing session: {str(e)}")
    
    # Show backend URL in development
    if "localhost" in API_URL:
        st.caption(f"🔧 Dev mode: {API_URL}")
    
    # Health check indicator
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("🟢 Backend connected")
        else:
            st.error("🔴 Backend unavailable")
    except:
        st.error(f"🔴 Backend unavailable at {API_URL}")

# Main chat interface
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to start chatting")
    st.info("🔑 Don't have an API key? Get one for free at [console.groq.com](https://console.groq.com)")
elif not st.session_state.content_uploaded:
    st.info("📤 Please upload PDF files or add website URLs in the sidebar to begin")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if available
            if message["role"] == "assistant" and "citations" in message:
                display_citations(message["citations"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents or websites..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "groq_api_key": st.session_state.api_key
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        citations = result.get("citations", [])
                        
                        st.markdown(answer)
                        display_citations(citations)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    else:
                        error_msg = response.json().get('detail', 'Unknown error')
                        st.error(f"Error: {error_msg}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Cannot connect to backend. Please check if the backend is running.")
                except Exception as e:
                    st.error(f"Error querying documents: {str(e)}")

# Footer
st.divider()
