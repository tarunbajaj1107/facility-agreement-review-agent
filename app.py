import os
import streamlit as st
from ingest import process_and_store_document, get_db_stats
from analyzer import review_agreement
from config import UPLOAD_DIR

st.set_page_config(
    page_title="Facility Agreement Review Agent",
    page_icon="⚖️",
    layout="wide"
)

# Sidebar Configuration
st.sidebar.title("🔑 Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get your free key at https://console.groq.com")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Train precedents first in Tab 1, then run your agreement review in Tab 2.")

st.title("⚖️ AI Facility Agreement Review Agent")
st.markdown("Open-source precedent-based contract intelligence for debt & project finance.")

# Tabs
tab1, tab2 = st.tabs(["📚 1. Train Precedents (Database)", "🔍 2. Review Facility Agreement"])

with tab1:
    st.header("Precedent Database Training")
    st.markdown("Upload past successful or standard facility agreements (PDF or DOCX) to build your firm's precedent vector database.")

    stats = get_db_stats()
    col1, col2 = st.columns(2)
    col1.metric("Total Precedent Documents", stats["total_documents"])
    col2.metric("Total Indexed Clauses/Chunks", stats["total_chunks"])

    if stats["documents"]:
        with st.expander("View Stored Precedent Files"):
            for doc in stats["documents"]:
                st.text(f"• {doc}")

    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload Precedent Agreements", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process & Ingest Precedents"):
        with st.spinner("Parsing and embedding documents into ChromaDB..."):
            total_added = 0
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                chunks_count = process_and_store_document(file_path, uploaded_file.name)
                total_added += chunks_count
            
            st.success(f"Successfully ingested {len(uploaded_files)} files ({total_added} chunks added to vector store)!")
            st.rerun()

with tab2:
    st.header("Draft Facility Agreement Review")
    st.markdown("Upload a new incoming draft agreement to perform comparative gap analysis against your trained precedents.")

    draft_file = st.file_uploader("Upload New Draft Facility Agreement", type=["pdf", "docx"])

    if draft_file:
        # Extract text for analysis
        import fitz  # PyMuPDF
        import docx

        draft_text = ""
        file_bytes = draft_file.read()
        
        if draft_file.name.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                draft_text += page.get_text()
        elif draft_file.name.endswith(".docx"):
            temp_path = os.path.join(UPLOAD_DIR, "temp_draft.docx")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            doc_docx = docx.Document(temp_path)
            draft_text = "\n".join([p.text for p in doc_docx.paragraphs])

        st.info(f"Loaded draft document: **{draft_file.name}** ({len(draft_text)} characters extracted).")

        if st.button("🚀 Run Comprehensive Agreement Review"):
            if not groq_api_key:
                st.error("Please enter your Groq API Key in the sidebar.")
            else:
                with st.spinner("Running RAG precedent retrieval and LLM comparative gap analysis..."):
                    try:
                        report = review_agreement(draft_text, groq_api_key)
                        st.session_state["review_report"] = report
                        st.success("Review complete!")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

    if "review_report" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Legal Review Report & Redline Recommendations")
        st.markdown(st.session_state["review_report"])

        st.download_button(
            label="📥 Download Review Report (Markdown)",
            data=st.session_state["review_report"],
            file_name="facility_agreement_review_report.md",
            mime="text/markdown"
        )
