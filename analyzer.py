import os
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_embeddings
from config import CHROMA_PERSIST_DIR, LLM_MODEL_NAME

def review_agreement(new_agreement_text: str, groq_api_key: str) -> str:
    """
    Compares a new draft agreement against ChromaDB precedents using RAG 
    and generates a clause-by-clause review report.
    """
    if not groq_api_key:
        raise ValueError("Groq API Key is required. Please set it in the sidebar.")

    # 1. Initialize Vector Store & Retriever
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Retrieve relevant precedent clauses
    relevant_precedents = retriever.get_relevant_documents(new_agreement_text[:3000])
    precedent_context = "\n\n---\n\n".join([doc.page_content for doc in relevant_precedents])

    if not precedent_context:
        precedent_context = "No prior precedents found in database. Rely on standard international debt and project finance norms (LMA standards)."

    # 3. Setup LLM via Groq
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=LLM_MODEL_NAME,
        temperature=0.1
    )

    # 4. Construct Expert Legal Prompt
    system_prompt = (
        "You are an expert Debt Finance and Project Finance Legal Counsel. "
        "Your task is to conduct a rigorous clause-by-clause comparative review of an incoming draft "
        "Facility Agreement against past precedent transactions and standard market practice (LMA standards).\n\n"
        "Precedent Context Retrieved from Database:\n"
        "-------------------------------------\n"
        f"{precedent_context}\n"
        "-------------------------------------\n"
    )

    human_prompt = (
        "Please review the following draft Facility Agreement text/excerpt. "
        "Provide a structured analysis covering:\n"
        "1. Executive Summary & Risk Scorecard (High/Medium/Low risk rating).\n"
        "2. Clause-by-Clause / Section-by-Section Breakdown containing:\n"
        "   - Clause Name & Estimated Location\n"
        "   - Draft Summary\n"
        "   - Precedent Comparison (How past deals handled this)\n"
        "   - Risk Level (High / Medium / Low)\n"
        "   - Actionable Recommendation & Specific Redline / Propose Amendment Wording.\n\n"
        "Draft Agreement Text:\n"
        "-------------------------------------\n"
        f"{new_agreement_text[:12000]}\n"
        "-------------------------------------\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | llm
    response = chain.invoke({})
    
    return response.content
