# Facility Agreement Review Agent

A 100% open-source, free-to-run AI agent application built for debt and project finance teams to review new draft facility agreements against past precedent transactions using RAG (Retrieval-Augmented Generation).

## Tech Stack
- **Backend & RAG**: Python, LangChain, ChromaDB
- **Embeddings**: Hugging Face (`sentence-transformers/all-mpnet-base-v2`)
- **LLM**: Llama-3.3-70b via Groq API (Free Tier)
- **Frontend**: Streamlit

---

## Local Setup & Installation

1. **Unzip the downloaded archive** into a directory of your choice.

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain a Free Groq API Key:**
   - Sign up for free at [Groq Console](https://console.groq.com).
   - Copy your API key.

5. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## Free Deployment on Streamlit Community Cloud

1. Push your repository to a public **GitHub** repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New app** and select your GitHub repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**! Your web link will be live globally in under 2 minutes for free.
