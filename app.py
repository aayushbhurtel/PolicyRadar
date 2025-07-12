import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# Langchain & DeepSeek
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize DeepSeek Chat via Langchain
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.4,
)

# Prompts for summary and Q&A
summary_prompt = ChatPromptTemplate.from_template("""
You are a helpful civic assistant. Summarize the following congressional bill using this format:

**One sentence summary**: One sentence summary.  
**Key Points**:  
- Bullet 1  
- Bullet 2  
- Bullet 3  

**Summary**: A short paragraph that explains the bill in plain English.

Bill:
{text}
""")

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful policy assistant. Given the following bill text, answer the user's question concisely and accurately.

Bill:
{text}

Question:
{question}

Answer in plain English, and cite sections from the bill when helpful.
""")

summary_chain = summary_prompt | llm | StrOutputParser()
qa_chain = qa_prompt | llm | StrOutputParser()

# --- PDF Extraction ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# --- URL Extraction (requests + BeautifulSoup) ---
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching or parsing URL: {e}"

# --- Streamlit App UI ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: AI-Powered Bill Assistant")

option = st.radio("Choose input method:", ("📄 Upload PDF", "🔗 Enter URL"))
bill_text = None

# Handle PDF input
if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a congressional bill (PDF)", type=["pdf"])
    if st.button("🔍 Summarize PDF"):
        if uploaded_file:
            with st.spinner("Extracting text..."):
                bill_text = extract_text_from_pdf(uploaded_file)
        else:
            st.warning("Please upload a PDF file.")

# Handle URL input
if option == "🔗 Enter URL":
    url = st.text_input("Enter a public bill URL")
    if st.button("🔍 Summarize URL"):
        if url:
            with st.spinner("Fetching and processing page..."):
                bill_text = extract_text_from_url(url)
        else:
            st.warning("Please enter a valid URL.")

# Show Summary
if bill_text:
    with st.spinner("Summarizing with DeepSeek..."):
        summary = summary_chain.invoke({"text": bill_text[:15000]})
    st.markdown("### 🧠 Summary")
    st.markdown(summary)

    # Q&A Section
    st.markdown("### ❓ Ask a Question About This Bill")
    question = st.text_input("Your question:")
    if question and st.button("💬 Ask"):
        with st.spinner("Answering..."):
            answer = qa_chain.invoke({"text": bill_text[:15000], "question": question})
        st.markdown("**Answer:**")
        st.markdown(answer)

# Footer
st.markdown("""
<hr style="margin-top: 3em;">
<div style='text-align: center; font-size: 0.9em;'>
    Created with ❤️ using <strong>Streamlit</strong>, <strong>Langchain</strong>, and <strong>DeepSeek</strong>
</div>
""", unsafe_allow_html=True)
