import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load DeepSeek API key from .env
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- Streamlit Config ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: Bill Summarizer")

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# --- Web Text Extraction from URL ---
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        return f"Error fetching URL: {e}"

# --- DeepSeek LLM Setup ---
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.4,
)

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful civic assistant. Summarize the following congressional bill using this format:

**TL;DR**: One sentence summary.  
**Key Points**:  
- Bullet 1  
- Bullet 2  
- Bullet 3  

**Summary**: A short paragraph that explains the bill in plain English.

Bill:
{text}
""")

chain = prompt_template | llm | StrOutputParser()

# --- Input Options ---
option = st.radio("Choose input method:", ("📄 Upload PDF", "🔗 Enter URL"))

bill_text = None

if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a congressional bill (PDF)", type=["pdf"])
    if uploaded_file and st.button("🔍 Summarize PDF"):
        with st.spinner("Extracting text from PDF..."):
            bill_text = extract_text_from_pdf(uploaded_file)

elif option == "🔗 Enter URL":
    url = st.text_input("Enter a link to a bill or policy page:")
    if url and st.button("🔍 Summarize URL"):
        with st.spinner("Fetching and extracting text..."):
            bill_text = extract_text_from_url(url)

# --- Summarization ---
if bill_text:
    with st.spinner("Summarizing with DeepSeek..."):
        summary = chain.invoke({"text": bill_text[:15000]})  # avoid overflow

    st.markdown("### 🧠 Summary")
    st.markdown(summary)

# --- Footer ---
st.markdown("""
<hr style="margin-top: 3em;">
<div style='text-align: center; font-size: 0.9em;'>
    Created with ❤️ using <strong>Streamlit</strong>, <strong>Langchain</strong>, and <strong>DeepSeek</strong>
</div>
""", unsafe_allow_html=True)
