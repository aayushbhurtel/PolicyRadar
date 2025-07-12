import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API Key from .env
load_dotenv()
#DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]

# --- Streamlit Config ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: Bill Summarizer")

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# --- Langchain LLM Configuration using DeepSeek ---
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.4,
)

# --- Prompt Template ---
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

# --- Create Langchain Summarization Chain ---
chain = prompt_template | llm | StrOutputParser()

# --- UI File Upload ---
uploaded_file = st.file_uploader("Upload a congressional bill (PDF)", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        bill_text = extract_text_from_pdf(uploaded_file)
        truncated_text = bill_text[:15000]  # prevent token overflow

    with st.spinner("Summarizing with DeepSeek..."):
        summary = chain.invoke({"text": truncated_text})

    st.markdown("### 🧠 Summary")
    st.markdown(summary)
