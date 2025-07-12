import streamlit as st
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
import os
import time

# Langchain & DeepSeek
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Load API key
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Langchain LLM
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.4,
)

# Prompts
summary_prompt = ChatPromptTemplate.from_template("""
You are a helpful civic assistant. Summarize the following congressional bill using this format:

**One Sentence Summary**: One sentence summary.  
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

# PDF extractor
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# URL extractor via Selenium
def extract_text_from_url(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3)  # wait for JS to render

        body = driver.find_element(By.TAG_NAME, "body")
        text = body.text
        driver.quit()

        return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    except Exception as e:
        return f"Error fetching page: {e}"

# --- Streamlit App ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: AI-Powered Bill Assistant")

option = st.radio("Choose input method:", ("📄 Upload PDF", "🔗 Enter URL"))
bill_text = None

# Handle PDF
if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a congressional bill (PDF)", type=["pdf"])
    if st.button("🔍 Summarize PDF"):
        if uploaded_file:
            with st.spinner("Extracting text..."):
                bill_text = extract_text_from_pdf(uploaded_file)
        else:
            st.warning("Please upload a PDF first.")


# Handle URL
if option == "🔗 Enter URL":
    url = st.text_input("Enter a public bill URL")
    if st.button("🔍 Summarize URL"):
        if url:
            with st.spinner("Fetching and rendering..."):
                bill_text = extract_text_from_url(url)
        else:
            st.warning("Please enter a valid URL.")


# Show summary
if bill_text:
    with st.spinner("Summarizing with DeepSeek..."):
        summary = summary_chain.invoke({"text": bill_text[:15000]})
    st.markdown("### 🧠 Summary")
    st.markdown(summary)

    # Q&A section
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
