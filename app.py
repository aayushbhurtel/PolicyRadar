import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
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

# --- Load environment variables ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- LLM Setup (Langchain + DeepSeek) ---
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.4,
)

# --- PDF text extraction ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# --- URL text extraction with Selenium ---
def extract_text_from_url(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(3)  # wait for page to load JS

        body = driver.find_element(By.TAG_NAME, "body")  # OR use .find_element(By.CSS_SELECTOR, "pre")
        text = body.text
        driver.quit()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching or rendering page: {e}"

# --- Prompt Templates ---
summary_prompt = ChatPromptTemplate.from_template("""
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

# --- Streamlit UI ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: AI-Powered Bill Assistant")

option = st.radio("Choose input method:", ("📄 Upload PDF", "🔗 Enter URL"))
bill_text = None
summary_triggered = False

# --- PDF Upload Section ---
if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a congressional bill (PDF)", type=["pdf"])
    pdf_cols = st.columns([1, 1, 2])
    with pdf_cols[2]:
        if st.button("🔍 Summarize PDF"):
            summary_triggered = True
            if uploaded_file:
                with st.spinner("Extracting text from PDF..."):
                    bill_text = extract_text_from_pdf(uploaded_file)
            else:
                st.warning("Please upload a PDF first.")

# --- URL Input Section ---
elif option == "🔗 Enter URL":
    url = st.text_input("Enter a public bill URL (e.g., congress.gov)")
    url_cols = st.columns([1, 1, 2])
    with url_cols[2]:
        if st.button("🔍 Summarize URL"):
            summary_triggered = True
            if url:
                with st.spinner("Fetching and rendering URL..."):
                    bill_text = extract_text_from_url(url)
            else:
                st.warning("Please enter a valid URL.")

# --- Summary Output ---
if summary_triggered and bill_text:
    with st.spinner("Summarizing with DeepSeek..."):
        summary = summary_chain.invoke({"text": bill_text[:15000]})
    st.markdown("### 🧠 Summary")
    st.markdown(summary)

    # --- Q&A Section ---
    st.markdown("### ❓ Ask Questions About This Bill")
    user_question = st.text_input("Enter your question:", placeholder="e.g., Who enforces this?")
    if st.button("💬 Ask"):
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke({"text": bill_text[:15000], "question": user_question})
        st.markdown("**Answer:**")
        st.markdown(answer)

# --- Footer ---
st.markdown("""
<hr style="margin-top: 3em;">
<div style='text-align: center; font-size: 0.9em;'>
    Created with ❤️ using <strong>Streamlit</strong>, <strong>Langchain</strong>, and <strong>DeepSeek</strong>
</div>
""", unsafe_allow_html=True)
