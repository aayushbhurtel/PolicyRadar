import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def summarize_text(text):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful civic assistant. Summarize the following policy or bill into:
        - A one-sentence TL;DR
        - Three key bullet points
        - A one-paragraph summary

        Policy:
        {text}
        """
    )
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = prompt | chat
    response = chain.invoke({"text": text[:4000]})  # truncate for token limit
    return response.content

def generate_questions(text, zip_code):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a civic engagement advisor. Based on this policy, generate three personalized questions
        that a citizen from ZIP code {zip_code} might ask their representative.

        Policy:
        {text}
        """
    )
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = prompt | chat
    response = chain.invoke({"text": text[:4000], "zip_code": zip_code})
    return response.content

# --- STREAMLIT UI ---
st.set_page_config(page_title="PolicyRadar", layout="centered")
st.title("📜 PolicyRadar: AI for Civic Engagement")

input_method = st.radio("Choose input method:", ["Upload PDF", "Enter URL"])
text = ""

if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a policy or bill PDF", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
elif input_method == "Enter URL":
    url = st.text_input("Paste the URL of the policy or bill")
    if url:
        try:
            text = extract_text_from_url(url)
        except:
            st.error("Failed to fetch or parse the URL.")

if text:
    st.subheader("Generated TL;DR Summary")
    summary = summarize_text(text)
    st.markdown(summary)

    zip_code = st.text_input("Enter your ZIP code for personalized insights:")
    if zip_code:
        st.subheader("Questions to Ask Your Representative")
        questions = generate_questions(text, zip_code)
        st.markdown(questions)

st.markdown("---")
st.caption("Built with ❤️ using GPT-4, Streamlit, and LangChain")
