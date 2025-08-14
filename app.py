import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import csv
import spacy
from PyPDF2 import PdfReader
from docx import Document

# --- Utility Function ---
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load spaCy model
 --- Load spaCy Model Safely ---
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Page Config ---
st.set_page_config(page_title="Resume Critique App", page_icon="🧠", layout="wide")

# --- Custom CSS for Instagram-style UI ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #333;
}
h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #222;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    margin-bottom: 0.5em;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #eee;
    padding: 1rem;
}
section[data-testid="stSidebar"] .stTextInput, 
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stButton {
    border-radius: 8px;
    margin-bottom: 1rem;
}
.stSubheader, .stTextArea, .stExpander, .stProgress {
    background-color: #fff;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}
button[kind="primary"] {
    background: linear-gradient(to right, #ff6a00, #ee0979);
    color: white;
    border-radius: 25px;
    padding: 0.5rem 1.5rem;
    font-weight: bold;
    transition: all 0.3s ease;
}
button[kind="primary"]:hover {
    transform: scale(1.05);
}
[data-testid="stToast"] {
    background-color: #fff0f5;
    border-left: 5px solid #ee0979;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Title Banner ---
st.markdown("<h1 style='text-align: center;'>🧠 Resume Critique + JD Matcher + GPT-2 Generator</h1>", unsafe_allow_html=True)
st.markdown("Upload your resume and job description to extract keywords, calculate match score, generate feedback, and create a tailored cover letter.")

# --- Display Current Date & Time ---
current_time = datetime.now().strftime("%A, %d %B %Y | %I:%M %p")
st.markdown(f"<div style='text-align:center; font-size:1.1rem; color:#555;'>🕒 {current_time}</div>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("Resume (PDF/DOCX)", type=["pdf", "docx"])
jd_file = st.sidebar.file_uploader("Job Description (PDF/DOCX)", type=["pdf", "docx"])
st.sidebar.markdown("---")
prompt = st.sidebar.text_area("📝 GPT-2 Prompt")
generate_btn = st.sidebar.button("🚀 Generate Text")
tone = st.sidebar.selectbox("🎯 Cover Letter Tone", ["Formal", "Enthusiastic", "Confident", "Persuasive"])
generate_cover_btn = st.sidebar.button("✉️ Generate Cover Letter")

# --- Text Extraction ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# --- Keyword Extraction ---
def extract_keywords(text):
    doc = nlp(text)
    return set([token.text.lower() for token in doc if token.is_alpha and not token.is_stop])

# --- Match Score Calculation ---
def calculate_match(resume_keywords, jd_keywords):
    matched = resume_keywords.intersection(jd_keywords)
    score = round(len(matched) / len(jd_keywords) * 100, 2) if jd_keywords else 0
    missing = jd_keywords - resume_keywords
    return matched, missing, score

# --- Resume + JD Processing ---
if resume_file:
    resume_text = extract_text(resume_file)
    resume_keywords = extract_keywords(resume_text)

    st.markdown("### 📌 Resume Keywords", unsafe_allow_html=True)
    st.write(", ".join(sorted(resume_keywords)))

    with open("resume_keywords.csv", "a", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([get_timestamp(), resume_file.name, ", ".join(sorted(resume_keywords))])
    st.toast(f"✅ Resume keywords saved at {get_timestamp()}", icon="💾")

if resume_file and jd_file:
    jd_text = extract_text(jd_file)
    jd_keywords = extract_keywords(jd_text)
    matched_keywords, missing_keywords, match_score = calculate_match(resume_keywords, jd_keywords)

    st.markdown("### 📊 JD Match Score", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.2rem;'>{match_score}% keyword overlap</div>", unsafe_allow_html=True)
    st.progress(int(match_score))

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 Matched Keywords"):
            st.write(", ".join(sorted(matched_keywords)))
    with col2:
        with st.expander("❌ Missing Keywords from Resume"):
            st.write(", ".join(sorted(missing_keywords)))

    with open("match_results.csv", "a", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            get_timestamp(),
            resume_file.name,
            jd_file.name,
            match_score,
            ", ".join(sorted(matched_keywords)),
            ", ".join(sorted(missing_keywords))
        ])
    st.toast(f"📁 Match results saved at {get_timestamp()}", icon="📊")

# --- GPT-2 Text Generator ---
if generate_btn and prompt:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### 🧠 GPT-2 Response", unsafe_allow_html=True)
        st.code(response, language="markdown")

        with open("responses.txt", "a", encoding="utf-8") as txt_file:
            txt_file.write(f"\n[{get_timestamp()}]\nPrompt: {prompt}\nResponse: {response}\n")

        with open("responses.csv", "a", newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([get_timestamp(), prompt, response])

        st.toast(f"✅ GPT-2 response saved at {get_timestamp()}", icon="💾")

    except ValueError as e:
        st.error(f"⚠️ Generation Error: {str(e)}")
        st.markdown("Try reducing input length or increasing `max_new_tokens`.")

# --- Cover Letter Generator ---
if generate_cover_btn and resume_file and jd_file:
    prompt_text = f"""
    Write a {tone.lower()} cover letter for the following job description and resume.

    Job Description:
    {jd_text}

    Resume:
    {resume_text}

    The cover letter should highlight relevant skills, express interest in the role, and be tailored to the job.
    """

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        cover_letter = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### ✉️ Generated Cover Letter", unsafe_allow_html=True)
        st.text_area("Preview", cover_letter, height=300)

        with open("cover_letters.txt", "a", encoding="utf-8") as txt_file:
            txt_file.write(f"\n[{get_timestamp()}]\nTone: {tone}\n{cover_letter}\n")

        with open("cover_letters.csv", "a", newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([get_timestamp(), tone, cover_letter])

        st.toast(f"✅ Cover letter saved at {get_timestamp()}", icon="📨")

    except ValueError as e:
        st.error(f"⚠️ Generation Error: {str(e)}")
        st.markdown("Try reducing input length or increasing `max_new_tokens`.")

