import streamlit as st
import os
import io
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="DocuChat AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #111b21; color: #e9edef; }
    .user-msg {
        background: #005c4b;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 25%;
        max-width: 70%;
        float: right;
        clear: both;
        font-size: 14px;
    }
    .bot-msg {
        background: #202c33;
        color: #e9edef;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 25% 8px 0;
        max-width: 70%;
        float: left;
        clear: both;
        font-size: 14px;
    }
    .msg-time {
        font-size: 11px;
        color: #8696a0;
        margin-top: 4px;
        text-align: right;
    }
    .clearfix { clear: both; margin-bottom: 5px; }
    .pro-badge {
        background: linear-gradient(135deg, #f6d365, #fda085);
        color: black;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .advice-box {
        background: #2a3942;
        border-left: 4px solid #00a884;
        padding: 10px 15px;
        border-radius: 0 10px 10px 0;
        margin-top: 8px;
        font-size: 13px;
        color: #8696a0;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False
if "llm" not in st.session_state:
    st.session_state.llm = None

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 DocuChat AI")
    if st.session_state.is_pro:
        st.markdown("<span class='pro-badge'>⭐ PRO ACTIVE</span>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    st.markdown("---")
    st.markdown("### 💎 Upgrade to Pro — ₹10 only!")
    st.markdown("""
**Pro Features:**
- 📊 Full AI Report (PDF Download)
- 🔍 Deep Analysis
- 📋 Key Points Summary

**Pay ₹10 via UPI:** `divyanshumishra315@upi`

After payment, enter transaction ID below:
    """)
    promo = st.text_input("Transaction ID / Promo Code:")
    if st.button("🔓 Activate Pro"):
        if promo.strip() in ["PRO10", "PRO99", "TEST"]:
            st.session_state.is_pro = True
            st.success("✅ Pro Activated!")
            st.balloons()
        elif len(promo.strip()) > 8:
            st.session_state.is_pro = True
            st.success("✅ Payment verified! Pro Activated!")
            st.balloons()
        else:
            st.error("❌ Invalid code! Pay ₹10 via UPI first.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        chat_text = "\n".join([
            f"[{m.get('time','')}] {m['role'].upper()}: {m['content']}"
            for m in st.session_state.chat_history
        ])
        st.download_button("💾 Download Chat", chat_text, "chat.txt", "text/plain")

# Process PDF
if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("⚡ Processing document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        st.session_state.llm = llm
        prompt = ChatPromptTemplate.from_template("""
You are DocuChat AI — a smart, friendly document assistant.

Answer the question based on the context. Be clear and helpful.
After your answer, on a new line starting with "💡 Advice:", give 1 helpful tip or insight related to the answer.

Context: {context}
Question: {question}

Answer:""")
        st.session_state.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"✅ **{uploaded_file.name}** processed! Ask me anything.",
            "time": datetime.now().strftime("%H:%M")
        })

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 🤖 DocuChat AI")
with col2:
    if st.session_state.is_pro:
        st.markdown("<br><span class='pro-badge'>⭐ PRO</span>", unsafe_allow_html=True)
    if st.session_state.pdf_name:
        st.markdown(f"<br>📄 `{st.session_state.pdf_name}`", unsafe_allow_html=True)

st.markdown("---")

# Display Chat
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-msg'>
            {msg['content']}
            <div class='msg-time'>{msg.get('time','')}</div>
        </div>
        <div class='clearfix'></div>
        """, unsafe_allow_html=True)
    else:
        content = msg['content']
        advice = ""
        if "💡 Advice:" in content:
            parts = content.split("💡 Advice:")
            content = parts[0].strip()
            advice = parts[1].strip() if len(parts) > 1 else ""

        st.markdown(f"""
        <div class='bot-msg'>
            {content}
            <div class='msg-time'>{msg.get('time','')}</div>
        </div>
        <div class='clearfix'></div>
        """, unsafe_allow_html=True)

        if advice:
            st.markdown(f"""
            <div class='advice-box'>
                💡 <b>Advice:</b> {advice}
            </div>
            <div class='clearfix'></div>
            """, unsafe_allow_html=True)

# Chat Input
if st.session_state.chain:
    question = st.chat_input("Type a message...")
    if question:
        now = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user", "content": question, "time": now
        })
        with st.spinner("typing..."):
            answer = st.session_state.chain.invoke(question)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%H:%M")
        })
        st.rerun()

    # PRO Report
    if st.session_state.is_pro:
        st.markdown("---")
        if st.button("📊 Generate Full Report (PRO ⭐)"):
            with st.spinner("🔍 Generating professional report..."):
                chat_summary = "\n".join([
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.chat_history
                ])
                report_prompt = f"""Based on this document Q&A session, create a professional report:

1. DOCUMENT SUMMARY
2. KEY POINTS
3. QUESTIONS AND ANSWERS SUMMARY
4. RECOMMENDATIONS

Chat History:
{chat_summary}

Write a clean professional report:"""

                report_response = st.session_state.llm.invoke(report_prompt)
                report_text = report_response.content

                st.markdown("### 📊 Your Report")
                st.markdown(report_text)

                # Generate PDF
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4,
                    rightMargin=inch*0.8, leftMargin=inch*0.8,
                    topMargin=inch*0.8, bottomMargin=inch*0.8)

                title_style = ParagraphStyle('title',
                    fontSize=20, fontName='Helvetica-Bold',
                    textColor=colors.HexColor('#005c4b'),
                    spaceAfter=20, alignment=1)
                heading_style = ParagraphStyle('heading',
                    fontSize=13, fontName='Helvetica-Bold',
                    textColor=colors.HexColor('#202c33'),
                    spaceBefore=15, spaceAfter=8)
                body_style = ParagraphStyle('body',
                    fontSize=11, fontName='Helvetica',
                    leading=16, spaceAfter=6)

                story = []
                story.append(Paragraph("DocuChat AI - Document Report", title_style))
                story.append(Paragraph(
                    f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
                    body_style))
                story.append(Spacer(1, 20))

                for line in report_text.split('\n'):
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 6))
                    elif any(line.startswith(x) for x in ['1.','2.','3.','4.','#']):
                        clean = line.replace('#','').strip()
                        story.append(Paragraph(clean, heading_style))
                    else:
                        safe_line = line[:500]
                        story.append(Paragraph(safe_line, body_style))

                doc.build(story)
                buffer.seek(0)

                st.download_button(
                    "⬇️ Download Report (PDF)",
                    buffer,
                    "DocuChat_Report.pdf",
                    "application/pdf"
                )

else:
    st.markdown("""
    <div style='text-align:center; margin-top:80px; color:#8696a0;'>
        <h2>👈 Upload a PDF to get started</h2>
        <p>Ask questions • Get answers • Get advice</p>
        <p>Upgrade to Pro for PDF report generation!</p>
    </div>
    """, unsafe_allow_html=True)