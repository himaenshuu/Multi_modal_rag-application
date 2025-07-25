import streamlit as st
from rag_app import ask_question, add_pdf_to_vectorstore, add_url_to_vectorstore, add_research_paper_to_vectorstore
from audio_video_rag import audio_rag_pipeline, video_rag_pipeline

st.markdown("""
    <h1 style='text-align: center;'>Multimodal RAG based Q&A Platform</h1>
""", unsafe_allow_html=True)

# Dropdown 
input_type = st.selectbox("Select file type:", ["PDF", "URL", "Research Paper", "Audio File", "Video File"])

if input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        if add_pdf_to_vectorstore(f"temp_{uploaded_file.name}"):
            st.success("PDF indexed successfully!")
        else:
            st.error("Failed to index PDF.")
elif input_type == "URL":
    url = st.text_input("Enter a URL:")
    if st.button("Add URL") and url:
        if add_url_to_vectorstore(url):
            st.success("Web page indexed successfully!")
        else:
            st.error("Failed to process web page.")
elif input_type == "Research Paper":
    research_file = st.file_uploader("Upload a research paper PDF (Grobid)", type=["pdf"])
    if research_file is not None:
        with open(f"temp_{research_file.name}", "wb") as f:
            f.write(research_file.getbuffer())
        if add_research_paper_to_vectorstore(f"temp_{research_file.name}"):
            st.success("Research paper indexed successfully with Grobid!")
        else:
            st.error("Failed to index research paper.")
elif input_type == "Audio File":
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if audio_file is not None:
        with open(f"temp_{audio_file.name}", "wb") as f:
            f.write(audio_file.getbuffer())
        query = st.text_input("Ask a question about the audio:")
        if st.button("Process Audio") and query:
            result = audio_rag_pipeline(f"temp_{audio_file.name}", query)
            st.write("**Answer:**", result["gemini_answer"])
elif input_type == "Video File":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file is not None:
        with open(f"temp_{video_file.name}", "wb") as f:
            f.write(video_file.getbuffer())
        query = st.text_input("Ask a question about the video:")
        if st.button("Process Video") and query:
            result = video_rag_pipeline(f"temp_{video_file.name}", query)
            st.write("**Answer:**", result["gemini_answer"])

question = st.text_input("Ask a question:")
if st.button("Submit") and question:
    answer = ask_question(question)
    st.write("**Answer:**", answer) 


# 🧬 Signature: HimanshuWassHere [hash: HR2025X]
    
