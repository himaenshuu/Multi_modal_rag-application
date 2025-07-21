import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from faster_whisper import WhisperModel
import subprocess

llm = ChatGoogleGenerativeAI(model=os.getenv("model_name"), google_api_key=os.getenv("GEMINI_API"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = QdrantClient(":memory:")
COLLECTION_NAME = "audio_video_rag_collection"
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": len(embeddings.embed_query("test")), "distance": "Cosine"}
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

def transcribe_audio_local(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    text = " ".join(segment.text for segment in segments)
    return text.strip()

def extract_audio_from_video(video_path: str, output_audio_path: str = "audio_from_video.wav"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    command = [
        "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", output_audio_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_audio_path

def audio_rag_pipeline(audio_path: str, query: str):
    transcript = transcribe_audio_local(audio_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    vector_store.add_texts(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    semantic_chunks = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(semantic_chunks)
    prompt = f"""
You are a helpful assistant. Based on the following context from an audio transcription, answer the user's question.

Context:
{context}

Question:
{query}
"""
    gemini_response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"semantic_chunks": semantic_chunks, "gemini_answer": gemini_response}

def video_rag_pipeline(video_path: str, query: str):
    audio_path = extract_audio_from_video(video_path)
    return audio_rag_pipeline(audio_path, query) 
