import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser

dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# In-memory Qdrant for demo;  for production - replace with persistent
client = QdrantClient(":memory:")
COLLECTION_NAME = "rag_app_collection"

# Ensure collection exists
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

def add_pdf_to_vectorstore(file_path):

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(pages)
    if splits:
        vector_store.add_documents(splits)
        return True
    return False

def add_url_to_vectorstore(url):

    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if splits:
        vector_store.add_documents(splits)
        return True
    return False

def add_research_paper_to_vectorstore(file_path):

    loader = GenericLoader.from_filesystem(
        file_path,
        glob="*",
        suffixes=[".pdf"],
        parser=GrobidParser(segment_sentences=False)
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if splits:
        vector_store.add_documents(splits)
        return True
    return False

def ask_question(question, k=3):

    results = vector_store.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in results)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content 

