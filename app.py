import streamlit as st
import os
import hashlib
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.embeddings.base import Embeddings
import numpy as np

llm = OllamaLLM(base_url="http://localhost:11434", model="gpt-oss:20b")

if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

if not os.path.exists("faiss_db"):
    os.makedirs("faiss_db")

METADATA_FILE = "faiss_db/file_metadata.json"

def get_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_file_metadata():
    """Load metadata of processed files"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_file_metadata(metadata):
    """Save metadata of processed files"""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def get_new_or_modified_files():
    """Get list of files that are new or have been modified"""
    metadata = load_file_metadata()
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    new_or_modified = []
    
    for pdf_file in pdf_files:
        filepath = os.path.join("pdfs", pdf_file)
        current_hash = get_file_hash(filepath)
        
        if pdf_file not in metadata or metadata[pdf_file]["hash"] != current_hash:
            new_or_modified.append(pdf_file)
    
    return new_or_modified

class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or list of strings")

if "embeddings_model" not in st.session_state:
    model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    st.session_state.embeddings_model = CustomEmbeddings(model)

if "reranker_model" not in st.session_state:
    st.session_state.reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "loader" not in st.session_state:
    st.session_state.loader = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None

if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

def vector_embedding(uploaded_file=None):
    if uploaded_file:
        with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    new_or_modified_files = get_new_or_modified_files()
    
    if not new_or_modified_files:
        st.sidebar.info("All files already processed. Loading existing embeddings...")
        load_existing_vectors()
        return
    
    existing_vectors = None
    if os.path.exists("faiss_db/index.faiss"):
        try:
            existing_vectors = FAISS.load_local(
                "faiss_db", 
                st.session_state.embeddings_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Could not load existing vectors: {e}")
    
    st.sidebar.info(f"Processing {len(new_or_modified_files)} new/modified files...")
    
    all_docs = []
    metadata = load_file_metadata()
    
    for pdf_file in new_or_modified_files:
        filepath = os.path.join("pdfs", pdf_file)
        temp_dir = f"temp_{pdf_file.replace('.', '_')}"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        import shutil
        shutil.copy(filepath, os.path.join(temp_dir, pdf_file))
        
        loader = PyPDFDirectoryLoader(temp_dir)
        docs = loader.load()
        all_docs.extend(docs)
        
        metadata[pdf_file] = {
            "hash": get_file_hash(filepath),
            "processed_at": str(st.session_state.get('current_time', 'unknown'))
        }
        
        shutil.rmtree(temp_dir)
    
    if all_docs:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        new_documents = st.session_state.text_splitter.split_documents(all_docs)
        st.session_state.final_documents = new_documents
        
        new_vectors = FAISS.from_documents(
            new_documents,
            st.session_state.embeddings_model
        )
        
        if existing_vectors:
            existing_vectors.merge_from(new_vectors)
            st.session_state.vectors = existing_vectors
        else:
            st.session_state.vectors = new_vectors
        
        st.session_state.vectors.save_local("faiss_db")
        save_file_metadata(metadata)
        
        st.sidebar.success(f"Processed {len(new_or_modified_files)} files and updated embeddings!")
    else:
        load_existing_vectors()

def load_existing_vectors():
    """Load existing FAISS database if it exists"""
    if os.path.exists("faiss_db/index.faiss"):
        try:
            st.session_state.vectors = FAISS.load_local(
                "faiss_db", 
                st.session_state.embeddings_model,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            st.error(f"Error loading existing vectors: {e}")
            return False
    return False

def rerank_documents(query, documents, top_k=5):
    """Rerank documents using cross-encoder and return top_k"""
    if not documents:
        return documents
    
    pairs = [(query, doc.page_content) for doc in documents]
    
    scores = st.session_state.reranker_model.predict(pairs)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]

st.title("RAG")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Upload & Embed"):
    vector_embedding(uploaded_file)
    st.sidebar.success("Document uploaded and embedded!")

st.sidebar.header("Indexed Files")
metadata = load_file_metadata()
if metadata:
    st.sidebar.write(f"**{len(metadata)} files indexed:**")
    for filename, file_info in metadata.items():
        with st.sidebar.expander(f"📄 {filename}"):
            st.write(f"**Hash:** {file_info['hash'][:8]}...")
            st.write(f"**Processed:** {file_info.get('processed_at', 'Unknown')}")
            
            if os.path.exists(os.path.join("pdfs", filename)):
                st.write("File exists")
            else:
                st.write("File missing")
else:
    st.sidebar.write("*No files indexed yet*")

prompt1 = st.text_input("Ask a question from the document")

if st.button("Ask"):
    if prompt1:
        if "vectors" not in st.session_state or not st.session_state.vectors:
            if not load_existing_vectors():
                vector_embedding()

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide a concise and direct response. Keep your answer brief and to the point.
            Avoid unnecessary elaboration or repetition.
            IMPORTANT: Answer in the same language as the question.
            <context>
            {context}
            </context>
            Questions: {input}
            
            Answer (be concise and in the same language as the question):
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
        
        initial_docs = retriever.invoke(prompt1)
        
        reranked_docs = rerank_documents(prompt1, initial_docs, top_k=5)
        
        response = {
            'answer': document_chain.invoke({
                'context': reranked_docs,
                'input': prompt1
            }),
            'context': reranked_docs
        }
        
        answer = response['answer']
        source_documents = response.get('context', [])

        st.write("### Answer")
        st.write(answer)

        if source_documents:
            st.write("### Source References")
            for i, doc in enumerate(source_documents):
                with st.expander(f"Source {i+1} - Page {doc.metadata.get('page', 'Unknown')}"):
                    st.write("**Content:**")
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.write("**Metadata:**")
                        for key, value in doc.metadata.items():
                            st.write(f"- {key}: {value}")
