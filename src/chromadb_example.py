import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def process_documents(data_dir: str, persist_dir: str):
    print("Loading documents...")
    loader = DirectoryLoader(data_dir, glob='**/*.txt', loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Generating embeddings and saving to Chroma...")
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return db, embeddings

def query_chroma_db(persist_dir: str, embeddings):
    print("Querying Chroma DB...")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    query = "what are the different kinds of pets people commonly own?"
    matching_docs = db.similarity_search_with_score(query, k=2)
    print("Top matches:")
    for doc, score in matching_docs:
        print(f"Score: {score:.4f} | Content: {doc.page_content}")

def run_qa_chain():
    print("\nRunning QA Chain...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)
    prompt = PromptTemplate.from_template("Answer the Question: {question}")
    chain = prompt | llm
    
    response = chain.invoke({"question": "what are the emotional benefits of owning a pet?"})
    print("\n--- Response ---")
    print(response.content)

if __name__ == "__main__":
    data_directory = './data'
    persist_directory = './chroma_db'
    
    if os.path.exists(data_directory):
        db, embeddings = process_documents(data_directory, persist_directory)
        query_chroma_db(persist_directory, embeddings)
    else:
        print(f"Data directory '{data_directory}' not found. Skipping document processing.")
        
    run_qa_chain()
