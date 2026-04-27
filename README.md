# Langgraph Projects

A collection of professional, production-ready AI workflows and examples built with LangGraph, LangChain, and Groq.

## Features
- **BMI Workflow (`bmi_workflow.py`)**: A basic LangGraph state graph demonstrating simple state management and condition checks without LLMs.
- **Simple LLM Workflow (`simple_llm_workflow.py`)**: A foundational LangGraph pipeline integrating `ChatGroq` for simple Q&A.
- **Prompt Chaining (`prompt_chaining.py`)**: A multi-step workflow that chains prompts to first generate a blog outline, then generate the full blog content.
- **ChromaDB Example (`chromadb_example.py`)**: An example of loading text documents, splitting them, storing them in ChromaDB using `SentenceTransformerEmbeddings`, and running a LangChain QA pipeline.

## Tech Stack
- **LangChain & LangGraph**: Core orchestration framework and state-machine workflows.
- **Groq**: High-performance LLM API integration.
- **ChromaDB**: Local vector database for Retrieval-Augmented Generation (RAG).
- **Python**: Core programming language.

## Folder Structure
```
Langgraph-Projects/
│
├── src/                      # Source code for applications
│   ├── bmi_workflow.py       # Basic state graph example
│   ├── simple_llm_workflow.py# Simple LLM Q&A workflow
│   ├── prompt_chaining.py    # Multi-node prompt chaining workflow
│   └── chromadb_example.py   # RAG pipeline with ChromaDB
│
├── data/                     # Source documents for RAG (if applicable)
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Langgraph-Projects.git
   cd Langgraph-Projects
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

## Usage Examples

### Run the Prompt Chaining Workflow
```bash
python src/prompt_chaining.py
```

### Run the ChromaDB RAG Example
```bash
python src/chromadb_example.py
```

## Future Improvements
- Add Streamlit or FastAPI frontends for the workflows.
- Implement conversational memory (Checkpointer) within the LangGraph workflows.
- Add support for other Vector Databases like Pinecone or Qdrant.
