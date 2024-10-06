# RAG02H: From Zero to Hero in Retrieval-Augmented Generation

<img src="https://github.com/user-attachments/assets/70d6c2c4-8da1-466b-a40c-70c827d320e7" alt="image" width="500" height="400">

**RAG02H** is a hands-on Python project focused on implementing Retrieval-Augmented Generation (RAG) using open-source tools. This repository is designed for data scientists to build, improve, and extend RAG pipelines with a practical focus on document retrieval and answering questions using vector databases and large language models (LLMs).

## Project Overview

This repository contains two main pipelines:

### 1. Document Parsing and Storage Pipeline
This pipeline handles the ingestion and processing of documents, transforming them into embeddings, which are then stored in a vector database for efficient retrieval.

- **Document Ingestion**: The system reads documents from local files (e.g., PDF, DOCX, HTML) or a web URL.
- **Document Parsing**: The content of these documents is parsed and divided into chunks for better embedding generation.
- **Embedding Generation**: Each chunk of text is converted into a vector embedding using an open-source model like `sentence-transformers`.
- **Vector Database Storage**: The embeddings are stored in a vector database (e.g., Qdrant) for fast similarity search.

### 2. Retrieval-Augmented Generation (RAG) Pipeline
The RAG pipeline handles question answering by embedding the user's query, retrieving relevant document chunks, and using an LLM to generate a final answer.

- **User Query**: The user submits a question to the system.
- **Query Embedding**: The question is transformed into a vector embedding using the same model as used for documents.
- **Nearest Neighbors Search**: The vector database retrieves the top relevant document chunks based on cosine similarity to the query embedding.
- **LLM Generation**: The question and the retrieved chunks are fed into an open-source LLM (e.g., `GPT-4` or `LLAMA3.1`) to generate a comprehensive, context-aware answer.

### High-Level Pipeline Flow
1. **Document Parsing Pipeline**:
    - Read document → Parse content → Chunk text → Generate embeddings → Save to vector database.
    
2. **RAG Pipeline**:
    - User query → Query embedding → Retrieve relevant document chunks → Query + chunks fed to LLM → Generate final answer.

## Configuration and Setup

To configure the repository for RAG02H, follow these steps:

### Prerequisites
- Python 3.9+ (e.g., 3.9.7 https://www.python.org/downloads/release/python-397/)
- Recommended package manager: `pip` or `conda`
- Access to an LLM API (for generation, e.g., AwanLLM, OpenAI, HuggingFace Transformers)

### Required Libraries
Install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

# Vector Database Configuration

### Using Qdrant:
- Install Qdrant as specified in the requirements.txt

### Using FAISS [not implemented end-to-end]:
- No additional setup is required for FAISS. Simply ensure that the `faiss-cpu` package is installed as part of the project dependencies.

# LLM API Configuration

To generate answers using a Large Language Model (LLM), you'll need to configure your OpenAI or HuggingFace API keys.

### AWANLLM API:
1. Set your AWAN API (https://www.awanllm.com/) key as an environment variable:
    ```bash
    export AWAN_API_KEY="your-api-key"
    ```
2. Alternatively, you can store the API key in a `.env` file for easy loading.

# Example Repository Structure

```bash
RAG02H/
│
├── data/
│   ├── docs/             # Folder containing documents to parse (PDF, HTML, etc.)
├── embeddings/
│   ├── vdb/              # Qdrant vector DB
├── src/
|   ├── config/           # YAML file where paramenters and filepath are specified
│   ├── ingestion/        # Scripts for document parsing and embedding generation
│   ├── retrieval/        # Query embedding and nearest neighbor search
│   ├── llm/              # LLM integration for answer generation
│   ├── ui/               # Main application pipeline where it is launch the UI application
├── requirements.txt      # Required Python libraries
├── README.md             # Project description and setup instructions
```

<img src="https://github.com/user-attachments/assets/c9d1c57d-59d7-4246-b377-853f3b81e33b" alt="image" width="700" height="400">

# Running the Project

### Step 1: Parse Documents
To load, parse and save the documents in the vector db, run (or debug) the following:

```bash
src/ingestion/indexing_qd.py
```

This will generate embeddings from the documents and store them in the configured vector database.
Remember to set the environment variable from the file .env (e.g., in PyCharm you have to edit the configuration of the script indexing_qd.py).


### Step 2: Ask a Question
Once the documents are indexed, you can initiate the Retrieval-Augmented Generation (RAG) pipeline by asking a question, run or debug:

```bash
src/llm/api_call.py
```

Remember to set the environment variable from the file .env (e.g., in PyCharm you have to edit the configuration of the script api_call.py).
Warning: the first time SentenceTransformer: all-MiniLM-L6-v2 takes some minutes to be downloaded!

### Step 3 [optional]: Launch the Streamlit App
Start the Streamlit app to interact with the RAG pipeline (exactly the two steps before):

```bash
streamlit run .\src\ui\app_ui.py
```

Asked a question, the system will retrieve relevant document chunks and use the LLM to generate an answer based on your query.
Again, set in the terminal the .env variables before starting the comand line, as well as you want to run it in debug mode.
