# RAG02H: From Zero to Hero in Retrieval-Augmented Generation

<!-- ![image](https://github.com/user-attachments/assets/70d6c2c4-8da1-466b-a40c-70c827d320e7) -->

<img src="https://github.com/user-attachments/assets/70d6c2c4-8da1-466b-a40c-70c827d320e7" alt="image" width="500" height="400">

**RAG02H** is a hands-on Python project focused on implementing Retrieval-Augmented Generation (RAG) using open-source tools. This repository is designed for data scientists and banking professionals to build, improve, and extend RAG pipelines with a practical focus on document retrieval and answering questions using vector databases and large language models (LLMs).

## Project Overview

This repository contains two main pipelines:

### 1. Document Parsing and Storage Pipeline
This pipeline handles the ingestion and processing of documents, transforming them into embeddings, which are then stored in a vector database for efficient retrieval.

- **Document Ingestion**: The system reads documents from local files (e.g., PDF, DOCX, HTML) or a web URL.
- **Document Parsing**: The content of these documents is parsed and divided into chunks for better embedding generation.
- **Embedding Generation**: Each chunk of text is converted into a vector embedding using an open-source model like `sentence-transformers`.
- **Vector Database Storage**: The embeddings are stored in a vector database (e.g., FAISS, Milvus, or Pinecone) for fast similarity search.

### 2. Retrieval-Augmented Generation (RAG) Pipeline
The RAG pipeline handles question answering by embedding the user's query, retrieving relevant document chunks, and using an LLM to generate a final answer.

- **User Query**: The user submits a question to the system.
- **Query Embedding**: The question is transformed into a vector embedding using the same model as used for documents.
- **Nearest Neighbors Search**: The vector database retrieves the top relevant document chunks based on cosine similarity to the query embedding.
- **LLM Generation**: The question and the retrieved chunks are fed into an open-source LLM (e.g., `GPT-3` or `GPT-J`) to generate a comprehensive, context-aware answer.

### High-Level Pipeline Flow
1. **Document Parsing Pipeline**:
    - Read document → Parse content → Chunk text → Generate embeddings → Save to vector database.
    
2. **RAG Pipeline**:
    - User query → Query embedding → Retrieve relevant document chunks → Query + chunks fed to LLM → Generate final answer.

## Configuration and Setup

To configure the repository for RAG02H, follow these steps:

### Prerequisites
- Python 3.9+ (e.g., 3.9.7)
- Recommended package manager: `pip` or `conda`
- Access to an LLM API (for generation, e.g., OpenAI, HuggingFace Transformers)

### Required Libraries
Install the necessary Python libraries:

```bash
pip install sentence-transformers faiss-cpu milvus pymilvus openai transformers
```

# Vector Database Configuration

### Using FAISS:
- No additional setup is required for FAISS. Simply ensure that the `faiss-cpu` package is installed as part of the project dependencies.

### Using Milvus:
- Install Milvus by following their official documentation and set up either a local or cloud instance.
- Once Milvus is up and running, configure the `pymilvus` connection in the repository by updating the connection settings with your Milvus instance details.

# LLM API Configuration

To generate answers using a Large Language Model (LLM), you'll need to configure your OpenAI or HuggingFace API keys.

### OpenAI API:
1. Set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```
2. Alternatively, you can store the API key in a `.env` file for easy loading.

### HuggingFace API:
1. Install and configure the HuggingFace Transformers library.
2. You can either use a locally available model or make API calls to the HuggingFace model repository for inference.

# Example Repository Structure

```bash
RAG02H/
│
├── data/
│   ├── docs/             # Folder containing documents to parse (PDF, HTML, etc.)
├── embeddings/
│   ├── faiss_index/      # FAISS index storage
├── src/
│   ├── ingestion/        # Scripts for document parsing and embedding generation
│   ├── retrieval/        # Query embedding and nearest neighbor search
│   ├── llm/              # LLM integration for answer generation
│   ├── app.py            # Main application pipeline
├── requirements.txt      # Required Python libraries
├── README.md             # Project description and setup instructions
```

# Running the Project

### Step 1: Parse Documents
To load and parse documents, run the following command:

```bash
python src/ingestion/document_parser.py
```

This will generate embeddings from the documents and store them in the configured vector database.

### Step 2 [optional]: Launch the Streamlit App
Start the Streamlit app to interact with the RAG pipeline:

```bash
streamlit run src/app.py
```

### Step 3: Ask a Question
Once the documents are indexed, you can initiate the Retrieval-Augmented Generation (RAG) pipeline by asking a question:

```bash
python src/app.py --question "What is the financial outlook for 2024?"
```

Otherwise, if you run streamlit, once the app is running, you can enter a question in the Streamlit interface. The system will retrieve relevant document chunks and use the LLM to generate an answer based on your query.
