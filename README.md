# RAG-personalised-documents
Powered by Artificial Intelligence, this project generates relevant points indexed on supporting documents in a vector database, retrieves the most relevant personalised points for each user query related to the task at hand.

Documents are split by chunking on paragraphs.


## What it does

- Loads personal source documents from `data/`
- Splits them into retrieval-friendly chunks with metadata
- Builds an in-memory vector store for semantic search
- Retrieves relevant candidate evidence for a document
- Uses an LLM to generate a letter from:
  - prompt description
  - retrieved information
  - internet research
- Fact-checks company-specific claims in the generated letter and writes a validation
  report alongside the other run artifacts
- Archives each run in a unique timestamped directory under `logs/`, including the
  retrieved context, generated letter, validation report, and `run.log`

## Chunking strategy
Documents are split on paragraph boundaries first, then large chunks are further split into smaller overlapping pieces.
Metadata is attached to each chunk to help retrieval prioritize:
- technical skills
- soft skills
- leadership/achievement type

## Requirements

- Python 3.13+
- OpenAI API key for embeddings
- Optional: Bedrock credentials if you switch embedding provider
- Tavily API key if you use web search tooling
- Environment variables set including:
OPENAI_API_KEY=
OPENAI_MODEL=
OPENAI_BASE_URL=
BEDROCK_REGION=
BEDROCK_PROFILE=
BEDROCK_MODEL_ID=
TAVILY_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=

## How to use

1. Input your reference files into the data directory
2. Adjust the human and system prompts in the prompts directory
3. Run main.py
4. Find the AI output and validation report in the newest timestamped folder under `logs/`
