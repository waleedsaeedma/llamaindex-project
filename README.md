# Darwin RAG Agent

A multilingual RAG agent over Charles Darwin's *On the Origin of Species*, built with LlamaIndex while following the Hugging Face Agents . Ask in English, Dutch or Arabic: the question is translated to English, searched in the book, and the answer is translated back.

## How it works
- step2_ingest.py: loads the PDF, splits it into nodes, embeds them with BAAI/bge-small-en-v1.5 and stores them in Chroma.
- rag_setup.py: reopens the Chroma index and builds the query engine.
- step7_workflow.py: LlamaIndex Workflow with three steps (translate question, search book, translate answer).
- step6_multilingual.py: alternative version, an AgentWorkflow agent with chat memory.
- old_steps/: earlier learning steps.

## Setup
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Create a .env file with your OpenAI key:

    OPENAI_API_KEY=your-key-here

Put a PDF of the book in data/ (for example data/Origin_of_Species.pdf), then run:

    python step2_ingest.py
    python step7_workflow.py
