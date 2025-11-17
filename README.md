# Doc-RAG: Document Retrieval-Augmented Generation System

Hey there! This is a RAG (Retrieval-Augmented Generation) system I built using LangGraph, FastAPI, and Google's Gemini AI. It helps you ask questions about your documents and get intelligent answers.

## What Can It Do?

### Load and Search Your Documents
- Drop your documents in a folder and the system will process them
- Automatically chunks and embeds text for searching
- Uses FAISS to quickly find relevant information
- Everything is saved so you don't have to reload documents every time

### Three Ways to Ask Questions

**1. RAG Bot (The Smart One)**
This uses a 3-step process to answer your questions:
- First, it finds relevant documents
- Then, it generates an answer using those documents
- Finally, it checks if the answer makes sense (and retries if it doesn't)
- Remembers your conversation history

**2. Direct RAG Search (The Quick One)**
- Fast search and summarization
- Gets you answers quickly without all the validation steps

**3. Vector Search (The Raw One)**
- Just gives you the relevant document chunks
- No AI processing, just pure search results
- Fastest and cheapest option

### REST API
I've wrapped everything in a FastAPI server with 4 endpoints. Easy to use from any application.

## What You'll Need

- Python 3.8 or newer
- UV package manager (it's awesome, trust me)
- A Google API key for Gemini

## Getting Started

### Step 1: Get the Code
```bash
cd /home/boy/Desktop/program/python/lang/Doc-RAG/a
```

### Step 2: Install UV
If you don't have it yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Set Up Your Environment
```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
```

### Step 4: Install Everything You Need
```bash
uv pip install fastapi uvicorn python-dotenv
uv pip install langchain langchain-google-genai langgraph
uv pip install faiss-cpu sentence-transformers
uv pip install pydantic
```

### Step 5: Add Your API Key
Create a file called `.env` and add your Google API key:
```bash
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

You can get a key here: https://makersuite.google.com/app/apikey

### Step 6: Add Your Documents
Put your PDF, TXT, or MD files in a folder somewhere, like `./docs/`

## Running the Server

Just run this:
```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Your server will be running at `http://localhost:8000`

Want to see the interactive docs? Go to `http://localhost:8000/docs`

## How to Use the API

### Loading Your Documents
Tell the system where your documents are:

```bash
curl -X POST http://localhost:8000/data_loader \
  -H "Content-Type: application/json" \
  -d '{"path": "./docs"}'
```

### Asking Questions (Smart Mode)
Use the RAG Bot for validated answers:

```bash
curl -X POST http://localhost:8000/rag_bot \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SQL injection?"}'
```

### Asking Questions (Quick Mode)
Use Direct RAG for faster responses:

```bash
curl -X POST http://localhost:8000/Direct_RAG_Question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is OSCP certification?"}'
```

### Getting Raw Results
Use Vector Search to see the actual document chunks:

```bash
curl -X POST http://localhost:8000/Vector_Search \
  -H "Content-Type: application/json" \
  -d '{"question": "Kerberos authentication"}'
```

## How It Works

Think of it like this:

```
You ask a question
    ↓
RAG Bot does its magic:
  1. Searches your documents
  2. Generates an answer
  3. Checks if it's good (if not, tries again)
    ↓
You get your answer
```

## Testing

Want to see if everything works? Run these:

```bash
uv run test_agent.py              # Tests all the parts
uv run test_5_questions.py        # Tries 5 different questions
uv run agent.py                   # Runs the agent directly
```

## Project Layout

```
a/
├── src/                    # The core functionality
├── agent.py               # The smart 3-node agent
├── main.py                # The API server
├── test_agent.py          # Tests for everything
├── test_5_questions.py    # Sample questions
├── .env                   # Your secrets (don't share this!)
├── faiss_store/           # Where the vector database lives
└── README.md              # You're reading it!
```

## Common Issues

**"Can't find my API key!"**
Make sure you created the `.env` file with your `GOOGLE_API_KEY`

**"FAISS index not found"**
You need to load documents first using the `/data_loader` endpoint

**"Rate limit errors"**
The free Gemini tier allows 10 requests per minute. Just wait a bit or upgrade your plan.

**"Import errors"**
Try reinstalling: `uv pip install -r requirements.txt`

## Performance Notes

Just so you know what to expect:
- Vector Search: Super fast (10-50ms)
- Direct RAG: Pretty quick (1-3 seconds)
- RAG Bot: Takes a bit longer (3-8 seconds) because it validates everything
- Loading documents: Depends on how many you have

## Tweaking Things

### Want to use a different model?
Edit `agent.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Change this
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
```

### Want more search results?
Change `top_k` in your searches:
```python
summary = rag_search.search_and_summarize(question, top_k=5)
```

### Want different embeddings?
Edit `src/vectorstore.py`:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## What You Can Build With This

I've seen people use similar systems for:
- Answering questions about big document collections
- Searching company knowledge bases
- Analyzing research papers
- Building technical support bots
- Creating study tools
- Looking up security documentation

## Security Stuff

Important notes:
- Keep your API key in `.env` and never commit it to git
- Validate user inputs if you're making this public
- Add rate limiting if lots of people will use it
- Use HTTPS in production

## Ideas for Later

Some things I'm thinking about adding:
- Support for Word and PowerPoint files
- Multiple languages
- Streaming answers as they're generated
- User login system
- A web interface for conversations
- Docker setup for easy deployment
- Custom prompt templates
- Support for other vector databases

## License

MIT License - do whatever you want with it!

## Want to Contribute?

Pull requests are welcome! Just:
1. Fork it
2. Make your changes
3. Send a pull request

## Need Help?

Open an issue on GitHub and I'll try to help out.

---

Built with LangGraph, FastAPI, and Google Gemini