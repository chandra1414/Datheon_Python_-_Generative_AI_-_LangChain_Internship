# Datheon Python + Generative AI & LangChain Internship

Four hands-on Generative AI projects built during the **Datheon Python + Generative AI & LangChain Internship**, using **Python 3, LangChain, Google Gemini, ChromaDB, Pydantic and Streamlit**.

**Author:** Your Name
**LinkedIn:** _add your LinkedIn profile link_
**Live demo (Task 4):** _add your Streamlit link_

---

## Projects at a glance

| # | Project | What it does | Key tech |
|---|---------|--------------|----------|
| 1 | [JSON Extractor](#task-1-structured-llm-prompt-pipeline--json-extractor) | Turns messy invoices and resumes into strictly validated JSON | Gemini, Pydantic |
| 2 | [RAG Document Q&A](#task-2-document-qa-assistant-with-rag) | Answers questions only from your PDFs, with page citations | LangChain, ChromaDB |
| 3 | [Multi-Tool Agent](#task-3-multi-tool-autonomous-ai-agent) | Agent that picks between web search, calculator and file reader | Gemini function calling |
| 4 | [Streamlit Chatbot](#task-4-interactive-ai-chatbot-with-streamlit) | Streaming web chatbot with memory, saved chats and file upload | Streamlit, LangChain |

## Repository structure
```
Datheon_Python_-_Generative_AI_-_LangChain_Internship/
├── README.md
├── .gitignore
├── .env.example
├── Task1_JSON_Extractor/
│   ├── extractor.py        # main script
│   ├── schemas.py          # Pydantic models
│   ├── prompts.py          # prompt templates + few-shot examples
│   ├── samples/            # messy invoice and resume
│   └── requirements.txt
├── Task2_RAG_Document_QA/
│   ├── rag.py              # load, chunk, embed, retrieve, answer
│   ├── data/sample.pdf
│   └── requirements.txt
├── Task3_Multi_Tool_Agent/
│   ├── agent.py            # agent loop
│   ├── tools.py            # web search, calculator, file reader
│   ├── files/              # files the agent can read
│   └── requirements.txt
└── Task4_Streamlit_Chatbot/
    ├── app.py              # Streamlit app
    ├── .streamlit/config.toml
    └── requirements.txt
```

## Setup (common for all tasks)

**1. Get a free Gemini API key:** https://aistudio.google.com/apikey

**2. Create a `.env` file** in the repository root:
```
GOOGLE_API_KEY=your_key_here
```
Every script looks for `.env` in its own folder and in parent folders, so one file in the root works for all four tasks. Never commit it (it is already in `.gitignore`).

**3. For each task, use its own virtual environment:**
```bash
cd Task1_JSON_Extractor            # or Task2_..., Task3_..., Task4_...
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # Mac/Linux
pip install -r requirements.txt
```

> Gemini model names change often. The default is `gemini-3.5-flash-lite`. If you get a "model not found" error, set `GEMINI_MODEL=<model-name>` in `.env`.

---

## Task 1: Structured LLM Prompt Pipeline & JSON Extractor

Extracts structured data from unstructured text and guarantees the output is valid JSON.

**How it works**
1. `prompts.py` builds a parameterized prompt: task + JSON schema + few-shot example + the new document.
2. `extractor.py` sends it to Gemini.
3. `schemas.py` validates the reply with Pydantic (types, `YYYY-MM-DD` dates, `total = subtotal - discount + tax`, email format).
4. If validation fails, the error is sent back to the model to fix its answer (up to 3 attempts).

**Run**
```bash
cd Task1_JSON_Extractor
python extractor.py samples/invoice.txt --type invoice --save
python extractor.py samples/resume.txt --type resume --save
```

**Example output (invoice, shortened)**
```json
{
  "invoice_number": "INV/2026/0457",
  "invoice_date": "2026-10-14",
  "vendor_name": "Bluewave Electronics Pvt Ltd",
  "currency": "INR",
  "subtotal": 6645.0,
  "discount": 300.0,
  "tax": 1142.1,
  "total": 7487.1
}
```

**Screenshot:** _add terminal screenshot_

---

## Task 2: Document Q&A Assistant with RAG

A Retrieval-Augmented Generation system that answers **only from your documents** and cites the **file and page number**.

**How it works**
1. `PyPDFLoader` loads each PDF page by page.
2. `RecursiveCharacterTextSplitter` makes chunks (1000 characters, 200 overlap), keeping page metadata.
3. Gemini embeddings are stored in a **ChromaDB** vector store (rebuilt automatically when files change).
4. For a question, the 4 most similar chunks are retrieved and Gemini answers using only those.
5. If the answer is not in the documents, it says so instead of guessing.

**Run**
```bash
cd Task2_RAG_Document_QA
python rag.py                                      # interactive
python rag.py -q "How many days of sick leave?"    # one question
```
Put your own PDFs in `data/` and run again.

**Example**
```
You: How many days of sick leave do employees get?
🤖 Employees get 6 days of sick leave per year (sample.pdf, page 2).

You: Who won the 2011 cricket world cup?
🤖 I could not find the answer in the provided documents.
```

**Screenshot:** _add terminal screenshot_

---

## Task 3: Multi-Tool Autonomous AI Agent

A console agent (Gemini function calling) that decides on its own which tool to use and prints its reasoning: **Thought → Action → Observation → Final Answer**.

| Tool | Purpose |
|---|---|
| `web_search` | Real-time search with DuckDuckGo |
| `calculator` | Exact math via a safe parser (no `eval`) |
| `read_file` | Reads txt/csv/json/pdf files, only from the `files/` folder |

**Run**
```bash
cd Task3_Multi_Tool_Agent
python agent.py
python agent.py -q "Read project_budget.txt and add 18% GST to the total"
```

**Example**
```
--- Step 1 ---
🧠 Thought: I need to read the project_budget.txt file to find the project costs.
🔧 Action: read_file({"filename": "project_budget.txt"})
👀 Observation: Hackathon Project Budget ... Venue rental: 45000 ...

--- Step 2 ---
🔧 Action: calculator({"expression": "180050 * 1.18"})
👀 Observation: 212459

✅ Final Answer: Total including 18% GST: 212,459
```
Other questions to try: _"What is the current price of Bitcoin in USD? Multiply it by 0.5."_ and _"Read students.csv and find the average maths marks."_

**Screenshot:** _add terminal screenshot_

---

## Task 4: Interactive AI Chatbot with Streamlit

A web chatbot with a polished dark UI.

**Features**
- Streaming replies (token by token with `st.write_stream`)
- Conversation memory (recent messages sent to Gemini each turn)
- Session state management with `st.session_state`
- Multiple chats: new chat, switch between old chats, delete, saved in `chats.json`
- 📎 Upload images and PDF/DOCX/TXT/CSV files and ask questions about them
- Welcome screen with suggestions, settings (model, creativity, fast mode), error handling

**Run**
```bash
cd Task4_Streamlit_Chatbot
python -m streamlit run app.py
```
Open http://localhost:8501

**Notes:** Scanned PDFs have no text and cannot be read. On Streamlit Cloud, `chats.json` is shared between visitors and may reset, so it is best for demos.

**Screenshots:** _add: welcome screen, streaming chat, PDF Q&A, sidebar with several chats_

---

## Skills demonstrated
- Prompt engineering: templates, few-shot examples, structured output
- Data validation with Pydantic and automatic retry on invalid output
- Retrieval-Augmented Generation: chunking, embeddings, vector search, citations
- Tool-calling agents with a visible reasoning loop
- Streaming chat UI with session state, persistent history and file upload

## Tech stack
Python 3 · Google Gemini · LangChain · ChromaDB · Pydantic · Streamlit · DuckDuckGo Search · PyPDF