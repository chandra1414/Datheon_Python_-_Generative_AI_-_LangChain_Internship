# Task 3: Multi-Tool Autonomous AI Agent

A console AI agent (Google Gemini + LangChain function calling) that **decides on its own which tool to use** and shows its reasoning chain: **Thought → Action → Observation → Final Answer**.

## Tools
| Tool | What it does |
|---|---|
| `web_search` | Real-time web search with DuckDuckGo |
| `calculator` | Exact math using a safe expression parser (no `eval`, so no code can run) |
| `read_file` | Reads txt/csv/json/md/pdf files, only from the `files/` folder |

## How it works
1. The question and a system prompt go to Gemini, which has the 3 tools bound to it.
2. If Gemini wants a tool, it replies with a tool call. The script prints the **Thought** and **Action**, runs the tool, and prints the **Observation**.
3. The observation is sent back to Gemini. This repeats (max 8 steps) until Gemini has enough information.
4. Gemini gives the **Final Answer**.

## Folder structure
```
Task3_Multi_Tool_Agent/
├── agent.py            # agent loop + LangChain tool wrappers
├── tools.py            # web search, calculator, file reader
├── files/              # files the agent can read
│   ├── project_budget.txt
│   └── students.csv
├── requirements.txt
└── .env.example
```

## Setup
```bash
pip install -r requirements.txt
```
Add your key to a `.env` file (this folder or any parent folder):
```
GOOGLE_API_KEY=your_key_here
```

## Run
```bash
python agent.py                                  # interactive
python agent.py -q "your question here"          # one question
```

## Example questions
1. `Read project_budget.txt, add all the costs, and tell me the total including 18% GST.`
   → uses `read_file` then `calculator` (expected total: 212459)
2. `What is the current price of Bitcoin in USD? Multiply it by 0.5.`
   → uses `web_search` then `calculator`
3. `Read students.csv and find the average maths marks.`
   → uses `read_file` then `calculator`
4. `What is the current population of India? What is 2% of it?`
   → uses `web_search` then `calculator`

## Example output
```
--- Step 1 ---
🧠 Thought: I need to read the budget file first.
🔧 Action: read_file({"filename": "project_budget.txt"})
👀 Observation: Hackathon Project Budget ... Venue rental: 45000 ...

--- Step 2 ---
🧠 Thought: Now I will add the costs and apply 18% GST.
🔧 Action: calculator({"expression": "(45000+28500+60000+12750+33800)*1.18"})
👀 Observation: 212459

✅ Final Answer:
The total including 18% GST is INR 212,459.
```
_Replace with your real output and add screenshots below._

## Screenshots 📸 Project Output

![Project Output](screenshots/output1.png)
![Project Output](screenshots/output2.png)
![Project Output](screenshots/output3.png)
![Project Output](screenshots/output4.png)

## Tech stack
Python 3, LangChain, Google Gemini (function calling), DuckDuckGo Search