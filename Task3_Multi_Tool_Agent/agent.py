"""
Task 3: Multi-Tool Autonomous AI Agent (console)

The agent (Gemini + function calling) decides on its own which tool to use:
    - web_search : DuckDuckGo real-time search
    - calculator : exact math
    - read_file  : reads files from the 'files' folder

It runs a loop and prints the reasoning chain:
    Thought -> Action -> Observation -> ... -> Final Answer

Usage:
    python agent.py
    python agent.py -q "Read project_budget.txt and add 18% GST to the total"
"""
import argparse
import json
import os
import sys
import textwrap
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import tools as impl

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()  # reads GOOGLE_API_KEY from .env (this folder or any parent folder)

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_STEPS = 8


# ---------------- Tools (LangChain wrappers) ----------------
@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo for current information: news, prices, facts, people, statistics. Input: a short search query."""
    return impl.search_web(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly. Supports + - * / ** % //, parentheses, sqrt, log, sin, cos, tan, abs, round, pi, e. Example: '(45000 + 28500) * 1.18'. Always use this for arithmetic."""
    return impl.calculate(expression)


@tool
def read_file(filename: str) -> str:
    """Read a local file (txt, csv, json, md, pdf) from the 'files' folder. Input: the file name, for example 'project_budget.txt'."""
    return impl.read_local_file(filename)


TOOLS = [web_search, calculator, read_file]
TOOL_MAP = {t.name: t for t in TOOLS}


def build_system_prompt():
    files = ", ".join(impl.list_local_files()) or "none"
    return (
        "You are an autonomous AI research assistant that solves multi-step problems using tools.\n"
        "Tools: web_search (current information), calculator (ALL arithmetic), "
        "read_file (local files).\n"
        f"Local files available: {files}\n\n"
        "Rules:\n"
        "- Before EVERY tool call, write one short sentence starting with 'Thought:' that explains why.\n"
        "- Never do arithmetic in your head. Always use the calculator tool.\n"
        "- Use web_search for anything current (prices, news, recent facts). Do not guess them.\n"
        "- If a tool returns an error or poor results, try a different query or approach.\n"
        "- When you have everything needed, give a clear final answer with the key numbers and sources.\n"
        "- Write the final answer in plain text only: no LaTeX, no ** bold, no tables. Short lines starting with '-' are fine."
    )


# ---------------- Helpers ----------------
def response_text(resp):
    """Plain text from a Gemini reply (handles string or list formats)."""
    text = getattr(resp, "text", None)
    if not isinstance(text, str) and callable(text):  # older LangChain: .text() is a method
        text = text()
    if isinstance(text, str) and text:
        return str(text)
    content = resp.content
    if isinstance(content, str):
        return content
    out = ""
    for part in content or []:
        if isinstance(part, str):
            out += part
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            out += part["text"]
    return out


def short(text, width=450):
    return textwrap.shorten(" ".join(str(text).split()), width=width, placeholder=" ...")


def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_API_KEY not found. Add it to your .env file.")
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, google_api_key=api_key, timeout=60, max_retries=1
    )
    return llm.bind_tools(TOOLS)


# ---------------- The agent loop ----------------
def run_agent(question, llm):
    messages = [SystemMessage(content=build_system_prompt()), HumanMessage(content=question)]

    for step in range(1, MAX_STEPS + 1):
        ai = llm.invoke(messages)
        messages.append(ai)  # keep the original reply (needed for tool-call history)
        text = response_text(ai).strip()

        if not ai.tool_calls:  # no tool needed -> this is the final answer
            print(f"\n✅ Final Answer:\n{text or '(the model returned an empty answer)'}\n")
            return text

        print(f"\n--- Step {step} ---")
        if text:
            print(f"🧠 {text if text.lower().startswith('thought') else 'Thought: ' + text}")

        for call in ai.tool_calls:
            name, args = call["name"], call["args"]
            print(f"🔧 Action: {name}({json.dumps(args, ensure_ascii=False)})")
            if name not in TOOL_MAP:
                result = f"Unknown tool '{name}'. Available: {', '.join(TOOL_MAP)}"
            else:
                try:
                    result = str(TOOL_MAP[name].invoke(args))
                except Exception as e:
                    result = f"Tool error: {e}"
            print(f"👀 Observation: {short(result)}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=name))

    print("\n⚠️ Stopped: reached the maximum number of steps without a final answer.\n")
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Multi-tool autonomous AI agent")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    args = parser.parse_args()

    llm = get_llm()

    if args.question:
        run_agent(args.question, llm)
        return

    print("Multi-Tool AI Agent (web search + calculator + file reader). Type 'exit' to quit.")
    print(f"Local files: {', '.join(impl.list_local_files()) or 'none'}\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            try:
                run_agent(q, llm)
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()