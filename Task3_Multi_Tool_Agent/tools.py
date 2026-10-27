"""The three tools the agent can use: web search, math calculator, local file reader."""
import ast
import math
import operator as op
from pathlib import Path

FILES_DIR = Path(__file__).parent / "files"
MAX_FILE_CHARS = 8000


# ---------------- Tool 1: Web search (DuckDuckGo) ----------------
def search_web(query, max_results=5):
    try:
        try:
            from ddgs import DDGS  # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # old package name

        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}. Try a different query."

    if not results:
        return "No results found. Try different keywords."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('body', '')}\n   URL: {r.get('href', '')}")
    return "\n".join(lines)


# ---------------- Tool 2: Safe math calculator ----------------
_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}
_FUNCTIONS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial,
    "min": min, "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculate(expression):
    """Evaluate a math expression safely (no eval/exec, so no code can run)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not calculate '{expression}' ({e}). Use plain numbers and + - * / ** ()."


# ---------------- Tool 3: Local file reader ----------------
def list_local_files():
    if not FILES_DIR.exists():
        return []
    return sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())


def read_local_file(filename):
    """Read a file from the 'files' folder only (other locations are blocked)."""
    root = FILES_DIR.resolve()
    path = (FILES_DIR / filename).resolve()
    if root not in path.parents:
        return "Access denied: only files inside the 'files' folder can be read."
    if not path.is_file():
        return f"File not found: {filename}. Available files: {', '.join(list_local_files()) or 'none'}"

    try:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n".join(
                f"[Page {i}]\n{(p.extract_text() or '')}" for i, p in enumerate(reader.pages, 1)
            )
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"Could not read {filename}: {e}"

    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n...[file truncated]"
    return text or "(file is empty)"