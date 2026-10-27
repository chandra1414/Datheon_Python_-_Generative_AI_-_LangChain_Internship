"""
Task 2: Document Q&A Assistant with RAG (Retrieval-Augmented Generation)

Pipeline:  load PDF/TXT -> split into chunks -> embed (Gemini) -> store in ChromaDB
           -> retrieve top chunks for a question -> Gemini answers ONLY from them
           -> print answer with source file + page citations.

Usage:
    python rag.py                              # interactive Q&A on files in data/
    python rag.py -q "How many sick leaves?"   # ask one question
    python rag.py --rebuild                    # force re-indexing
    python rag.py --data my_pdfs               # use another folder
"""
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)  # hide langchain-community notice

load_dotenv()  # reads GOOGLE_API_KEY from .env (this folder or any parent folder)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
BATCH_SIZE = 40  # chunks embedded per API call (keeps us under rate limits)

NOT_FOUND = "I could not find the answer in the provided documents."

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer ONLY using the CONTEXT provided. Do not use outside knowledge. "
    f"If the context does not contain the answer, reply exactly: {NOT_FOUND} "
    "Keep answers clear and concise. After each fact, cite the source like (file.pdf, page 2)."
)


# ---------------- Setup helpers ----------------
def get_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("GOOGLE_API_KEY not found. Add it to your .env file.")
    return key


def response_text(resp):
    """Plain text from a Gemini reply (handles string or list formats)."""
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


def get_embeddings(api_key):
    """Create the embedding model. Tries the name with and without the 'models/' prefix."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    base = EMBED_MODEL.replace("models/", "")
    last_error = None
    for name in (base, f"models/{base}"):
        try:
            emb = GoogleGenerativeAIEmbeddings(model=name, google_api_key=api_key)
            emb.embed_query("test")  # quick check that the model really works
            return emb
        except Exception as e:
            last_error = e
    sys.exit(
        f"Could not use embedding model '{EMBED_MODEL}': {last_error}\n"
        "Set EMBED_MODEL in your .env to a model available for your key."
    )


def get_llm(api_key):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL, google_api_key=api_key, timeout=60, max_retries=1
    )


# ---------------- Step 1: load documents ----------------
def load_documents(data_dir):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    files = sorted(p for p in data_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        sys.exit(f"No PDF/TXT files found in {data_dir}. Put your documents there.")

    docs = []
    for f in files:
        try:
            if f.suffix.lower() == ".pdf":
                loaded = PyPDFLoader(str(f)).load()  # one Document per page
            else:
                loaded = TextLoader(str(f), encoding="utf-8").load()
        except Exception as e:
            print(f"  ! Could not read {f.name}: {e}")
            continue
        loaded = [d for d in loaded if d.page_content.strip()]
        for d in loaded:
            d.metadata["source_name"] = f.name
        print(f"  loaded {f.name}: {len(loaded)} page(s)/section(s)")
        docs.extend(loaded)

    if not docs:
        sys.exit("No readable text found (scanned PDFs need OCR).")
    return docs


# ---------------- Step 2: split into chunks ----------------
def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)  # page metadata is kept on every chunk


# ---------------- Step 3: embed + store in ChromaDB ----------------
def make_vectorstore(embeddings):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )


def fingerprint(data_dir):
    """Describes the current files, so we know when to re-index."""
    return [
        [p.name, p.stat().st_size, int(p.stat().st_mtime)]
        for p in sorted(data_dir.glob("*"))
        if p.suffix.lower() in {".pdf", ".txt", ".md"}
    ]


def ensure_index(data_dir, embeddings, rebuild=False):
    fp_file = DB_DIR / "fingerprint.json"
    current = fingerprint(data_dir)

    if DB_DIR.exists() and not rebuild:
        try:
            if json.loads(fp_file.read_text()) == current:
                print("Using existing index (documents unchanged).")
                return make_vectorstore(embeddings)
        except Exception:
            pass
        print("Documents changed, re-indexing...")

    shutil.rmtree(DB_DIR, ignore_errors=True)
    print("Loading documents...")
    chunks = split_documents(load_documents(data_dir))
    print(f"Created {len(chunks)} chunks. Embedding and storing in ChromaDB...")

    store = make_vectorstore(embeddings)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            store.add_documents(batch)
        except Exception as e:
            print(f"  rate limit / error ({e}); waiting 30s and retrying once...")
            time.sleep(30)
            store.add_documents(batch)
        print(f"  indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    DB_DIR.mkdir(exist_ok=True)
    fp_file.write_text(json.dumps(current))
    print("Index ready.\n")
    return store


# ---------------- Step 4: retrieve + answer ----------------
def citation(doc):
    name = doc.metadata.get("source_name") or Path(doc.metadata.get("source", "document")).name
    if "page" in doc.metadata:
        return f"{name}, page {int(doc.metadata['page']) + 1}"  # PyPDF pages start at 0
    return name


def ask(question, store, llm):
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(f"[{citation(d)}]\n{d.page_content}" for d in docs)

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    answer = response_text(llm.invoke([("system", SYSTEM_PROMPT), ("human", user_prompt)])).strip()

    print(f"\n🤖 {answer}")
    if NOT_FOUND not in answer:
        # Show only the pages the answer actually cites (fallback: all retrieved chunks)
        used = [d for d in docs if citation(d) in answer] or docs
        print("\n📚 Sources:")
        seen = set()
        for d in used:
            c = citation(d)
            if c not in seen:
                seen.add(c)
                snippet = " ".join(d.page_content.split())[:90]
                print(f"  • {c}: \"{snippet}...\"")
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAG Document Q&A")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument("--rebuild", action="store_true", help="Force re-indexing")
    parser.add_argument("--data", default=str(DATA_DIR), help="Folder with PDF/TXT files")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        sys.exit(f"Folder not found: {data_dir}")

    api_key = get_api_key()
    embeddings = get_embeddings(api_key)
    store = ensure_index(data_dir, embeddings, args.rebuild)
    llm = get_llm(api_key)

    if args.question:
        ask(args.question, store, llm)
        return

    print("Ask questions about your documents. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            try:
                ask(q, store, llm)
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()