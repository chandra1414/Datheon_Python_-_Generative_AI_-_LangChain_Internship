"""
Task 1: Structured LLM Prompt Pipeline & JSON Extractor

Usage:
    python extractor.py samples/invoice.txt --type invoice
    python extractor.py samples/resume.txt  --type resume --save
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from prompts import SYSTEM_PROMPT, build_prompt
from schemas import SCHEMAS

load_dotenv()  # reads GOOGLE_API_KEY from .env (this folder or any parent folder)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_RETRIES = 3


def get_llm(model):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_API_KEY not found. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key, timeout=60, max_retries=1
    )


def response_text(resp):
    """Get plain text from a Gemini response (handles string or list formats)."""
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


def clean_json(raw):
    """Remove ```json fences and any text around the JSON object."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in the model reply")
    return raw[start : end + 1]


def extract_structured(text, doc_type, llm, max_retries=MAX_RETRIES, verbose=True):
    """Prompt the LLM, validate the JSON with Pydantic, and retry with the error if invalid."""
    schema_cls = SCHEMAS[doc_type]
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", build_prompt(doc_type, text, schema_cls)),
    ]

    last_error = None
    for attempt in range(1, max_retries + 1):
        raw = response_text(llm.invoke(messages))
        try:
            data = json.loads(clean_json(raw))
            return schema_cls.model_validate(data)  # strict validation
        except (ValueError, ValidationError) as e:
            last_error = e
            if verbose:
                print(f"[attempt {attempt}/{max_retries}] invalid output, retrying...\n  {e}\n")
            # Feed the error back so the model can fix its own answer
            messages.append(("ai", raw))
            messages.append(
                ("human", f"That JSON was invalid: {e}\nReturn the corrected JSON object only.")
            )

    raise RuntimeError(f"Could not get valid JSON after {max_retries} attempts: {last_error}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract validated JSON from unstructured text")
    parser.add_argument("file", help="Path to a .txt file")
    parser.add_argument("--type", choices=list(SCHEMAS), default="invoice", help="Document type")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--save", action="store_true", help="Save result to output/<name>.json")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")

    print(f"Model: {args.model} | Type: {args.type} | File: {path.name}\n")
    result = extract_structured(text, args.type, get_llm(args.model))

    output = result.model_dump_json(indent=2)
    print("✅ Valid JSON:\n")
    print(output)

    if args.save:
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{path.stem}.json"
        out_file.write_text(output, encoding="utf-8")
        print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()