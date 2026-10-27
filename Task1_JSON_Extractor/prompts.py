"""Parameterized prompt templates + few-shot examples."""
import json

SYSTEM_PROMPT = (
    "You are a precise data-extraction engine. "
    "Return ONLY one valid JSON object that matches the given schema. "
    "No markdown, no code fences, no explanations. "
    "Use null for missing optional fields and never invent information. "
    "Dates must be YYYY-MM-DD. Numbers must be plain numbers (no currency symbols or commas)."
)

# Few-shot examples: (input text, expected JSON output)
EXAMPLES = {
    "invoice": [
        (
            "INVOICE #A-101 | Date: 5th March 2026\n"
            "From: Sharma Stationers, Pune\n"
            "Bill to: Ravi Traders\n"
            "2 x Notebook @ 50 = 100\n"
            "1 x Pen box @ 120 = 120\n"
            "Subtotal 220, GST 18% = 39.6, Total Rs. 259.6",
            {
                "invoice_number": "A-101",
                "invoice_date": "2026-03-05",
                "due_date": None,
                "vendor_name": "Sharma Stationers",
                "customer_name": "Ravi Traders",
                "currency": "INR",
                "items": [
                    {"description": "Notebook", "quantity": 2, "unit_price": 50, "amount": 100},
                    {"description": "Pen box", "quantity": 1, "unit_price": 120, "amount": 120},
                ],
                "subtotal": 220,
                "discount": 0,
                "tax": 39.6,
                "total": 259.6,
            },
        )
    ],
    "resume": [
        (
            "Anita Rao | anita.rao@mail.com | +91 98765 43210\n"
            "B.Tech CSE, JNTU Hyderabad, 2024\n"
            "Skills: Python, SQL, Pandas\n"
            "Intern - DataCorp (Jan 2024 - Jun 2024): built sales dashboards",
            {
                "full_name": "Anita Rao",
                "email": "anita.rao@mail.com",
                "phone": "+91 98765 43210",
                "summary": None,
                "skills": ["Python", "SQL", "Pandas"],
                "education": [
                    {"degree": "B.Tech CSE", "institution": "JNTU Hyderabad", "year": "2024"}
                ],
                "experience": [
                    {
                        "role": "Intern",
                        "company": "DataCorp",
                        "duration": "Jan 2024 - Jun 2024",
                        "highlights": ["Built sales dashboards"],
                    }
                ],
            },
        )
    ],
}

TASK_DESCRIPTION = {
    "invoice": "Extract the invoice details from the document.",
    "resume": "Extract the candidate details from the resume.",
}


def build_prompt(doc_type, text, schema_cls):
    """Fill the template with: task, JSON schema, few-shot examples and the new document."""
    schema = json.dumps(schema_cls.model_json_schema(), indent=2)

    examples = ""
    for i, (inp, out) in enumerate(EXAMPLES[doc_type], 1):
        examples += (
            f"--- Example {i} ---\n"
            f"DOCUMENT:\n{inp}\n"
            f"JSON:\n{json.dumps(out, indent=2)}\n\n"
        )

    return (
        f"{TASK_DESCRIPTION[doc_type]}\n\n"
        f"JSON SCHEMA:\n{schema}\n\n"
        f"{examples}"
        f"--- Now your turn ---\n"
        f"DOCUMENT:\n{text}\n"
        f"JSON:"
    )