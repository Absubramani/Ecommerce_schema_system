import json
import re

ECOMMERCE_KEYWORDS = [
    "ecommerce", "vendor", "product", "order", "customer",
    "cart", "payment", "inventory", "shipment", "catalog",
    "tenant", "marketplace", "supplier", "warehouse",
    "discount", "coupon", "refund", "wishlist", "review",
    "invoice", "brand", "subscription", "affiliate",
    "return", "tax", "segment", "promotion"
]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def is_valid_requirement(text: str) -> bool:
    if not text:
        return False

    t = text.lower().strip()

    if len(t) < 3:
        return False

    return any(k in t for k in ECOMMERCE_KEYWORDS)

def build_schema_prompt(requirement: str) -> str:
    return requirement.strip()

def extract_schema_only(text: str) -> str:
    stop_tokens = ["Explanation", "Note:", "Schema may"]

    for s in stop_tokens:
        if s in text:
            text = text.split(s, 1)[0]

    return text.strip()

def enforce_schema_rules(schema: str) -> str:
    tables = re.findall(r"Table\s+(\w+)\s*\{([^}]*)\}", schema, re.DOTALL)

    fixed_tables = []

    for table_name, body in tables:
        lines = [l.rstrip() for l in body.splitlines() if l.strip()]

        formatted_lines = []
        for line in lines:
            if not line.startswith("  "):
                formatted_lines.append("  " + line.strip())
            else:
                formatted_lines.append(line)

        fixed_tables.append(
            f"Table {table_name} {{\n" +
            "\n".join(formatted_lines) +
            "\n}"
        )

    return "\n\n".join(fixed_tables)
