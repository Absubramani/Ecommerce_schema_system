import re

def validate_schema(schema: str) -> dict:
    errors = []

    tables = re.findall(r"Table\s+(\w+)\s*\{([^}]*)\}", schema, re.DOTALL)

    if not tables:
        return {"is_valid": False, "errors": ["No DBML tables found"]}

    for table_name, body in tables:
        lines = [l.strip() for l in body.splitlines() if l.strip()]

        pk_count = sum(1 for l in lines if "[pk]" in l)

        if pk_count != 1:
            errors.append(f"Table '{table_name}' must have exactly one primary key")

        for line in lines:
            if line.startswith("id ") and "char(26)" not in line:
                errors.append(f"Table '{table_name}' id must be char(26)")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
