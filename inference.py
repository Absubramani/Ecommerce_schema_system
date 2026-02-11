import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from schema_validator import validate_schema
from utils import extract_schema_only, enforce_schema_rules, is_valid_requirement
from config import OUTPUT_DIR


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)

model.to(device)
model.eval()


def normalize_table_names(requirement: str) -> str:
    """
    Convert table names inside:
    'Create ecommerce schema for tables ...'
    into snake_case automatically.
    """

    pattern = r"(Create\s+ecommerce\s+schema\s+for\s+tables\s+)(.+)"
    match = re.search(pattern, requirement, re.IGNORECASE)

    if not match:
        return requirement

    prefix = match.group(1)
    tables_part = match.group(2)

    tables = [t.strip() for t in tables_part.split(",")]

    normalized_tables = []
    for table in tables:
        table = table.lower()
        table = re.sub(r"\s+", "_", table)
        normalized_tables.append(table)

    return prefix + ", ".join(normalized_tables)


def generate_schema(requirement: str):

    if not is_valid_requirement(requirement):
        return "", {"is_valid": False, "errors": ["Invalid requirement"]}

    requirement = normalize_table_names(requirement)

    inputs = tokenizer(
        requirement.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    schema = extract_schema_only(decoded)
    schema = enforce_schema_rules(schema)

    validation = validate_schema(schema)

    return schema, validation
