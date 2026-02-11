import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("cpu")
print("Device:", device)

dataset = load_dataset("json", data_files="data/train.json")["train"]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def preprocess(example):
    model_inputs = tokenizer(
        example["input"].strip(),
        truncation=True,
        max_length=MAX_SOURCE_LENGTH
    )

    labels = tokenizer(
        text_target=example["output"].strip(),
        truncation=True,
        max_length=MAX_TARGET_LENGTH
    )

    model_inputs["labels"] = [
        t if t != tokenizer.pad_token_id else -100
        for t in labels["input_ids"]
    ]

    return model_inputs

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=1
)

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    report_to="none",
    no_cuda=True,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")
