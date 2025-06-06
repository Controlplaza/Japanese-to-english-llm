from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import torch

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-ja-en"
dataset = load_dataset("iwslt2017", "iwslt2017-ja-en", trust_remote_code=True)

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(batch):
    ja_texts = [item["ja"] for item in batch["translation"]]
    en_texts = [item["en"] for item in batch["translation"]]

    model_inputs = tokenizer(
        ja_texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            en_texts,
            truncation=True,
            padding="max_length",
            max_length=128
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt-ja-en-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs"
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
save_path = "C:/Users/Zeke/ChatBot-llm/models/ja-en-finetuned"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)