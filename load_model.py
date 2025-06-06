from transformers import MarianTokenizer, MarianMTModel
import torch

# Path to your fine-tuned model
model_dir = "C:/Users/Zeke/ChatBot-llm/models/ja-en-finetuned"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text: str, source_lang: str) -> str:
    """
    Translate between English and Japanese.
    source_lang should be "ja" or "en"
    """
    if source_lang == "ja":
        # Translate Japanese → English
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        translated_tokens = model.generate(**inputs)
        output = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return f"[EN] {output}"

    elif source_lang == "en":
        # Translate English → Japanese
        # Add forced language tokens for Japanese
        tokenizer.src_lang = "en"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        translated_tokens = model.generate(**inputs)
        output = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return f"[JA] {output}"

    else:
        return "Invalid source language. Use 'ja' or 'en'."

# Chat loop
print("Japanese ↔ English Translator")
print("Type 'exit' to quit.\n")

while True:
    src = input("Enter text: ")
    if src.lower() == "exit":
        break

    # Detect simple language type
    lang = "ja" if any("\u3040" <= ch <= "\u30ff" for ch in src) else "en"
    translation = translate(src, lang)
    print(translation)
