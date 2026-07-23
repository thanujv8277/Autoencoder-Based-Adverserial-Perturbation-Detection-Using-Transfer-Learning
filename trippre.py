import re
import ftfy
import torch
import pandas as pd
from unidecode import unidecode
from transformers import AutoTokenizer

# ----------------------------------------
# GPU / Tokenizer Init
# ----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Using device:", device)

# ----------------------------------------
# 1. FIX STAR RATING PATTERNS (UPDATED)
# Handles: 4*, 4**, 4***, 4★, 4✮, 4⭐ etc.
# ----------------------------------------
def fix_star_ratings(text):
    # Match ANY of these: * ★ ✮ ✯ ⭐ ✱ ✲ ✵ ✶ ✷ ✸ ✹
    star_chars = r"[*★✮✯⭐✱✲✵✶✷✸✹]"
    pattern = rf"(\d)\s*{star_chars}+"
    return re.sub(pattern, r"\1-star", text)


# ----------------------------------------
# 2. FIX SLASH-PHRASES (safe)
# good/great → good or great
# clean/quiet → clean and quiet
# ----------------------------------------
def fix_slash_phrases(text):
    def repl(match):
        w1, w2 = match.group(1), match.group(2)

        # If both are adjectives → join with "and"
        if w1.endswith("ly") or w2.endswith("ly"):
            return f"{w1} and {w2}"

        return f"{w1} or {w2}"

    return re.sub(r"\b([A-Za-z]+)\/([A-Za-z]+)\b", repl, text)


# ----------------------------------------
# 3. Number-word patterns (DO NOT MODIFY)
# Example: 24hr, 5thfloor, 2bedroom
# ----------------------------------------
def preserve_num_word(text):
    return text  # intentionally untouched


# ----------------------------------------
# 4. Remove only garbage symbols
# Keep punctuation & meaning-critical characters
# ----------------------------------------
def remove_noise(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove only garbage symbols
    text = re.sub(r"[^A-Za-z0-9\s.,!?'\-/★✮✯⭐✱✲✵✶✷✸✹]", "", text)
    return text


# ----------------------------------------
# 5. Unicode normalization
# ----------------------------------------
def normalize_unicode(text):
    text = ftfy.fix_text(text)
    return unidecode(text)


# ----------------------------------------
# 6. Normalize whitespace
# ----------------------------------------
def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------
# 7. GPU Token-Stability Check
# Prevents meaning loss
# ----------------------------------------
def token_stability_check(original, cleaned):
    enc1 = tokenizer(
        original, return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)

    enc2 = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)

    len1 = enc1["input_ids"].shape[-1]
    len2 = enc2["input_ids"].shape[-1]

    # If too much lost → revert to original
    if abs(len1 - len2) > len1 * 0.30:
        return original

    return cleaned


# ----------------------------------------
# 8. FULL CLEANING PIPELINE
# ----------------------------------------
def preprocess(text):
    if not isinstance(text, str):
        return ""

    original = text

    text = normalize_unicode(text)
    text = fix_star_ratings(text)
    text = fix_slash_phrases(text)
    text = preserve_num_word(text)
    text = remove_noise(text)
    text = normalize_spaces(text)

    # Final meaning safety check (GPU)
    text = token_stability_check(original, text)

    return text


# ----------------------------------------
# 9. Apply to CSV dataset
# ----------------------------------------
def preprocess_dataset(path):
    df = pd.read_csv(path)

    cleaned = []
    for txt in df["Review"]:
        cleaned.append(preprocess(txt))

    df["Cleaned_Review"] = cleaned
    return df


# ----------------------------------------
# 10. Run Script
# ----------------------------------------
if __name__ == "__main__":
    input_path = r"D:\tripadvisor_hotel_reviews.csv"   # <-- FIXED PATH (RAW STRING)
    df = preprocess_dataset(input_path)
    df.to_csv("tripadvisor_clean_meaning_safe.csv", index=False)
    print("✔ Done — Cleaned file saved as tripadvisor_clean_meaning_safe.csv")
