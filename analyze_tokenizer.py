from tokenizers import Tokenizer
import json
import os

def analyze_tokenizer():
    tokenizer_path = "otter_tokenizer_id_wiki_32k.json"

    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file '{tokenizer_path}' not found. Please run tokenizer.py first to train it.")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Sample Indonesian text (Wikipedia style)
    samples = [
        "Indonesia adalah negara kepulauan di Asia Tenggara yang dilintasi garis khatulistiwa.",
        "Bahasa Indonesia adalah bahasa resmi Republik Indonesia dan bahasa persatuan bangsa Indonesia.",
        "Perekonomian Indonesia merupakan salah satu kekuatan ekonomi baru dunia yang sedang berkembang.",
        "Kemerdekaan Indonesia diproklamasikan pada tanggal 17 Agustus 1945.",
        "Masalah utama yang dihadapi adalah tingginya tingkat kemiskinan dan ketimpangan sosial."
    ]

    print("\n--- Tokenization Analysis ---")

    total_tokens = 0
    total_words = 0

    for text in samples:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        ids = encoding.ids

        # Word count (approximation by splitting space)
        words = len(text.split())
        num_tokens = len(tokens)
        fertility = num_tokens / words

        total_tokens += num_tokens
        total_words += words

        print(f"\nText: {text[:60]}...")
        print(f"Words: {words}, Tokens: {num_tokens}")
        print(f"Fertility (Tokens/Word): {fertility:.2f}")
        print(f"Tokens: {tokens}")

    avg_fertility = total_tokens / total_words
    print(f"\n--- Summary ---")
    print(f"Average Fertility Rate: {avg_fertility:.2f}")

    print("\n--- Interpretation ---")
    if avg_fertility < 1.3:
        print("✅ Excellent! The tokenizer is very efficient (words are mostly single tokens).")
    elif avg_fertility < 1.6:
        print("✓ Good. Typical for subword tokenization.")
    else:
        print("⚠️ High fertility. Many words are split into small pieces. Consider increasing vocab size.")

if __name__ == "__main__":
    analyze_tokenizer()
