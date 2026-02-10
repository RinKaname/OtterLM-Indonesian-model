from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

# 1. Load the SPECIFIC Indonesian subset of FineWiki (Aug 2025)
dataset = load_dataset(
    "HuggingFaceFW/finewiki", 
    name="id", 
    split="train", 
    streaming=True
).shuffle(buffer_size=10_000, seed=42)

def batch_iterator(batch_size=1000, max_samples=700_000):
    batch = []
    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        # FineWiki stores the structured article in the 'text' field
        text = ex.get("text", "")
        if text and text.strip():
            batch.append(text + "</s>")
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# 2. INIT TOKENIZER
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# 3. Define Special Tokens (ID 0, 1, 2)
special_tokens = ["<|unk|>", "</s>", "<pad>"]
tokenizer.add_special_tokens(special_tokens)

# 4. Trainer
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=5, # Lowered to catch Indonesian root affixes
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True
)

# 5. TRAIN
print("🚀 Training OtterTokenizer on FineWiki Indonesian (Aug 2025)...")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# 6. Save
tokenizer.save("otter_tokenizer_id_wiki_32k.json")
print("\n✅ Indonesian Wikipedia Tokenizer trained and saved!")
