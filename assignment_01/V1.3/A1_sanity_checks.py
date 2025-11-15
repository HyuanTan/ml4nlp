"""
============================================================
  File:        A1_tests.py
  Description: Programming Assignment 1 Sanity Checks and Evaluation
               This script performs unit and integration tests for all
               implemented components from A1.py of the RNN-based language model.
               
  Authors:  
      Yiming Li       <liyim@student.chalmers.se>
      Yajing Zhang    <yajingz@student.chalmers.se>
      Huoyuan Tan     <gushuota@student.gu.se>
      Yinghao Chen    <yinghao.chen@chalmers.se>

  Course:      DAT450 / DIT247 Machine learning for natural language processing
  Date:        2025-11-10

  Usage Notes:
  • To skip time-consuming training during testing, set:
        enable_part4 = False
============================================================
"""


import os, argparse
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import BatchEncoding

# Modules for testing
from A1 import A1Tokenizer, build_tokenizer
from A1 import A1RNNModelConfig, A1RNNModel
from A1 import plot_training_monitor, EarlyStopping, A1Trainer
from A1 import load_previous_model_and_tokenizer, predict_next_word, quantitative_evaluate, nearest_neighbors, plot_embeddings_pca


enable_part4 = False #Skip testing training and directly evaluate a pre-trained model

print("\n\n--- Running Sanity Checks with Dummy Data ---")
# dummy files
DUMMY_TRAIN_FILE = "dummy_train.txt"
DUMMY_VAL_FILE = "dummy_val.txt"
DUMMY_MAX_VOCAB = 15  # 4 special + 11 words
DUMMY_MODEL_MAX_LEN = 10

with open(DUMMY_TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("the quick brown fox jumps over the lazy dog .\n")
    f.write("a test sentence with the and and .\n")
    f.write("another test , with punctuation !\n")
    f.write("Rare words like cuboidal and epiglottis .\n")
    f.write("the the the the and and fox .\n")  # frequency
    f.write("\n")  # empty line
    f.write("This is a test .\n")
    f.write("Another test .\n")

with open(DUMMY_VAL_FILE, "w", encoding="utf-8") as f:
    f.write("This is a validation sentence .\n")
    f.write("\n")  # empty line
    f.write("Another validation sentence .\n")

#
# --- Part 1: Tokenization ---
#

print("\n--- Part 1: Tokenization START ---")
print("\n--- Part 1.1: Vocabulary START ---")

# tokenizer
tokenizer = build_tokenizer(
    DUMMY_TRAIN_FILE, max_voc_size=DUMMY_MAX_VOCAB, model_max_length=DUMMY_MODEL_MAX_LEN
)

# 1. Check size
print(f"[Check 1.1_1] Vocab size <= {DUMMY_MAX_VOCAB}?")
assert len(tokenizer) <= DUMMY_MAX_VOCAB
print(f"PASS. Size is {len(tokenizer)}.")

# 2. Check special symbols
print("\n[Check 1.1_2] Special symbols exist and are correct?")
assert tokenizer.str_to_int[tokenizer.bos_token] == 0
assert tokenizer.str_to_int[tokenizer.eos_token] == 1
assert tokenizer.str_to_int[tokenizer.unk_token] == 2
assert tokenizer.str_to_int[tokenizer.pad_token] == 3
assert tokenizer.pad_token_id == 3
print(f"PASS. Special tokens mapped: BOS:0, EOS:1, UNK:2, PAD:3.")

# 3. Check frequent/rare words
print("\n[Check 1.1_3] Frequent/Rare words included/excluded?")
# Freq: 'the' (6), '.' (5), 'and' (4), 'test' (3), 'fox' (2)
assert "the" in tokenizer.str_to_int
assert "and" in tokenizer.str_to_int
assert "." in tokenizer.str_to_int
print("PASS. Frequent words ('the', 'and', '.') are in vocab.")

# 'cuboidal' and 'epiglottis' are rare (1 count) and should be excluded
# by DUMMY_MAX_VOCAB=15
assert "cuboidal" not in tokenizer.str_to_int
assert "epiglottis" not in tokenizer.str_to_int
print("PASS. Rare words ('cuboidal', 'epiglottis') are not in vocab.")

# 4. Check round-trip mapping
print("\n[Check 1.1_4] Round-trip mapping?")
test_word_in = "the"
test_id_in = tokenizer.str_to_int[test_word_in]
assert tokenizer.int_to_str[test_id_in] == test_word_in
print(
    f"PASS. In-vocab: '{test_word_in}' -> {test_id_in} -> '{tokenizer.int_to_str[test_id_in]}'"
)

test_word_out = "cuboidal"
test_id_out = tokenizer.str_to_int.get(test_word_out, tokenizer.unk_token_id)
assert test_id_out == tokenizer.unk_token_id
assert tokenizer.int_to_str[test_id_out] == tokenizer.unk_token
print(
    f"PASS. Out-of-vocab: '{test_word_out}' -> {test_id_out} -> '{tokenizer.int_to_str[test_id_out]}'"
)

print("\n--- Part 1.1 Checks Passed ---")

print("\n\n--- Part 1.2: Tokenizer Functionality START ---")

test_texts = ["This is a test.", "Another test."]
print(f"Input texts: {test_texts}")

# Tokenize with padding, truncation, and tensors
batch_encoding = tokenizer(
    test_texts, return_tensors="pt", padding=True, truncation=True
)

print("\n[Check 1.2_1] Tokenizer output:")
print(f"Type: {type(batch_encoding)}")
print(f"Keys: {batch_encoding.keys()}")

print("\n'input_ids':")
print(batch_encoding["input_ids"])

print("\n'attention_mask':")
print(batch_encoding["attention_mask"])

# Verify padding
# 'This is a test.' -> ['this', 'is', 'a', 'test', '.'] -> [BOS, ..., EOS] (7 tokens)
# 'Another test.'   -> ['another', 'test', '.']     -> [BOS, ..., EOS] (5 tokens)
# Padded to 7.
expected_shape = torch.Size([2, 7])
assert batch_encoding["input_ids"].shape == expected_shape
assert batch_encoding["attention_mask"].shape == expected_shape
print(f"\nPASS. Output shape is {expected_shape}.")

# Check that the second sentence is padded
assert batch_encoding["input_ids"][1, 5] == tokenizer.pad_token_id
assert batch_encoding["input_ids"][1, 6] == tokenizer.pad_token_id
assert batch_encoding["attention_mask"][1, 5] == 0
assert batch_encoding["attention_mask"][1, 6] == 0
# Check that the second sentence's real tokens are correct
assert batch_encoding["input_ids"][1, 4] == tokenizer.eos_token_id
assert batch_encoding["attention_mask"][1, 4] == 1
print("PASS. Padding and attention mask are correct.")

# Check truncation (using DUMMY_MODEL_MAX_LEN = 10)
long_text = (
    "the the the the the the the the the the the the"  # 12 tokens + BOS/EOS = 14
)
trunc_encoding = tokenizer(long_text, return_tensors="pt", truncation=True)
assert trunc_encoding["input_ids"].shape[1] == DUMMY_MODEL_MAX_LEN
assert trunc_encoding["input_ids"][0, -1] == tokenizer.eos_token_id  # Ends with EOS
print("PASS. Truncation to model_max_length works.")

# Check save and load
print("\n[Check 1.2_2] Save and Load:")
DUMMY_TOKENIZER_FILE = "a1_dummy_tokenizer.pkl"
tokenizer.save(DUMMY_TOKENIZER_FILE)
loaded_tokenizer = A1Tokenizer.from_file(DUMMY_TOKENIZER_FILE)

assert len(tokenizer) == len(loaded_tokenizer)
assert tokenizer.pad_token_id == loaded_tokenizer.pad_token_id
print("PASS. Loaded tokenizer has same vocab size and pad ID.")

print("\n--- Part 1.2 Checks Passed ---")
print("\n--- Part 1 Checks Passed ---")


#
# --- Part 2: Loading the text files and creating batches ---
#

print("\n\n--- Part 2: Loading the text files and creating batches ---")
print("\n\n--- Part 2.1: Dataset Loading START ---")

print("Loading datasets...")
dataset = load_dataset(
    "text", data_files={"train": DUMMY_TRAIN_FILE, "val": DUMMY_VAL_FILE}
)
print(
    f"Original dataset sizes: Train={len(dataset['train'])}, Val={len(dataset['val'])}"
)

# remove empty lines
print("Filtering empty lines...")
dataset = dataset.filter(lambda x: x["text"].strip() != "")

print(f"\n[Check 2.1_1] Filtered dataset sizes:")
print(f"Train: {len(dataset['train'])}")
print(f"Val: {len(dataset['val'])}")

assert len(dataset["train"]) == 7
assert len(dataset["val"]) == 2

# optional: subsetting
print("\n[Check 2.1_2] Subsetting dataset (optional step):")
for sec in ["train", "val"]:
    dataset[sec] = Subset(dataset[sec], range(2))  # Take first 2 of each

print(f"Subset sizes: Train={len(dataset['train'])}, Val={len(dataset['val'])}")
assert len(dataset["train"]) == 2
assert len(dataset["val"]) == 2
print("PASS. Subsetting works.")

print("\n--- Part 2.1 Checks Passed ---")

print("\n\n--- Part 2.2: DataLoader START ---")

# A collate function that uses our tokenizer
def collate_fn(batch):
    # 'batch' is a list of dicts, e.g., [{'text': '...'}, {'text': '...'}]
    # Need to extract the text strings first
    texts = [item["text"] for item in batch]

    # Use the tokenizer to pad, truncate, and create tensors
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

print("Creating DataLoader with batch_size=2...")
# Use the 'train' subset which has 2 items
dl = DataLoader(
    dataset["train"], batch_size=2, shuffle=False, collate_fn=collate_fn
)

print("\n[Check 2.2_1] Fetching first batch:")
# Get the first (and only) batch
first_batch = next(iter(dl))

print(first_batch)

# Confirm it corresponds to expectations
assert isinstance(first_batch, BatchEncoding)
assert "input_ids" in first_batch
assert "attention_mask" in first_batch
assert isinstance(first_batch["input_ids"], torch.Tensor)
assert first_batch["input_ids"].shape[0] == 2  # batch_size=2
print(
    "\nPASS. First batch fetched successfully and is a BatchEncoding with correct tensors."
)

print("\n--- Part 2.2 Checks Passed ---")


#
# --- Part 3: Defining the language model neural network ---
#

print("\n--- Part 3: RNN Model START ---")
# Create simple batch
test_texts = ["This is a test.", "Another test."]
enc = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc["input_ids"]

# Instantiate model
config = A1RNNModelConfig(
    vocab_size=len(tokenizer), embedding_size=16, hidden_size=32
)
model = A1RNNModel(config)

# Prepare X (all but last token) and Y (all but first token)
X = input_ids[:, :-1]
Y = input_ids[:, 1:]

logits = model(X)
print("Model output shape:", logits.shape)

# Shape checks
assert logits.shape[0] == X.shape[0]
assert logits.shape[1] == X.shape[1]
assert logits.shape[2] == len(tokenizer)

# Simple loss check to ensure end-to-end differentiability
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
print("Loss:", float(loss.item()))
assert torch.isfinite(loss)
print("\n--- Part 3 PASS. Model forward and loss computation work. ---")

#
# --- Part 4: Training the model ---
#
if enable_part4:
    # parse optional CLI args for training (keeps sensible defaults)
    parser = argparse.ArgumentParser(description="Train A1 RNN language model")
    parser.add_argument("--use_cpu", action="store_true", help="Force use CPU")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="a1_output")
    parser.add_argument("--rnn_type", type=str, default="LSTM", help="RNN type: LSTM, GRU, or RNN")
    cli_args, _ = parser.parse_known_args()

    # build args namespace expected by A1Trainer
    args = argparse.Namespace(
        optim='adamw_torch',
        eval_strategy='epoch',
        use_cpu=cli_args.use_cpu,
        no_cuda=cli_args.no_cuda,
        learning_rate=cli_args.learning_rate,
        num_train_epochs=cli_args.num_train_epochs,
        per_device_train_batch_size=cli_args.per_device_train_batch_size,
        per_device_eval_batch_size=cli_args.per_device_eval_batch_size,
        output_dir=cli_args.output_dir,
    )

    print("\n--- Part 4: Training the model START ---")
    if True:
        TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
        VAL_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
        MAX_VOCAB = 50000 # size for vocabulary
        MODEL_MAX_LEN = 256 # 128 # length for sentence truncation
        TOKENIZER_FILE = "a1_tokenizer.pkl"

        EMBEDDING_SIZE = 256 # 128
        HIDDEN_SIZE = 512 # 256
    else:
        TRAIN_FILE = "test_data/train.txt"
        VAL_FILE = "test_data/val.txt"
        MAX_VOCAB = 500 # size for vocabulary
        MODEL_MAX_LEN = 128 # length for sentence truncation
        TOKENIZER_FILE = "a1_tokenizer_test.pkl"
        EMBEDDING_SIZE = 128
        HIDDEN_SIZE = 256

    # tokenizer
    if not os.path.exists(TOKENIZER_FILE):
        tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=MAX_VOCAB, model_max_length=MODEL_MAX_LEN)
        tokenizer.save(TOKENIZER_FILE)
    else:
        tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)

    # --- Prepare datasets for training ---
    print("Loading full datasets for training...")
    full_dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "val": VAL_FILE})
    # filter empty lines
    full_dataset = full_dataset.filter(lambda x: x["text"].strip() != "")

    # --- Build model, args and trainer ---
    config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, rnn_type=cli_args.rnn_type)
    model = A1RNNModel(config)

    train_ds = full_dataset['train']
    val_ds = full_dataset['val']

    args.output_dir = f"{cli_args.output_dir}_{config.rnn_type}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    trainer = A1Trainer(model, args, train_ds, val_ds, tokenizer)
    trained_model, history = trainer.train()

    # final plots (redundant with per-epoch updates but useful)
    try:
        plot_training_monitor(history, args.output_dir)
    except Exception as e:
        print(f"Warning: could not save final training plots: {e}")

    print("\n--- Part 4: Finish ---")



#
# --- Part 5: Evaluation and analysis ---
#
print("\n--- Part 5: Evaluation and analysis START ---")

print("Load pre-trained model...")
MODEL_DIR = "V1.2/a1_output_LSTM"
TOKENIZER_FILE = f"{MODEL_DIR}/a1_tokenizer.pkl"
model, tokenizer, device = load_previous_model_and_tokenizer(model_dir=MODEL_DIR, tokenizer_path=TOKENIZER_FILE)


# 1. Predicting the next word
print("\n[Check 5.1] Predicting the next word...")
prompts = [
    "She lives in San",
    "The capital of France is",
    "Deep learning models are trained with",
]
try:
    results = predict_next_word(model, tokenizer, device, prompts)
    assert len(results) == len(prompts)
    print("predict_next_word() ran successfully.")
except Exception as e:
    print(f"predict_next_word() failed: {e}")
    raise

# 2. Quantitative evaluation (perplexity)
print("\n[Check 5.2] Evaluating perplexity on dummy validation set...")
VAL_FILE = "dummy_val.txt"
try:
    ppl = quantitative_evaluate(model, tokenizer, val_file=VAL_FILE, device=device, batch_size=64)
    assert ppl > 0, "Perplexity must be positive."
    print(f"Perplexity computation successful: {ppl:.2f}")
except Exception as e:
    print(f"Perplexity evaluation failed: {e}")
    raise

# 3.1. Inspect nearest neighbors in embedding space
print("\n[Check 5.3_1] Inspecting nearest neighbors...")
test_words = ["sweden", "king", "city", "woman", "love"]
embedding = model.embedding
vocab = tokenizer.str_to_int
inv_vocab = tokenizer.int_to_str
try:
    for w in test_words:
        nn_list = nearest_neighbors(embedding, vocab, inv_vocab, w, n_neighbors=5)
        if nn_list:
            print(f"\nTop neighbors for '{w}':")
            for neigh, score in nn_list:
                print(f"  {neigh:15s} (cos={score:.4f})")
    print("Nearest neighbor inspection successful.")
except Exception as e:
    print(f"Nearest neighbor check failed: {e}")
    raise

# 3.2. PCA Visualization
print("\n[Check 5.3_2] PCA visualization of embeddings...")
try:
    plot_words = ["sweden", "denmark", "norway", "europe", "africa",
                  "london", "stockholm", "paris", "small", "large",
                  "good", "bad", "man", "woman", "child",
                  "seven", "ten", "hundred", "2005", "1984"]
    plot_embeddings_pca(model.embedding, vocab, plot_words)
    print("PCA embedding plot saved successfully.")
except Exception as e:
    print(f"PCA embedding visualization failed: {e}")
    raise

print("\n--- Part 5 PASS. Model evaluation functions work. ---")


# --- Cleanup ---
print("\n\nCleaning up dummy files...")
os.remove(DUMMY_TRAIN_FILE)
os.remove(DUMMY_VAL_FILE)
os.remove(DUMMY_TOKENIZER_FILE)
print("Done.")
