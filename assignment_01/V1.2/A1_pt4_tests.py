import json
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np
import sys, time, os, argparse
import matplotlib.pyplot as plt

# check NLTK 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("NLTK 'punkt' model not found. Downloading...")
    nltk.download("punkt", quiet=True)

# check NLTK 'punkt_tab' is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("NLTK 'punkt_tab' model not found. Downloading...")
    nltk.download("punkt_tab", quiet=True)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(
        self,
        str_to_int,
        int_to_str,
        tokenize_fun,
        model_max_length,
        pad_token,
        unk_token,
        bos_token,
        eos_token,
    ):
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun

        # special token strings
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # special token IDs
        self.pad_token_id = self.str_to_int[self.pad_token]
        self.unk_token_id = self.str_to_int[self.unk_token]
        self.bos_token_id = self.str_to_int[self.bos_token]
        self.eos_token_id = self.str_to_int[self.eos_token]

        # set max length
        self.model_max_length = int(model_max_length)

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

        Args:
          texts:           The texts to tokenize.
          truncation:      Whether the texts should be truncated to model_max_length.
          padding:         Whether the tokenized texts should be padded on the right side.
          return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

        Returns:
          A BatchEncoding where the field `input_ids` stores the integer-encoded texts and `attention_mask` stores the corresponding attention mask.
        """
        if return_tensors and return_tensors != "pt":
            raise ValueError("return_tensors must be None or 'pt'")

        if isinstance(texts, str):  # if single string input
            texts = [texts]

        # check max length
        effective_max_length = self.model_max_length
        if truncation and not effective_max_length:
            raise ValueError("Truncation set to True, but no max_length provided.")

        # 1. encode all texts
        all_input_ids = []
        for text in texts:
            # split word
            tokens = self.tokenize_fun(text)

            # convert to IDs (also handles unknown tokens)
            ids = [self.str_to_int.get(token, self.unk_token_id) for token in tokens]

            # add special tokens
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

            # apply truncation if needed
            if truncation and effective_max_length and len(ids) > effective_max_length:
                ids = ids[:effective_max_length]
                # make sure the last token is EOS if we truncate it off
                if ids[-1] != self.eos_token_id:
                    ids[-1] = self.eos_token_id

            all_input_ids.append(ids)

        # 2. padding
        batch_input_ids = []
        batch_attention_mask = []

        if not padding:
            batch_input_ids = all_input_ids
        else:
            # find the longest sequence in the batch
            max_len_in_batch = max(len(ids) for ids in all_input_ids)

            # apply padding and create attention masks
            for ids in all_input_ids:
                pad_len = max_len_in_batch - len(ids)

                padded_ids = ids + [self.pad_token_id] * pad_len
                attention_mask = [1] * len(ids) + [0] * pad_len

                batch_input_ids.append(padded_ids)
                batch_attention_mask.append(attention_mask)

        # 3. convert to tensors if needed
        if return_tensors == "pt":
            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            if padding:
                batch_attention_mask = torch.tensor(
                    batch_attention_mask, dtype=torch.long
                )

        # 4. return BatchEncoding
        batch_encoding_dict = {"input_ids": batch_input_ids}
        if padding:
            batch_encoding_dict["attention_mask"] = batch_attention_mask

        return BatchEncoding(batch_encoding_dict)

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, "rb") as f:
            return pickle.load(f)


def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]
def build_tokenizer(
    train_file,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=None,
    model_max_length=None,
    pad_token="<PAD>",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
):
    """Build a tokenizer from the given file.

    Args:
         train_file:        The name of the file containing the training texts.
         tokenize_fun:      The function that maps a text to a list of string tokens.
         max_voc_size:      The maximally allowed size of the vocabulary.
         model_max_length:  Truncate texts longer than this length.
         pad_token:         The dummy string corresponding to padding.
         unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
         bos_token:         The dummy string corresponding to the beginning of the text.
         eos_token:         The dummy string corresponding to the end the text.
    """
    print(f"Building vocabulary from {train_file}...")

    # 1. go thru file and get token frequencies
    counter = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # empty lines
                tokens = tokenize_fun(line)
                counter.update(tokens)

    print(f"Found {len(counter)} unique raw tokens.")

    # 2. create vocabulary mappings
    str_to_int = {}
    int_to_str = {}

    special_tokens = [bos_token, eos_token, unk_token, pad_token]
    for i, token in enumerate(special_tokens):  # add special symbols
        str_to_int[token] = i
        int_to_str[i] = token

    # 3. voc_size truncation (at most (max_voc_size - 4) words)
    num_words_to_keep = -1  # default = keep all
    if max_voc_size is not None:
        num_words_to_keep = max_voc_size - len(special_tokens)

        # handle error
        if num_words_to_keep < 0:
            print(
                f"Warning: max_voc_size ({max_voc_size}) is smaller than number of special tokens ({len(special_tokens)})."
            )
            num_words_to_keep = 0

    # get most common words
    if num_words_to_keep != -1:
        # truncate to n most common tokens
        most_common_word_freqs = counter.most_common(num_words_to_keep)
        most_common_words = [word for word, freq in most_common_word_freqs]
        print(
            f"Truncating vocabulary to {num_words_to_keep} most common words + {len(special_tokens)} special tokens."
        )
    else:
        # keep all unique tokens
        most_common_words = list(counter.keys())
        print(
            f"Keeping all {len(most_common_words)} unique words + {len(special_tokens)} special tokens."
        )

    # 4. add og words
    current_index = len(special_tokens)  # start from index 4
    for word in most_common_words:
        # make sure words donot overwrite special token strings
        if word not in str_to_int:
            str_to_int[word] = current_index
            int_to_str[current_index] = word
            current_index += 1

    print(f"Final vocabulary size: {len(str_to_int)}")

    # 5. return a Tokenizer object (use the top-level A1Tokenizer so it can be pickled)
    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )


###
### Part 3. Defining the model.
###


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""

    def __init__(
        self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size


class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.vocab_size is not None
            and config.embedding_size is not None
            and config.hidden_size is not None
        )

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_size
        )
        self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True,
        )
        self.unembedding = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size
        )

    def forward(self, X):
        """The forward pass of the RNN-based language model.

        Args:
          X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
        Returns:
          The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)
        return out


def plot_training_monitor(history, output_dir, epoch=None):
    """Plot training/validation loss and perplexity per epoch and save PNG files.

    Args:
        history: dict with keys 'train_loss', optional 'val_loss', optional 'perplexity'
        output_dir: directory to save plots
        epoch: optional int, current epoch number (used for per-epoch filenames)
    """
    os.makedirs(output_dir, exist_ok=True)

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    perps = history.get("perplexity", [])

    epochs = list(range(1, len(train_losses) + 1))

    # Loss plot
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="train_loss", marker="o")
        if val_losses:
            plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label="val_loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.grid(True)
        plt.legend()
        fname = os.path.join(output_dir, f"loss_plot{('_ep'+str(epoch)) if epoch else ''}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Warning saving loss plot: {e}")

    # Perplexity plot (if available)
    if perps:
        try:
            plt.figure()
            plt.plot(list(range(1, len(perps) + 1)), perps, label="perplexity", marker="o", color="C3")
            plt.xlabel("Epoch")
            plt.ylabel("Perplexity")
            plt.title("Validation Perplexity per Epoch")
            plt.grid(True)
            plt.legend()
            fname = os.path.join(output_dir, f"perplexity_plot{('_ep'+str(epoch)) if epoch else ''}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Warning saving perplexity plot: {e}")

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.

           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(args.learning_rate))

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        # collate function for DataLoader: uses tokenizer to create padded tensors
        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            # return PyTorch tensors
            return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        train_batch_size = int(args.per_device_train_batch_size)
        eval_batch_size = int(args.per_device_eval_batch_size)

        # pin_memory is beneficial when using CUDA
        pin_memory = (not args.use_cpu) and (not args.no_cuda) and torch.cuda.is_available()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        num_epochs = int(args.num_train_epochs)
        history = {"train_loss": [], "val_loss": [], "perplexity": []}

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_tokens = 0
            num_batches = 0

            print(f"Epoch {epoch+1}/{num_epochs}")

            for batch in train_loader:
                # batch is a BatchEncoding with tensors
                input_ids = batch["input_ids"].to(device)

                # X: all tokens except last, Y: all tokens except first
                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]

                logits = self.model(X)

                loss = loss_func(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_loss / max(1, num_batches)
            print(f"  Train loss: {avg_train_loss:.4f}")
            history["train_loss"].append(avg_train_loss)

            # Evaluation after each epoch (args.eval_strategy is assumed to be 'epoch')
            if args.eval_strategy == "epoch":
                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        X = input_ids[:, :-1]
                        Y = input_ids[:, 1:]

                        logits = self.model(X)
                        val_loss = loss_func(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))

                        val_loss_sum += float(val_loss.item())
                        val_batches += 1

                avg_val_loss = val_loss_sum / max(1, val_batches)
                try:
                    perplexity = float(np.exp(avg_val_loss))
                except OverflowError:
                    perplexity = float("inf")
                print(f"  Val loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
                history["val_loss"].append(avg_val_loss)
                history["perplexity"].append(perplexity)
                # update realtime plots for monitoring
                try:
                    plot_training_monitor(history, args.output_dir, epoch+1)
                except Exception as e:
                    print(f"Warning: could not update training monitor: {e}")

        # ensure output dir exists
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving to {args.output_dir}.")
        self.model.save_pretrained(args.output_dir)

        # Save training history
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f)
        return self.model, history

###
### Run sanity checks
###
if __name__ == "__main__":
    enable_part1_to3 = False
    enable_part4 = True

    if enable_part1_to3:

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


        # --- Cleanup ---
        print("\n\nCleaning up dummy files...")
        os.remove(DUMMY_TRAIN_FILE)
        os.remove(DUMMY_VAL_FILE)
        os.remove(DUMMY_TOKENIZER_FILE)
        print("Done.")


    if enable_part4:
        # --- Part 4: Training the model ---
        print("\n--- Part 4: Training the model START ---")
        if True:
            TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
            VAL_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
            MAX_VOCAB = 50000 # size for vocabulary
            MODEL_MAX_LEN = 128 # length for sentence truncation
            TOKENIZER_FILE = "a1_tokenizer.pkl"

            EMBEDDING_SIZE = 128
            HIDDEN_SIZE = 256
        else:
            TRAIN_FILE = "test_data/train.txt"
            VAL_FILE = "test_data/val.txt"
            MAX_VOCAB = 500 # size for vocabulary
            MODEL_MAX_LEN = 128 # length for sentence truncation
            TOKENIZER_FILE = "a1_tokenizer_test.pkl"

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
        config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
        model = A1RNNModel(config)

        # parse optional CLI args for training (keeps sensible defaults)
        parser = argparse.ArgumentParser(description="Train A1 RNN language model")
        parser.add_argument("--use_cpu", action="store_true", help="Force use CPU")
        parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--num_train_epochs", type=int, default=3)
        parser.add_argument("--per_device_train_batch_size", type=int, default=64)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
        parser.add_argument("--output_dir", type=str, default="a1_output")

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

        # optionally subset datasets for quick runs
        train_ds = full_dataset['train']
        val_ds = full_dataset['val']

        os.makedirs(args.output_dir, exist_ok=True)

        trainer = A1Trainer(model, args, train_ds, val_ds, tokenizer)
        trained_model, history = trainer.train()

        # final plots (redundant with per-epoch updates but useful)
        try:
            plot_training_monitor(history, args.output_dir)
        except Exception as e:
            print(f"Warning: could not save final training plots: {e}")

        print("\n--- Part 4: Finish ---")


    # print("\n--- Part 5: Evaluation and analysis START ---")

