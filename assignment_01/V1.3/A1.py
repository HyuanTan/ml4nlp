"""
============================================================
  File:        A1.py
  Description: Programming Assignment 1 Introduction to language modeling
               Implements all the classes and functions required by the full pipeline for
               a simple neural language model:
                 • Tokenization (Part 1)
                 • Dataset loading and batching (Part 2 
                                                -- functionalities are included in Part 4
                                                -- sanity checks maded seperately in A1_tests.py)
                 • Model definition (Part 3)
                 • Training loop and monitoring (Part 4)
                 • Evaluation and analysis (Part 5)
               
  Authors:  
      Yiming Li       <liyim@student.chalmers.se>
      Yajing Zhang    <yajingz@student.chalmers.se>
      Huoyuan Tan     <gushuota@student.gu.se>
      Yinghao Chen    <yinghao.chen@chalmers.se>

  Course:      DAT450 / DIT247 Machine learning for natural language processing
  Date:        2025-11-10

  Usage Notes:
  • All class and function sanity checks are implemented in the companion
    script `A1_tests.py`. Example usages can be found there.
  • The training section (Part 4) contains additional parameter tuning,
    which will be documented separately in an accompanying configuration file.
============================================================
"""


import json
import torch, nltk, pickle
import torch
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np
import sys, time, os, argparse
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from sklearn.decomposition import TruncatedSVD


# THIS IS ONLY FOR RESOLVING ERRORS:
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


###
### Part 1. Tokenization.
###

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
        """Initialize the tokenizer.

        Args:
            str_to_int: Mapping from token string to integer ID.
            int_to_str: Mapping from integer ID to token string.
            tokenize_fun: Function to split text into string tokens.
            model_max_length: Default max length for truncation.
            pad_token: String for the padding token.
            unk_token: String for the unknown token.
            bos_token: String for the beginning-of-sequence token.
            eos_token: String for the end-of-sequence token.
        """
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
          A BatchEncoding where the field `input_ids` stores the integer-encoded texts and !OPTIONAL BUT I IMPLEMENTED! `attention_mask` stores the corresponding attention mask.
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

    # 5. return a Tokenizer object
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
### Part 2. Loading the text files and creating batches.
###

"""Part 2 was already implemented and tested as part of Part 4’s code, and the sanity checks (A1_tests.py) will verify that everything works."""


###
### Part 3. Defining the model.
###


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""

    def __init__(
        self, vocab_size=None, embedding_size=None, hidden_size=None, rnn_type="LSTM", **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type.upper()


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

        rnn_type = getattr(config, "rnn_type", "LSTM").upper()
        print(f"Initializing A1RNNModel with RNN type: {rnn_type}")

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'LSTM', 'GRU', or 'RNN'.")


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


###
### Part 4. Training the language model.
###

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

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        # check if metric improved
        if metric > self.best - self.min_delta:
            self.num_bad_epochs += 1
        else:
            self.best = metric
            self.num_bad_epochs = 0

        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
            return True
        return False


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
        num_epochs = int(args.num_train_epochs)

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(args.learning_rate), weight_decay=0.01)


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
    
        # recomend：5% warmup
        warmup_ratio = 0.05
        # total_steps = epoch * steps_per_epoch
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # linear warmup: from 0 to lr
                return float(current_step) / float(max(1, warmup_steps))
            # warmup over, linear decay
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        history = {"train_loss": [], "val_loss": [], "perplexity": []}

        early_stopper = EarlyStopping(patience=3, min_delta=0.05)
        best_model_path = os.path.join(args.output_dir, "best_model.pt")

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
                scheduler.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_loss / max(1, num_batches)
            print(f"Train loss: {avg_train_loss:.4f}")
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
                
                # Check early stopping
                if early_stopper.step(avg_val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                # Save best model
                if avg_val_loss == early_stopper.best:
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Saved best model so far to {best_model_path}")

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
### Part 5. Evaluation and analysis.
###

def load_previous_model_and_tokenizer(model_dir: str, tokenizer_path: str):
    """Load a previously trained A1RNNModel and A1Tokenizer."""

    model = A1RNNModel.from_pretrained(model_dir)
    tokenizer = A1Tokenizer.from_file(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model and tokenizer loaded successfully on device: {device}")
    return model, tokenizer, device


def predict_next_word(model, tokenizer, device, prompts, save_path=None, top_k=5):
    """
    Generate next-word predictions for a list of text prompts using a trained language model.
    """
    model.eval()

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if save_path is None:
        save_path = os.path.join(current_dir, "part5_nextword.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = []

    with open(save_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            f.write(f"Prompt: {prompt}\n")

            enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
            input_ids = enc["input_ids"].to(device)

            with torch.no_grad():
                logits = model(input_ids)

            last_logits = logits[0, -2, :]

            greedy_id = int(torch.argmax(last_logits))
            greedy_word = tokenizer.int_to_str[greedy_id]

            top_vals, top_idx = torch.topk(last_logits, k=top_k)
            top_probs = torch.softmax(top_vals, dim=-1).tolist()
            top_words = [tokenizer.int_to_str[i] for i in top_idx.tolist()]

            print(f"Greedy next word: {greedy_word}")
            print("Top-5 candidates:")
            for rank, (w, p) in enumerate(zip(top_words, top_probs), start=1):
                print(f"  {rank}. {w} (p={p:.4f})")

            f.write(f"Greedy next word: {greedy_word}\n")
            f.write("Top candidates:\n")
            for rank, (w, p) in enumerate(zip(top_words, top_probs), start=1):
                f.write(f"  {rank}. {w} (p={p:.4f})\n")
            f.write("\n")

            results.append({
                "prompt": prompt,
                "greedy": greedy_word,
                "top_k": list(zip(top_words, top_probs))
            })

    print(f"\n Predictions saved to {save_path}")
    return results



def quantitative_evaluate(model, tokenizer, val_file: str, device, batch_size: int):
    """Compute the perplexity of a trained model on a validation dataset. """

    dataset = load_dataset("text", data_files={"val": val_file})["val"]
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss, count = 0.0, 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            X, Y = input_ids[:, :-1], input_ids[:, 1:]
            logits = model(X)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(1, count)
    perplexity = float(np.exp(avg_loss))

    print(f"Validation Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")
    return perplexity

def nearest_neighbors(emb, vocab, inv_vocab, word: str, n_neighbors: int = 5):
    """Find nearest neighbors of a given word in the learned embedding space. """

    if word not in vocab:
        print(f"'{word}' not found in vocabulary.")
        return []

    with torch.no_grad():
        test_emb = emb.weight[vocab[word]]
        sim_func = nn.CosineSimilarity(dim=1)
        cosine_scores = sim_func(test_emb, emb.weight)
        top_vals, top_idx = cosine_scores.topk(n_neighbors + 1)  # +1 to skip itself

        neighbors = [
            (inv_vocab[i.item()], top_vals[j + 1].item()) for j, i in enumerate(top_idx[1:])
        ]

    print(f"\n Nearest neighbors for '{word}':")
    for n, score in neighbors:
        print(f"  {n:15s} (cos={score:.4f})")
    return neighbors


def plot_embeddings_pca(emb, vocab, words, save_path=None):
    """Project embeddings of given words into 2D using PCA and plot them."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if save_path is None:
        save_path = os.path.join(current_dir, "pca_embeddings.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    valid_words = [w for w in words if w in vocab]
    if not valid_words:
        print("No valid words found in vocabulary for plotting.")
        return

    vectors = np.vstack([emb.weight[vocab[w]].cpu().detach().numpy() for w in valid_words])
    vectors -= vectors.mean(axis=0)
    twodim = TruncatedSVD(n_components=2).fit_transform(vectors)
    plt.figure(figsize=(5,5))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.02, y, word)
    plt.title("2D PCA projection of selected word embeddings")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PCA visualization to {save_path}")
